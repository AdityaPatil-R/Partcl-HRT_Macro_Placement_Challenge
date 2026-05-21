"""
Xplace Placer — uses real Xplace (CUHK-EDA) as cong-aware gradient frontend.

Approach:
  1. Convert benchmark to Bookshelf format files (reuses dreamplace_placer's writer).
  2. Invoke Xplace via subprocess with --custom_path for bookshelf input.
  3. Read back optimized placement from Xplace's output .pl.
  4. Refine with iterative CD + LNS + swap pipeline (cong_w=0.5, true_proxy gated).

Why Xplace over DREAMPlace: DREAMPlace optimizes HPWL + electrostatic density only,
which is misaligned with our proxy (WL + 0.5*den + 0.5*cong; cong dominates).
Xplace adds RUDY-based routing congestion to its gradient via --congest_weight.

Env vars:
  XP_TOTAL_BUDGET   = total wall-clock seconds (default 2000)
  XP_GRAD_BUDGET    = seconds for Xplace global placement (default 300)
  XP_CONGEST_WEIGHT = Xplace cong weight (default 1.0 — non-zero unlike DREAMPlace)
  XP_INNER_ITER     = Xplace gradient iterations (default 2000)
  XP_VERBOSE        = 1 for verbose logging
  XP_MAX_REFINE     = max outer refinement iterations (default 8)
"""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

# DREAMPlace v4.1.0 PlaceDB.py uses removed np.str/np.bool aliases. Patch
# before importing anything else.
for _name, _typ in (("str", str), ("bool", bool), ("int", int),
                     ("float", float), ("object", object), ("long", int)):
    if not hasattr(np, _name):
        setattr(np, _name, _typ)

from macro_place.benchmark import Benchmark

_HERE = Path(__file__).resolve().parent
_SUBMISSIONS = _HERE.parent

# Reuse lns_placer infrastructure for legalization + LNS refinement.
_LNS_FILE = _SUBMISSIONS / "lns_placer" / "placer.py"
_spec = importlib.util.spec_from_file_location("_lns_for_xp", str(_LNS_FILE))
_lns = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lns)

# Reuse the bookshelf writer from dreamplace_placer — identical input format.
sys.path.insert(0, str(_SUBMISSIONS / "dreamplace_placer"))
from bookshelf_writer import write_bookshelf, SCALE as XP_SCALE  # noqa: E402


def _ef(name, default):
    try: return float(os.environ.get(name, default))
    except Exception: return default
def _ei(name, default):
    try: return int(os.environ.get(name, default))
    except Exception: return default
def _es(name, default):
    return os.environ.get(name, default)

XP_TOTAL_BUDGET   = _ef("XP_TOTAL_BUDGET", 2000.0)
XP_GRAD_BUDGET    = _ef("XP_GRAD_BUDGET", 300.0)
XP_CONGEST_WEIGHT = _ef("XP_CONGEST_WEIGHT", 1.0)
XP_INNER_ITER     = _ei("XP_INNER_ITER", 2000)
XP_VERBOSE        = _ei("XP_VERBOSE", 1)
XP_HOME           = _es("XP_HOME", "/opt/Xplace")


def _rewrite_scl_for_xplace(scl_path, benchmark, scale):
    """Replace the single-giant-row .scl that bookshelf_writer emits with a
    multi-row .scl that satisfies Xplace's requirement
    `rowHeight == siteHeight`. Xplace computes siteHeight as GCD of all node
    heights; we mirror that and emit ceil(canvas_h / siteHeight) rows of
    matching height that tile the canvas."""
    from math import gcd
    from functools import reduce
    sizes = benchmark.macro_sizes.numpy()
    n_macros = sizes.shape[0]
    heights_dbu = []
    for i in range(n_macros):
        h = max(1, int(round(float(sizes[i, 1]) * scale)))
        heights_dbu.append(h)
    if benchmark.port_positions.numel():
        heights_dbu.append(1)  # ports are 1x1 in our writer
    site_height = reduce(gcd, heights_dbu) if heights_dbu else 1
    site_height = max(1, site_height)

    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    n_sites_x = max(1, int(round(cw * scale)))
    canvas_h_dbu = max(1, int(round(ch * scale)))
    n_rows = max(1, canvas_h_dbu // site_height)

    with open(scl_path, "w") as f:
        f.write("UCLA scl 1.0\n\n")
        f.write(f"NumRows : {n_rows}\n\n")
        for i in range(n_rows):
            y = i * site_height
            f.write("CoreRow Horizontal\n")
            f.write(f"  Coordinate    :   {y}\n")
            f.write(f"  Height        :   {site_height}\n")
            f.write("  Sitewidth     :   1\n")
            f.write("  Sitespacing   :   1\n")
            f.write("  Siteorient    :   N\n")
            f.write("  Sitesymmetry  :   Y\n")
            f.write(f"  SubrowOrigin  :   0\tNumSites :   {n_sites_x}\n")
            f.write("End\n")
    if XP_VERBOSE:
        print(f"[XP] rewrote .scl: siteHeight={site_height}, {n_rows} rows, "
              f"{n_sites_x} sites/row (canvas={cw:.1f}x{ch:.1f} um)")


def run_xplace(benchmark, init_positions, time_budget_s):
    """Invoke Xplace via subprocess. Returns (positions_np, success)."""
    n_hard = benchmark.num_hard_macros
    sizes = benchmark.macro_sizes.numpy()

    work_dir = Path(tempfile.mkdtemp(prefix="xp_"))
    try:
        plc_obj = _lns._load_plc(benchmark.name)
        aux_path = write_bookshelf(work_dir, benchmark.name, benchmark, plc_obj, init_positions)
        # Overwrite the .scl that bookshelf_writer produced (single giant row)
        # with a multi-row format Xplace accepts.
        _rewrite_scl_for_xplace(work_dir / f"{benchmark.name}.scl", benchmark, XP_SCALE)
        result_dir = work_dir / "result"
        result_dir.mkdir(exist_ok=True)

        # Xplace CLI invocation. Use --custom_path with bookshelf variety.
        # --mixed_size True is critical: tells Xplace to treat large nodes
        # (our hard macros) as movable macros, not just standard cells.
        cmd = [
            "python3", "main.py",
            "--custom_path",
            f"benchmark:ispd2005,bookshelf_variety:ispd2005,design_name:{benchmark.name},aux:{aux_path}",
            "--global_placement", "True",
            "--legalization", "False",       # we legalize ourselves downstream
            "--detail_placement", "False",
            "--final_route_eval", "False",
            # CRITICAL: with LG/DP disabled the final .pl write is also skipped.
            # write_global_placement forces a .pl to be written after GP so we
            # can actually consume the result.
            "--write_global_placement", "True",
            "--write_placement", "True",
            "--inner_iter", str(XP_INNER_ITER),
            "--congest_weight", str(XP_CONGEST_WEIGHT),
            # use_route_force triggers Global Routing, which only supports
            # LEF/DEF input — bookshelf input crashes Xplace's GR module.
            # Disable; congest_weight alone uses RUDY which doesn't need GR.
            "--use_route_force", "False",
            "--route_freq", "999999",  # belt-and-suspenders: never invoke GR
            "--route_weight", "0",
            "--num_route_iter", "0",
            "--mixed_size", "True",
            "--result_dir", str(result_dir),
            "--exp_id", "xp_run",
            "--log_dir", str(work_dir),
            "--log_name", "xplace.log",
            "--num_threads", "4",
            "--deterministic", "True",
            "--seed", "0",
            "--gpu", "0",
        ]
        if XP_VERBOSE:
            print(f"[XP] running: cd {XP_HOME} && {' '.join(cmd)}")
        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd, cwd=XP_HOME,
                capture_output=True, text=True,
                timeout=max(60.0, time_budget_s + 30.0),
            )
        except subprocess.TimeoutExpired:
            print(f"[XP] subprocess timeout ({time_budget_s + 30:.0f}s)")
            return None, False
        if XP_VERBOSE:
            print(f"[XP] Xplace finished in {time.time()-t0:.0f}s, rc={proc.returncode}")
            print(f"[XP] stdout tail:\n{proc.stdout[-2000:]}")
            if proc.stderr:
                print(f"[XP] stderr tail:\n{proc.stderr[-2000:]}")
        if proc.returncode != 0:
            print(f"[XP] Xplace returned non-zero exit; treating as failure")
            return None, False

        # Find the output .pl. Xplace writes under result_dir/<exp_id>/...
        # Recursive search for any .pl that isn't our input.
        input_pl = work_dir / f"{benchmark.name}.pl"
        candidates = [p for p in result_dir.rglob("*.pl")
                       if p.resolve() != input_pl.resolve()]
        if not candidates:
            print(f"[XP] no output .pl in {result_dir}; tree:")
            for p in result_dir.rglob("*"):
                print(f"[XP]   {p.relative_to(result_dir)}")
            return None, False
        # Prefer files named <design>.gp.pl / .lg.pl / .dp.pl; else newest.
        priority = {".gp.pl": 3, ".lg.pl": 2, ".dp.pl": 1, ".pl": 0}
        def _pr(p):
            for ext, score in priority.items():
                if p.name.endswith(ext):
                    return score
            return 0
        result_pl = max(candidates, key=lambda p: (_pr(p), p.stat().st_mtime))
        if XP_VERBOSE:
            print(f"[XP] reading positions from {result_pl.relative_to(result_dir)}")

        # Parse Xplace's output .pl. Bookshelf format: lines like "o123 X Y : N"
        # where X,Y are lower-left in DBU (we wrote with × SCALE).
        out = init_positions.copy()
        with open(result_pl) as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line or not line.startswith("o"):
                    continue
                parts = line.split()
                try:
                    idx = int(parts[0][1:])
                    x_ll = float(parts[1])
                    y_ll = float(parts[2])
                except (ValueError, IndexError):
                    continue
                if idx < n_hard:
                    out[idx, 0] = x_ll / XP_SCALE + sizes[idx, 0] / 2
                    out[idx, 1] = y_ll / XP_SCALE + sizes[idx, 1] / 2

        if XP_VERBOSE:
            mov_mask = benchmark.get_movable_mask()[:n_hard].numpy()
            dmax = float(np.abs(out[mov_mask] - init_positions[mov_mask]).max()) if mov_mask.any() else 0.0
            print(f"[XP] max movable position delta (microns) = {dmax:.4f}")
        return out, True
    except Exception as e:
        print(f"[XP] unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return None, False
    finally:
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass


class XplacePlacer:
    def __init__(self, seed=42):
        self.seed = seed

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        import random
        random.seed(self.seed)
        np.random.seed(self.seed)

        n_hard = benchmark.num_hard_macros
        sizes_np = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
        cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
        hw = sizes_np[:, 0] / 2
        hh = sizes_np[:, 1] / 2
        movable = benchmark.get_movable_mask()[:n_hard].numpy()
        movable_idx = np.where(movable)[0]
        if len(movable_idx) == 0:
            return benchmark.macro_positions.clone()
        sx = (sizes_np[:, 0:1] + sizes_np[:, 0:1].T) / 2
        sy = (sizes_np[:, 1:2] + sizes_np[:, 1:2].T) / 2

        plc = _lns._load_plc(benchmark.name)
        if plc is None:
            return benchmark.macro_positions.clone()
        hard_plc_indices = plc.hard_macro_indices

        wl_data, density_data, cong_data = _lns._build_cost_data(benchmark, plc)
        neighbors = _lns._build_neighbors(wl_data, n_hard)

        init_pos = benchmark.macro_positions[:n_hard].numpy().copy().astype(np.float64)
        leg0 = _lns._minimal_fix(init_pos.copy(), movable, sizes_np, hw, hh, cw, ch, n_hard)
        true_leg0, _, _, _ = _lns._true_proxy_plc(leg0, plc, hard_plc_indices, n_hard)
        if true_leg0 is None: true_leg0 = float('inf')
        if XP_VERBOSE:
            print(f"[XP] leg0 true_proxy={true_leg0:.4f}")

        t0 = time.time()

        # ── Phase 0: Xplace ────────────────────────────────────────────────────
        grad_pos, success = run_xplace(benchmark, leg0, XP_GRAD_BUDGET)
        if success and grad_pos is not None:
            grad_legal = _lns._minimal_fix(
                grad_pos.copy(), movable, sizes_np, hw, hh, cw, ch, n_hard, gap=5e-3)
            if _lns._count_overlaps(grad_legal, n_hard, sizes_np) > 0:
                grad_legal = _lns._legalize(grad_legal, movable, sizes_np, hw, hh,
                                             cw, ch, n_hard, sx, sy, gap=5e-3)
            true_xp, _, _, _ = _lns._true_proxy_plc(grad_legal, plc, hard_plc_indices, n_hard)
            if true_xp is None: true_xp = float('inf')
            if XP_VERBOSE:
                print(f"[XP] Xplace placement: true={true_xp:.4f}  leg0={true_leg0:.4f}")
            best_pos = grad_legal
            best_true_proxy = true_xp
        else:
            if XP_VERBOSE:
                print(f"[XP] Xplace failed; falling back to leg0")
            best_pos = leg0
            best_true_proxy = true_leg0

        # ── Iterative cong-aware refinement ───────────────────────────────────
        n_mov = len(movable_idx)
        K = max(10, min(80, n_mov // 6))
        mini_iters = max(5000, min(25000, 4_000_000 // max(1, n_mov)))
        XP_MAX_REFINE = _ei("XP_MAX_REFINE", 8)

        best_known_pos = best_pos.copy()
        best_known_proxy = best_true_proxy

        for outer in range(XP_MAX_REFINE):
            time_left = XP_TOTAL_BUDGET - (time.time() - t0) - 30.0
            if time_left < 30:
                break
            improved = False
            cd_budget   = min(time_left * 0.35, 300.0)
            lns_budget  = min(time_left * 0.50, 300.0)
            swap_budget = min(time_left * 0.15, 60.0)
            if XP_VERBOSE:
                print(f"[XP-refine] outer{outer} budgets: CD={cd_budget:.0f}s "
                      f"LNS={lns_budget:.0f}s swap={swap_budget:.0f}s  best={best_true_proxy:.4f}")

            if cd_budget > 10:
                try:
                    p = _lns._cd_full_proxy(
                        best_pos, wl_data, density_data, cong_data, movable_idx,
                        n_hard, sizes_np, hw, hh, cw, ch,
                        time_budget_secs=cd_budget,
                        cong_w_calib=0.5, den_w_cd=0.5)
                    tc, _, _, _ = _lns._true_proxy_plc(p, plc, hard_plc_indices, n_hard)
                    if tc is not None and tc < best_true_proxy - 1e-6:
                        best_pos, best_true_proxy = p, tc
                        improved = True
                        if XP_VERBOSE: print(f"[XP-refine] outer{outer} CD   ✓ → {tc:.4f}")
                    elif XP_VERBOSE:
                        print(f"[XP-refine] outer{outer} CD   reject ({tc:.4f})")
                except Exception as e:
                    if XP_VERBOSE: print(f"[XP-refine] CD exception: {e}")

            if XP_TOTAL_BUDGET - (time.time() - t0) < 35: break

            if lns_budget > 20:
                try:
                    p_lns, _, _ = _lns._lns(
                        best_pos, wl_data, density_data, cong_data, movable, movable_idx,
                        sizes_np, sx, sy, hw, hh, cw, ch, n_hard, neighbors,
                        K=K, mini_iters=mini_iters,
                        time_budget_secs=lns_budget,
                        seed=self.seed + 17 + outer,
                        cong_w_calib=0.5,
                    )
                    tl, _, _, _ = _lns._true_proxy_plc(p_lns, plc, hard_plc_indices, n_hard)
                    if tl is not None and tl < best_true_proxy - 1e-6:
                        best_pos, best_true_proxy = p_lns, tl
                        improved = True
                        if XP_VERBOSE: print(f"[XP-refine] outer{outer} LNS  ✓ → {tl:.4f}")
                    elif XP_VERBOSE:
                        print(f"[XP-refine] outer{outer} LNS  reject ({tl:.4f})")
                except Exception as e:
                    if XP_VERBOSE: print(f"[XP-refine] LNS exception: {e}")

            if XP_TOTAL_BUDGET - (time.time() - t0) < 35: break

            if swap_budget > 5:
                try:
                    p = _lns._pairwise_swap_search(
                        best_pos, wl_data, density_data, cong_data, movable_idx,
                        sizes_np, sx, sy, hw, hh, cw, ch, n_hard,
                        time_budget_secs=swap_budget)
                    ts, _, _, _ = _lns._true_proxy_plc(p, plc, hard_plc_indices, n_hard)
                    if ts is not None and ts < best_true_proxy - 1e-6:
                        best_pos, best_true_proxy = p, ts
                        improved = True
                        if XP_VERBOSE: print(f"[XP-refine] outer{outer} swap ✓ → {ts:.4f}")
                    elif XP_VERBOSE:
                        print(f"[XP-refine] outer{outer} swap reject ({ts:.4f})")
                except Exception:
                    pass

            if best_true_proxy < best_known_proxy - 1e-6:
                best_known_pos = best_pos.copy()
                best_known_proxy = best_true_proxy

            if not improved:
                if XP_VERBOSE:
                    print(f"[XP-refine] outer{outer} no improvement — stopping")
                break

        best_pos = best_known_pos
        best_true_proxy = best_known_proxy

        # ── Final overlap sanity (f32 round-trip) ──────────────────────────────
        def _f32_clean(p):
            p32 = p.astype(np.float32).astype(np.float64)
            return _lns._count_overlaps(p32, n_hard, sizes_np) == 0

        best_pos_clean = best_pos
        if not _f32_clean(best_pos_clean):
            for gap in (5e-3, 1e-2, 2e-2):
                cand = _lns._minimal_fix(best_pos, movable, sizes_np,
                                          hw, hh, cw, ch, n_hard, gap=gap)
                if _f32_clean(cand):
                    best_pos_clean = cand
                    break
            else:
                for gap in (5e-3, 1e-2, 2e-2, 5e-2, 1e-1):
                    cand = _lns._legalize(best_pos, movable, sizes_np, hw, hh,
                                           cw, ch, n_hard, sx, sy, gap=gap)
                    if _f32_clean(cand):
                        best_pos_clean = cand
                        break
        best_pos = best_pos_clean

        if XP_VERBOSE:
            ov = _lns._count_overlaps(best_pos.astype(np.float32).astype(np.float64), n_hard, sizes_np)
            print(f"[XP] DONE. final_true_proxy={best_true_proxy:.4f} "
                  f"elapsed={time.time()-t0:.0f}s overlaps_f32={ov}")

        out = benchmark.macro_positions.clone()
        for i in range(n_hard):
            out[i, 0] = best_pos[i, 0]
            out[i, 1] = best_pos[i, 1]
        return out


def get_placer(seed=42):
    return XplacePlacer(seed=seed)
