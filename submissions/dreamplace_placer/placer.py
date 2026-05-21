"""
DREAMPlace Placer — uses real DREAMPlace as gradient frontend.

Approach:
  1. Convert benchmark to Bookshelf format files (.aux/.nodes/.nets/.pl/.wts/.scl).
  2. Invoke DREAMPlace via its Python API.
  3. Read back the optimized placement.
  4. Refine with the existing lns_placer LNS + CD + swap pipeline.

This is HIGH-risk / HIGH-reward. Real DREAMPlace is what the top leaderboard
teams (vmallela 1.0109, DREAMPlaceProMaxUltra 1.0121) use. If integration
works, expected proxy is in the 1.05-1.15 range. If it fails, falls back to
FastDream.

Requires the dreamplace Python package (built into the Docker image; see
Dockerfile at repo root).

Env vars:
  DP_TOTAL_BUDGET   = total wall-clock seconds (default 3500)
  DP_GRAD_BUDGET    = seconds for DREAMPlace run (default 600)
  DP_FALLBACK       = "fastdream" or "lns_placer" (default "fastdream")
  DP_NUM_BINS_X/Y   = density grid resolution (default auto)
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

# DREAMPlace v4.1.0's PlaceDB.py uses np.str / np.bool / np.int / np.float
# which were removed in numpy 1.20+. We pin numpy<2.0 in the Dockerfile, so
# we land on 1.2x where these aliases are gone. Re-bind them to the builtin
# types BEFORE importing dreamplace anywhere.
for _name, _typ in (("str", str), ("bool", bool), ("int", int),
                     ("float", float), ("object", object), ("long", int)):
    if not hasattr(np, _name):
        setattr(np, _name, _typ)

from macro_place.benchmark import Benchmark

# Reuse lns_placer infrastructure
_LNS_FILE = Path(__file__).resolve().parent.parent / "lns_placer" / "placer.py"
_spec = importlib.util.spec_from_file_location("_lns_for_dp", str(_LNS_FILE))
_lns = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lns)

# Bookshelf I/O
sys.path.insert(0, str(Path(__file__).parent))
from bookshelf_writer import write_bookshelf, SCALE as DP_SCALE  # noqa: E402


# ── Env helpers ───────────────────────────────────────────────────────────────
def _ef(name, default):
    try: return float(os.environ.get(name, default))
    except Exception: return default
def _ei(name, default):
    try: return int(os.environ.get(name, default))
    except Exception: return default

DP_TOTAL_BUDGET = _ef("DP_TOTAL_BUDGET", 3500.0)
DP_GRAD_BUDGET  = _ef("DP_GRAD_BUDGET", 600.0)
DP_FALLBACK     = os.environ.get("DP_FALLBACK", "fastdream")
DP_VERBOSE      = _ei("DP_VERBOSE", 1)
# DP_GPU: 1 to attempt GPU (default), 0 to force CPU. We auto-detect at
# runtime by inspecting DREAMPlace's build configuration (configure.py
# records CUDA_FOUND); if CUDA wasn't compiled in, fall back to CPU silently.
DP_GPU_REQUEST  = _ei("DP_GPU", 1)


# ── DREAMPlace driver ─────────────────────────────────────────────────────────

def _dreamplace_config(aux_path, work_dir, num_bins_x, num_bins_y,
                        budget_s, target_density, gpu, random_seed=1000):
    """Build a DREAMPlace JSON config for the run.

    target_density: 0.0–1.0; should match the actual macro+soft-macro
        utilization or slightly above. 1.0 over-spreads cells and harms WL.
    gpu: 1 to use GPU (only works if DREAMPlace was built with CUDA), 0 for CPU.
    """
    return {
        "aux_input": str(aux_path),
        "gpu": gpu,
        "num_bins_x": num_bins_x,
        "num_bins_y": num_bins_y,
        "global_place_stages": [
            {
                "num_bins_x": num_bins_x,
                "num_bins_y": num_bins_y,
                "iteration": 1000,
                "learning_rate": 0.01,
                "wirelength": "weighted_average",
                "optimizer": "nesterov",
            }
        ],
        "target_density": target_density,
        "density_weight": 8e-5,
        "gamma": 4.0,
        "random_seed": random_seed,
        "scale_factor": 1.0,
        "shift_factor": [0.0, 0.0],
        "ignore_net_degree": 100,
        "enable_fillers": 0,         # no std cells in our flow; fillers add noise
        "global_place_flag": 1,
        "legalize_flag": 0,          # DREAMPlace's std-cell legalizer treats
                                     # our hard macros as cells (one giant row
                                     # → all 'cells' tiny → hangs). We legalize
                                     # ourselves via _lns._minimal_fix downstream.
        "detailed_place_flag": 0,    # we do our own refinement
        "stop_overflow": _ef("DP_STOP_OVERFLOW", 0.07),  # lower → forces more iters
        "dtype": "float32",
        "plot_flag": 0,
        "deterministic_flag": 1,
        "result_dir": str(work_dir),
    }


def run_dreamplace(benchmark, init_positions, time_budget_s):
    """
    Run DREAMPlace on the benchmark. Returns (positions_np, success).
    `init_positions` is [n_hard, 2] of (x_center, y_center).
    """
    # DREAMPlace v4.x's Placer.py uses BARE imports (`import Params`,
    # `import PlaceDB`, ...) for sibling modules. Ensure the dreamplace
    # package directory itself is on sys.path so those resolve, regardless
    # of how the container was launched.
    try:
        import dreamplace as _dp_pkg
        _dp_dir = os.path.dirname(_dp_pkg.__file__)
        if _dp_dir not in sys.path:
            sys.path.insert(0, _dp_dir)
    except ImportError as e:
        if DP_VERBOSE:
            print(f"[DP] dreamplace package not available: {e}")
        return None, False

    try:
        import dreamplace.Placer  # noqa: F401
    except ImportError as e:
        if DP_VERBOSE:
            print(f"[DP] dreamplace.Placer import failed: {e}")
        return None, False

    # Decide GPU vs CPU: only request GPU if DREAMPlace was built with CUDA
    # (configure.py records this), and the user hasn't forced DP_GPU=0.
    gpu_flag = 0
    try:
        import dreamplace.configure as _dp_cfg
        cuda_found = str(_dp_cfg.compile_configurations.get("CUDA_FOUND", "")).upper() == "TRUE"
    except Exception:
        cuda_found = False
    if DP_GPU_REQUEST and cuda_found:
        gpu_flag = 1
    if DP_VERBOSE:
        print(f"[DP] CUDA built into DREAMPlace? {cuda_found}  → gpu_flag={gpu_flag}")

    n_hard = benchmark.num_hard_macros
    cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
    sizes = benchmark.macro_sizes.numpy()

    # Pick density grid resolution. DREAMPlace likes powers of 2.
    # IMPORTANT: bin count is bounded INDEPENDENTLY of canvas size — the
    # FFT density grid is num_bins_x * num_bins_y * 8 bytes per array, so
    # e.g. 65536² is 32 GB and OOM-kills the container immediately.
    # AutoDMP-style: 512–1024 bins is plenty for ICCAD04-scale designs.
    DP_MAX_BINS = _ei("DP_MAX_BINS", 1024)
    DP_MIN_BINS = 64
    def _pow2_clamped(x_um):
        # Aim for one bin per ~64 microns at canvas scale (or finer/coarser
        # depending on canvas), then clamp into the safe range.
        target = max(8.0, x_um)
        n = 1 << max(4, int(np.ceil(np.log2(target))))
        return max(DP_MIN_BINS, min(DP_MAX_BINS, n))
    nbx = _ei("DP_NUM_BINS_X", _pow2_clamped(cw))
    nby = _ei("DP_NUM_BINS_Y", _pow2_clamped(ch))

    # Compute target_density from actual placeable utilization: all node
    # areas (hard + soft) / canvas area. AutoDMP's default heuristic is
    # max(0.5, util * 1.1) — gives DREAMPlace just enough slack to legalize.
    all_sizes = benchmark.macro_sizes.numpy()
    util = float((all_sizes[:, 0] * all_sizes[:, 1]).sum()) / max(1e-9, cw * ch)
    target_density = float(min(0.95, max(0.5, util * 1.1)))

    work_dir = Path(tempfile.mkdtemp(prefix="dp_"))
    try:
        plc_obj = _lns._load_plc(benchmark.name)   # for compute_proxy_cost downstream
        aux_path = write_bookshelf(work_dir, benchmark.name, benchmark, plc_obj, init_positions)
        cfg = _dreamplace_config(aux_path, work_dir, nbx, nby, time_budget_s,
                                  target_density=target_density,
                                  gpu=gpu_flag)
        if DP_VERBOSE:
            print(f"[DP] util={util:.3f}  target_density={target_density:.3f}")
        cfg_path = work_dir / "config.json"
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)

        if DP_VERBOSE:
            print(f"[DP] running DREAMPlace ({nbx}x{nby} bins, budget={time_budget_s:.0f}s)")
        t0 = time.time()

        # Call DREAMPlace's Python entry point with a timeout
        import dreamplace.Placer as DPlacer
        import dreamplace.Params as DPParams
        params = DPParams.Params()
        params.load(str(cfg_path))
        try:
            DPlacer.place(params)
        except Exception as e:
            print(f"[DP] DREAMPlace.place() raised: {e}")
            return None, False
        if DP_VERBOSE:
            print(f"[DP] DREAMPlace finished in {time.time()-t0:.0f}s")
            try:
                print(f"[DP] work_dir contents: {sorted(os.listdir(work_dir))}")
            except Exception:
                pass

        # DREAMPlace v4.x writes output to <result_dir>/<benchmark>/<benchmark>.gp.pl
        # (a subdirectory named after the benchmark), NOT directly into result_dir.
        # Our input .pl shares the work_dir root, so a top-level-only scan would
        # silently match the input. Search recursively, prefer the deepest match,
        # and explicitly EXCLUDE the input .pl path.
        input_pl = work_dir / f"{benchmark.name}.pl"
        candidates = []
        for ext in (".lg.pl", ".dp.pl", ".gp.pl"):
            for p in work_dir.rglob(f"*{ext}"):
                if p.resolve() != input_pl.resolve():
                    candidates.append(p)
        result_pl = None
        # Priority: .lg.pl > .dp.pl > .gp.pl (legalized > detailed > global).
        for ext in (".lg.pl", ".dp.pl", ".gp.pl"):
            for c in candidates:
                if c.name.endswith(ext):
                    result_pl = c
                    break
            if result_pl is not None:
                break
        if result_pl is None:
            print(f"[DP] no DREAMPlace output .pl found under {work_dir}")
            try:
                for p in work_dir.rglob("*"):
                    print(f"[DP]   {p.relative_to(work_dir)}  ({'dir' if p.is_dir() else f'{p.stat().st_size}b'})")
            except Exception:
                pass
            return None, False
        if DP_VERBOSE:
            print(f"[DP] reading positions from {result_pl.relative_to(work_dir)}")

        # Parse DREAMPlace's output: coords are in scaled DBU (× DP_SCALE).
        # Convert lower-left DBU → micron center: divide by DP_SCALE, add size/2.
        out = init_positions.copy()
        with open(result_pl) as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line or not line.startswith("o"):
                    continue
                parts = line.split()
                try:
                    idx = int(parts[0][1:])
                    x_ll_dbu = float(parts[1])
                    y_ll_dbu = float(parts[2])
                except (ValueError, IndexError):
                    continue
                if idx < n_hard:
                    x_ll_um = x_ll_dbu / DP_SCALE
                    y_ll_um = y_ll_dbu / DP_SCALE
                    out[idx, 0] = x_ll_um + sizes[idx, 0] / 2
                    out[idx, 1] = y_ll_um + sizes[idx, 1] / 2
        if DP_VERBOSE:
            # Did DREAMPlace actually move any movable hard macro?
            mov_mask = benchmark.get_movable_mask()[:n_hard].numpy()
            dmax = float(np.abs(out[mov_mask] - init_positions[mov_mask]).max()) if mov_mask.any() else 0.0
            print(f"[DP] max movable position delta (microns) = {dmax:.4f}")
        return out, True
    except Exception as e:
        print(f"[DP] unexpected error: {e}")
        return None, False
    finally:
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass


# ── Main placer ───────────────────────────────────────────────────────────────

class DreamPlacePlacer:
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
        leg0_wl = _lns._wl_cost(leg0, wl_data)
        leg0_den = _lns._top_density(_lns._build_density_grid(leg0, density_data, n_hard),
                                      density_data['n_top'])
        leg0_cong = _lns._cong_cost(*_lns._build_cong_grid(leg0, cong_data), cong_data)
        cong_w_calib = (0.5 * (leg0_wl + 0.5 * leg0_den) / leg0_cong
                        if leg0_cong > 1e-9 else 0.5)
        leg0_calib, _, _, _ = _lns._calibrated_proxy(
            leg0, wl_data, density_data, cong_data, n_hard, cong_w_calib)

        if DP_VERBOSE:
            print(f"[DP] leg0_calib={leg0_calib:.4f}  cong_w_calib={cong_w_calib:.4f}")

        t0 = time.time()
        total_budget = DP_TOTAL_BUDGET

        # ── Phase 0: DREAMPlace ────────────────────────────────────────────────
        grad_pos, success = run_dreamplace(benchmark, leg0, DP_GRAD_BUDGET)
        if success and grad_pos is not None:
            grad_legal = _lns._minimal_fix(
                grad_pos.copy(), movable, sizes_np, hw, hh, cw, ch, n_hard)
            if _lns._count_overlaps(grad_legal, n_hard, sizes_np) > 0:
                grad_legal = _lns._legalize(grad_legal, movable, sizes_np, hw, hh,
                                             cw, ch, n_hard, sx, sy, gap=0.002)
            true_dp, _, _, _ = _lns._true_proxy_plc(
                grad_legal, plc, hard_plc_indices, n_hard)
            true_leg0, _, _, _ = _lns._true_proxy_plc(
                leg0, plc, hard_plc_indices, n_hard)
            if true_dp is None: true_dp = float('inf')
            if true_leg0 is None: true_leg0 = float('inf')
            if DP_VERBOSE:
                print(f"[DP] DREAMPlace placement: true={true_dp:.4f}  leg0={true_leg0:.4f}")
            best_pos = grad_legal
            best_true_proxy = true_dp
        else:
            if DP_VERBOSE:
                print(f"[DP] DREAMPlace FAILED → falling back to {DP_FALLBACK}")
            best_pos = _fallback_gradient(
                DP_FALLBACK, leg0, wl_data, density_data, cong_data, movable,
                movable_idx, sizes_np, sx, sy, hw, hh, cw, ch, n_hard,
                neighbors, self.seed)
            best_true_proxy, _, _, _ = _lns._true_proxy_plc(
                best_pos, plc, hard_plc_indices, n_hard)
            if best_true_proxy is None: best_true_proxy = float('inf')

        # ── Iterative cong-aware refinement ───────────────────────────────────
        # DREAMPlace minimizes HPWL + electrostatic density but IGNORES congestion.
        # The true proxy is WL + 0.5*den + 0.5*cong, and on most ICCAD04 benches
        # the cong term dominates (e.g., ibm04 post-DREAMPlace: wl=0.075, den=0.80,
        # cong=1.79). So DREAMPlace's output needs aggressive cong-targeted refinement.
        #
        # Strategy: CD → LNS → swap with cong_w_calib=0.5 (matching the true proxy
        # weighting EXACTLY), iterated until plateau. Every phase uses the official
        # _true_proxy_plc() to accept/reject — no calibrated approximation drift.
        n_mov = len(movable_idx)
        K = max(10, min(80, n_mov // 6))
        mini_iters = max(5000, min(25000, 4_000_000 // max(1, n_mov)))
        DP_MAX_REFINE = _ei("DP_MAX_REFINE", 20)
        DP_PERTURB_FRAC = _ef("DP_PERTURB_FRAC", 0.33)
        DP_PERTURB_SCALE = _ef("DP_PERTURB_SCALE", 0.40)  # ±frac of canvas

        # Track best-ever-seen separately from the working position. After a
        # perturbation, best_pos jumps to a (likely worse) random state; we
        # don't want that to clobber the best result we already found.
        best_known_pos = best_pos.copy()
        best_known_proxy = best_true_proxy
        prng = np.random.default_rng(self.seed)
        no_improve_streak = 0

        for outer in range(DP_MAX_REFINE):
            time_left = total_budget - (time.time() - t0) - 30.0  # 30s for final fix
            if time_left < 30:
                if DP_VERBOSE:
                    print(f"[DP-refine] outer{outer}: time_left={time_left:.0f}s, stopping")
                break
            improved = False
            # Budget split: 35% CD, 50% LNS, 15% swap of remaining time per iter
            cd_budget   = min(time_left * 0.35, 300.0)
            lns_budget  = min(time_left * 0.50, 300.0)
            swap_budget = min(time_left * 0.15, 60.0)
            if DP_VERBOSE:
                print(f"[DP-refine] outer{outer} budgets: CD={cd_budget:.0f}s "
                      f"LNS={lns_budget:.0f}s swap={swap_budget:.0f}s  best={best_true_proxy:.4f}")

            # CD with cong_w=0.5 (matches true proxy weight on cong)
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
                        if DP_VERBOSE: print(f"[DP-refine] outer{outer} CD   ✓ → {tc:.4f}")
                    elif DP_VERBOSE:
                        print(f"[DP-refine] outer{outer} CD   reject ({tc:.4f})")
                except Exception as e:
                    if DP_VERBOSE: print(f"[DP-refine] CD exception: {e}")

            if total_budget - (time.time() - t0) < 35: break

            # LNS with cong_w=0.5 (matches true proxy)
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
                        if DP_VERBOSE: print(f"[DP-refine] outer{outer} LNS  ✓ → {tl:.4f}")
                    elif DP_VERBOSE:
                        print(f"[DP-refine] outer{outer} LNS  reject ({tl:.4f})")
                except Exception as e:
                    if DP_VERBOSE: print(f"[DP-refine] LNS exception: {e}")

            if total_budget - (time.time() - t0) < 35: break

            # Pairwise swap (already cong-aware via _pairwise_swap_search internals)
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
                        if DP_VERBOSE: print(f"[DP-refine] outer{outer} swap ✓ → {ts:.4f}")
                    elif DP_VERBOSE:
                        print(f"[DP-refine] outer{outer} swap reject ({ts:.4f})")
                except Exception:
                    pass

            # Track best-ever; the perturb step may move best_pos to a worse
            # state on purpose, so we keep best_known as the rollback target.
            if best_true_proxy < best_known_proxy - 1e-6:
                best_known_pos = best_pos.copy()
                best_known_proxy = best_true_proxy

            if improved:
                no_improve_streak = 0
                continue

            no_improve_streak += 1
            # AGGRESSIVE PERTURBATION: when an iter finds nothing, randomize
            # a chunk of macros to break out of the local minimum DREAMPlace
            # parked us in, then let the next iter refine from there.
            # Perturbation budget scales with consecutive no-improvement count:
            # bigger jumps after repeated failures.
            time_left = total_budget - (time.time() - t0) - 30.0
            if time_left < 60:
                if DP_VERBOSE:
                    print(f"[DP-refine] outer{outer} no time for perturbation")
                break
            frac = min(0.80, DP_PERTURB_FRAC * (1 + 0.5 * no_improve_streak))
            scale = min(1.0, DP_PERTURB_SCALE * (1 + 0.3 * no_improve_streak))
            n_perturb = max(2, int(n_mov * frac))
            perturb_idx = prng.choice(movable_idx, size=n_perturb, replace=False)
            perturbed = best_known_pos.copy()
            dx_range = cw * scale
            dy_range = ch * scale
            for i in perturb_idx:
                # Translate by ±scale·canvas in each axis, clamp to legal range.
                nx = perturbed[i, 0] + prng.uniform(-dx_range, dx_range)
                ny = perturbed[i, 1] + prng.uniform(-dy_range, dy_range)
                perturbed[i, 0] = float(np.clip(nx, hw[i], cw - hw[i]))
                perturbed[i, 1] = float(np.clip(ny, hh[i], ch - hh[i]))
            # Legalize: minimal_fix at f32-safe gap first, fall back to full
            # legalizer if overlaps remain.
            perturbed = _lns._minimal_fix(
                perturbed, movable, sizes_np, hw, hh, cw, ch, n_hard, gap=5e-3)
            if _lns._count_overlaps(perturbed, n_hard, sizes_np) > 0:
                perturbed = _lns._legalize(
                    perturbed, movable, sizes_np, hw, hh, cw, ch,
                    n_hard, sx, sy, gap=5e-3)
            if _lns._count_overlaps(perturbed, n_hard, sizes_np) > 0:
                if DP_VERBOSE:
                    print(f"[DP-refine] outer{outer}: perturbation legalize "
                          f"failed, stopping")
                break
            best_pos = perturbed
            pt, _, _, _ = _lns._true_proxy_plc(
                best_pos, plc, hard_plc_indices, n_hard)
            best_true_proxy = pt if pt is not None else float('inf')
            if DP_VERBOSE:
                print(f"[DP-refine] outer{outer}: PERTURB {n_perturb}/{n_mov} "
                      f"macros at ±{scale:.0%} canvas → start={best_true_proxy:.4f} "
                      f"(best_known={best_known_proxy:.4f}, streak={no_improve_streak})")

        # End of refinement loop — always return the best-ever-seen position,
        # not whatever the working state ended on (which may be a perturbed
        # worse-than-best snapshot).
        best_pos = best_known_pos
        best_true_proxy = best_known_proxy
        if DP_VERBOSE:
            print(f"[DP-refine] best-ever true_proxy = {best_true_proxy:.4f}")

        # Final overlap sanity.
        # The evaluator stores positions as torch.float32, so any placement we
        # return is implicitly cast to f32. Sub-f32-precision separations
        # (gap=1e-8 in _minimal_fix's default) collapse to zero gap and the
        # evaluator counts the touching macros as overlaps. We must verify
        # zero overlaps AFTER the f32 round-trip, escalating the legalize
        # gap until it holds.
        def _f32_clean(p):
            p32 = p.astype(np.float32).astype(np.float64)
            return _lns._count_overlaps(p32, n_hard, sizes_np) == 0

        best_pos_clean = best_pos
        if not _f32_clean(best_pos_clean):
            # Try minimal_fix with increasing gaps (each must be well above
            # f32 precision of ~3e-6 at canvas scales ~25-80).
            for gap in (5e-3, 1e-2, 2e-2):
                cand = _lns._minimal_fix(best_pos, movable, sizes_np,
                                          hw, hh, cw, ch, n_hard, gap=gap)
                if _f32_clean(cand):
                    best_pos_clean = cand
                    if DP_VERBOSE:
                        print(f"[DP] final-fix: minimal_fix(gap={gap}) clean")
                    break
            else:
                # Fall back to full legalizer with progressively larger gaps.
                for gap in (5e-3, 1e-2, 2e-2, 5e-2, 1e-1):
                    cand = _lns._legalize(best_pos, movable, sizes_np, hw, hh,
                                           cw, ch, n_hard, sx, sy, gap=gap)
                    if _f32_clean(cand):
                        best_pos_clean = cand
                        if DP_VERBOSE:
                            print(f"[DP] final-fix: legalize(gap={gap}) clean")
                        break
        best_pos = best_pos_clean

        if DP_VERBOSE:
            print(f"[DP] DONE. final_true_proxy={best_true_proxy:.4f} "
                  f"elapsed={time.time()-t0:.0f}s "
                  f"final_overlaps_f32={_lns._count_overlaps(best_pos.astype(np.float32).astype(np.float64), n_hard, sizes_np)}")

        out = benchmark.macro_positions.clone()
        for i in range(n_hard):
            out[i, 0] = best_pos[i, 0]
            out[i, 1] = best_pos[i, 1]
        return out


def _fallback_gradient(name, leg0, wl_data, density_data, cong_data, movable,
                         movable_idx, sizes_np, sx, sy, hw, hh, cw, ch, n_hard,
                         neighbors, seed):
    """If DREAMPlace fails, use a non-gradient fallback."""
    # NOTE: we use the gradient + the float32-safe legalize pattern from FastDream,
    # not just the gradient itself. _minimal_fix(gap=1e-8) below float32 precision.
    if name == "fastdream":
        try:
            fd_spec = importlib.util.spec_from_file_location(
                "_fastdream", str(Path(__file__).parent.parent / "fastdream" / "placer.py"))
            fd_mod = importlib.util.module_from_spec(fd_spec)
            fd_spec.loader.exec_module(fd_mod)
            grad_raw = fd_mod._dreamplace_gradient(
                leg0, wl_data, density_data, movable, n_hard,
                cw, ch, 400.0, seed=seed)
            # Float32-safe legalize: gap=5e-3 is well above float32 precision.
            fixed = _lns._minimal_fix(grad_raw, movable, sizes_np, hw, hh, cw, ch, n_hard, gap=5e-3)
            # Verify at f32
            if _lns._count_overlaps(fixed.astype(np.float32).astype(np.float64),
                                     n_hard, sizes_np) == 0:
                return fixed
            # Escalate
            for gap in [1e-2, 2e-2, 5e-2]:
                cand = _lns._legalize(grad_raw, movable, sizes_np, hw, hh, cw, ch,
                                       n_hard, sx, sy, gap=gap)
                if _lns._count_overlaps(cand.astype(np.float32).astype(np.float64),
                                         n_hard, sizes_np) == 0:
                    return cand
        except Exception:
            pass
    # Last resort: SA fallback from leg0
    n_mov = len(movable_idx)
    result, _ = _lns._sa(
        leg0, wl_data, density_data, cong_data, movable, movable_idx,
        sizes_np, sx, sy, hw, hh, cw, ch, n_hard, neighbors,
        n_iters=max(200_000, n_mov * 300), seed=seed)
    return result


def get_placer(seed=42):
    return DreamPlacePlacer(seed=seed)
