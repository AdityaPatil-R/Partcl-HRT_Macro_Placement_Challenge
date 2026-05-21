"""
FastDream Placer — DREAMPlace-faithful gradient + heavy LNS refinement.

Designed to close the gap to the top 2 leaderboard entries (vmallela 1.0109,
DREAMPlaceProMaxUltra 1.0121). Both use DREAMPlace + LNS. The gap is bridged
by getting the DREAMPlace details right:

  1. Annealed γ (HPWL smoothing): start high (γ_lo, smooth), end low (γ_hi, sharp)
  2. Annealed density grid: start coarse (global pull), end fine (local repulsion)
  3. Nesterov SGD with momentum (DREAMPlace's actual optimizer)
  4. Dynamic λ from gradient norm ratio (||grad_WL|| / ||grad_density||)
  5. Diverse multi-start initializations (random + spiral + leg0)
  6. Best-checkpoint tracking by smooth proxy across the run
  7. Multi-strategy legalization (pick best by post-legalize proxy)
  8. Heavy LNS refinement built on top of the existing lns_placer LNS

Reuses lns_placer's cost infrastructure (HPWL/density/cong/proxy/legalize/LNS/CD).
"""

import importlib.util
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from macro_place.benchmark import Benchmark


# ── Reuse the lns_placer infrastructure ───────────────────────────────────────
_LNS_FILE = Path(__file__).resolve().parent.parent / "lns_placer" / "placer.py"
_spec = importlib.util.spec_from_file_location("_lns_for_fastdream", str(_LNS_FILE))
_lns = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lns)

# Bind the names we need
_build_cost_data       = _lns._build_cost_data
_wl_cost               = _lns._wl_cost
_top_density           = _lns._top_density
_build_density_grid    = _lns._build_density_grid
_build_cong_grid       = _lns._build_cong_grid
_cong_cost             = _lns._cong_cost
_proxy                 = _lns._proxy
_calibrated_proxy      = _lns._calibrated_proxy
_count_overlaps        = _lns._count_overlaps
_minimal_fix           = _lns._minimal_fix
_legalize              = _lns._legalize
_load_plc              = _lns._load_plc
_build_neighbors       = _lns._build_neighbors
_true_proxy_plc        = _lns._true_proxy_plc
_sa                    = _lns._sa
_lns_loop              = _lns._lns
_pairwise_swap_search  = _lns._pairwise_swap_search
_cd_full_proxy         = _lns._cd_full_proxy
_coordinate_descent    = _lns._coordinate_descent
_build_gpu_tensors     = _lns._build_gpu_tensors
_wl_loss_gpu           = _lns._wl_loss_gpu
_density_loss_gpu      = _lns._density_loss_gpu
_electrostatic_density_loss_gpu = _lns._electrostatic_density_loss_gpu
_gpu_spread_legalize   = _lns._gpu_spread_legalize
_get_device            = _lns._get_device


# ── Env vars ──────────────────────────────────────────────────────────────────
def _ei(n, d):
    try: return int(os.environ.get(n, d))
    except: return d
def _ef(n, d):
    try: return float(os.environ.get(n, d))
    except: return d

FD_TOTAL_BUDGET   = _ef("FD_TOTAL_BUDGET", 3500.0)
FD_GRAD_BUDGET    = _ef("FD_GRAD_BUDGET", 600.0)
FD_LNS_FRAC       = _ef("FD_LNS_FRAC", 0.65)       # fraction of remaining budget for LNS
FD_N_STARTS       = _ei("FD_N_STARTS", 3)          # multi-start gradient
FD_GAMMA_LO       = _ef("FD_GAMMA_LO", 0.5)        # final γ (sharp HPWL)
FD_GAMMA_HI       = _ef("FD_GAMMA_HI", 8.0)        # initial γ (smooth)
FD_LAMBDA_MIN     = _ef("FD_LAMBDA_MIN", 1e-4)
FD_LAMBDA_MAX     = _ef("FD_LAMBDA_MAX", 50.0)
FD_LR             = _ef("FD_LR", -1.0)             # -1 = auto-scale to canvas
FD_USE_NESTEROV   = _ei("FD_USE_NESTEROV", 1)      # 1=Nesterov SGD, 0=Adam
FD_GRID_MULT_LO   = _ef("FD_GRID_MULT_LO", 0.5)    # coarse grid factor at start
FD_GRID_MULT_HI   = _ef("FD_GRID_MULT_HI", 1.0)    # fine grid factor at end
FD_FORCE_ACCEPT   = _ei("FD_FORCE_ACCEPT", 0)      # always use gradient result (don't fall back)
FD_VERBOSE        = _ei("FD_VERBOSE", 1)
FD_SKIP_GRAD      = _ei("FD_SKIP_GRAD", 0)         # skip bad gradient basin search; spend time on polish
FD_POLISH_ONLY    = _ei("FD_POLISH_ONLY", 0)       # alias for skip-grad + heavier LNS/CD polish

# ── New tuning knobs (added for v2) ───────────────────────────────────────────
# Auto-calibration of cong_w_calib collapsed to ~0.016 on ibm01, which made
# the multi-start scorer ignore congestion. True proxy weighs cong at 0.5.
FD_CONG_W_CALIB   = _ef("FD_CONG_W_CALIB", 0.5)    # -1 = use auto-calibration
# Sweep over multiple cong_w values per gradient init, take best by true proxy.
# Comma-separated list, e.g. "0.3,0.5,0.7". Empty = single value from above.
FD_CONG_W_SWEEP   = os.environ.get("FD_CONG_W_SWEEP", "").strip()
# Saddle escape: between gradient rounds, perturb previous-best by N*canvas.
# 0 disables. Typical: 0.02–0.05.
FD_SADDLE_NOISE   = _ef("FD_SADDLE_NOISE", 0.0)


# ── Initialization schemes ────────────────────────────────────────────────────

def _spiral_init(sizes_np, hw, hh, cw, ch, n_hard, movable, leg0, seed=0):
    """Counter-clockwise spiral from boundary."""
    rng = random.Random(seed)
    pos = leg0.copy()
    mov_idx = np.where(movable)[0].tolist()
    rng.shuffle(mov_idx)
    if not mov_idx:
        return pos

    ring = 0
    placed = 0
    n = len(mov_idx)
    while placed < n:
        margin = ring * max(1.0, (cw + ch) / 100)
        x_lo = margin + hw.max()
        x_hi = cw - margin - hw.max()
        y_lo = margin + hh.max()
        y_hi = ch - margin - hh.max()
        if x_lo >= x_hi or y_lo >= y_hi:
            cx_c, cy_c = cw / 2, ch / 2
            for bidx in mov_idx[placed:]:
                pos[bidx, 0] = float(np.clip(cx_c + rng.uniform(-1, 1), hw[bidx], cw - hw[bidx]))
                pos[bidx, 1] = float(np.clip(cy_c + rng.uniform(-1, 1), hh[bidx], ch - hh[bidx]))
                placed += 1
            break
        perim = 2 * (x_hi - x_lo) + 2 * (y_hi - y_lo)
        if perim <= 0: break
        step = perim / max(1, n - placed)
        t = 0.0
        edges = [(x_hi - x_lo, 0), (y_hi - y_lo, 1), (x_hi - x_lo, 2), (y_hi - y_lo, 3)]
        cum_edge = [0]
        for ln, _ in edges:
            cum_edge.append(cum_edge[-1] + ln)
        while placed < n and t < perim:
            for ei in range(4):
                if cum_edge[ei] <= t < cum_edge[ei+1]:
                    local = t - cum_edge[ei]
                    if ei == 0:   x, y = x_lo + local, y_lo
                    elif ei == 1: x, y = x_hi, y_lo + local
                    elif ei == 2: x, y = x_hi - local, y_hi
                    else:         x, y = x_lo, y_hi - local
                    break
            bidx = mov_idx[placed]
            pos[bidx, 0] = float(np.clip(x, hw[bidx], cw - hw[bidx]))
            pos[bidx, 1] = float(np.clip(y, hh[bidx], ch - hh[bidx]))
            placed += 1
            t += step
        ring += 1
    return pos


def _random_init(sizes_np, hw, hh, cw, ch, n_hard, movable, leg0, seed=0):
    rng = np.random.RandomState(seed)
    pos = leg0.copy()
    for bidx in np.where(movable)[0]:
        pos[bidx, 0] = rng.uniform(hw[bidx], cw - hw[bidx])
        pos[bidx, 1] = rng.uniform(hh[bidx], ch - hh[bidx])
    return pos


def _build_initial_positions(n_starts, sizes_np, hw, hh, cw, ch, n_hard,
                               movable, leg0):
    """Mix of init strategies. First is always leg0 (the warm start)."""
    out = [leg0.copy()]
    seeds = [11, 22, 33, 44, 55, 66, 77, 88]
    while len(out) < n_starts:
        si = seeds[len(out) - 1] if (len(out) - 1) < len(seeds) else random.randint(1, 1_000_000)
        if len(out) % 2 == 1:
            out.append(_spiral_init(sizes_np, hw, hh, cw, ch, n_hard, movable, leg0, seed=si))
        else:
            out.append(_random_init(sizes_np, hw, hh, cw, ch, n_hard, movable, leg0, seed=si))
    return out[:n_starts]


# ── DREAMPlace-faithful gradient ──────────────────────────────────────────────

def _dreamplace_gradient(pos_init_np, wl_data, density_data, movable_np,
                          n_hard, cw, ch, time_budget_secs, seed=42):
    """
    Single-start gradient with annealed γ, annealed density grid, Nesterov,
    dynamic λ from gradient norm ratio. Returns best position by smooth proxy.
    """
    device = _get_device()
    if device.type in ('cpu', 'mps') and not _ei("FD_FORCE_CPU", 0):
        raise RuntimeError(f"{device.type.upper()} gradient not cost-effective")
    torch.manual_seed(seed)

    gd_full = _build_gpu_tensors(wl_data, density_data, n_hard, device)
    sizes_t = gd_full['hard_sizes']
    hard_util = float((sizes_t[:, 0] * sizes_t[:, 1]).sum().item()) / (cw * ch)

    mov_t = torch.tensor(movable_np, dtype=torch.bool, device=device)
    hw_t = sizes_t[:, 0] / 2
    hh_t = sizes_t[:, 1] / 2
    lo = torch.stack([hw_t, hh_t], dim=1)
    hi = torch.stack([cw - hw_t, ch - hh_t], dim=1)
    pos0 = torch.tensor(pos_init_np, dtype=torch.float32, device=device)
    pos = pos0.clone().detach().requires_grad_(True)

    default_lr = max(0.01, min(0.05, (cw + ch) * 6e-4))
    lr = FD_LR if FD_LR > 0 else default_lr

    if FD_USE_NESTEROV:
        opt = torch.optim.SGD([pos], lr=lr, momentum=0.9, nesterov=True)
    else:
        opt = torch.optim.Adam([pos], lr=lr, betas=(0.9, 0.999))

    t0 = time.time()
    best_smooth = float('inf')
    best_np = pos_init_np.copy()

    # Estimate throughput from a 1-second probe
    warm_end = t0 + min(2.0, time_budget_secs * 0.02)
    probe_iters = 0
    while time.time() < warm_end:
        opt.zero_grad()
        wl = _wl_loss_gpu(pos, gd_full, 2.0)
        wl.backward()
        opt.step()
        with torch.no_grad():
            pos.data.clamp_(lo, hi)
            pos.data[~mov_t] = pos0[~mov_t]
        probe_iters += 1

    sps = probe_iters / max(0.1, time.time() - t0)
    main_budget = time_budget_secs - (time.time() - t0)
    total_steps = max(100, int(sps * main_budget))

    if FD_VERBOSE:
        print(f"  [FD-grad] sps≈{sps:.0f}, total_steps={total_steps}")

    # Build coarse-grid gd (for early annealed grid). Use lower res by downsampling fixed_density.
    def _make_gd(grid_mult):
        """Return a gd dict with rescaled grid."""
        if abs(grid_mult - 1.0) < 1e-6:
            return gd_full
        gd2 = dict(gd_full)
        gc_new = max(8, int(gd_full['gc'] * grid_mult))
        gr_new = max(8, int(gd_full['gr'] * grid_mult))
        # Resize fixed_density via interpolation
        fd = gd_full['fixed_density'].unsqueeze(0).unsqueeze(0)
        fd_new = torch.nn.functional.interpolate(fd, size=(gr_new, gc_new), mode='bilinear', align_corners=False)
        gd2['fixed_density'] = fd_new.squeeze(0).squeeze(0)
        gd2['gc'] = gc_new
        gd2['gr'] = gr_new
        gd2['gw'] = cw / gc_new
        gd2['gh'] = ch / gr_new
        return gd2

    # Annealing schedules
    def gamma_at(t):
        # log-anneal from γ_hi to γ_lo
        frac = t / max(1, total_steps - 1)
        return FD_GAMMA_HI * (FD_GAMMA_LO / FD_GAMMA_HI) ** frac

    def grid_mult_at(t):
        frac = t / max(1, total_steps - 1)
        return FD_GRID_MULT_LO + (FD_GRID_MULT_HI - FD_GRID_MULT_LO) * frac

    last_grid_mult = None
    gd_cur = gd_full

    for step in range(total_steps):
        if time.time() - t0 > time_budget_secs:
            break

        gamma = gamma_at(step)
        gm = grid_mult_at(step)
        # Rebuild gd_cur if grid resolution changed appreciably
        if last_grid_mult is None or abs(gm - last_grid_mult) > 0.05:
            gd_cur = _make_gd(gm)
            last_grid_mult = gm

        opt.zero_grad()
        wl = _wl_loss_gpu(pos, gd_cur, gamma)
        den = _electrostatic_density_loss_gpu(pos, gd_cur, hard_util)

        # Dynamic λ from gradient norms (DREAMPlace-style).
        # retain_graph=True on the first call so the second autograd.grad
        # can still walk shared upstream nodes; the second call frees
        # everything because we're not doing another backward.
        wl_grad = torch.autograd.grad(wl, pos, retain_graph=True)[0]
        den_grad = torch.autograd.grad(den, pos)[0]
        wl_norm = wl_grad.norm().item() + 1e-12
        den_norm = den_grad.norm().item() + 1e-12
        lam = max(FD_LAMBDA_MIN, min(FD_LAMBDA_MAX, wl_norm / den_norm))

        # Set pos.grad directly instead of calling loss.backward() again —
        # both wl's and den's graphs have already been consumed by the
        # autograd.grad calls above, so a second backward would crash with
        # "Trying to backward through the graph a second time".
        # detach() keeps pos.grad off any future autograd graph.
        pos.grad = (wl_grad + lam * den_grad).detach()
        torch.nn.utils.clip_grad_norm_([pos], 1.0)
        opt.step()
        with torch.no_grad():
            pos.data.clamp_(lo, hi)
            pos.data[~mov_t] = pos0[~mov_t]

        # Track best by smooth proxy every 100 steps
        if step % 100 == 0:
            smooth = wl.item() + 0.5 * den.item()
            if smooth < best_smooth:
                best_smooth = smooth
                best_np = pos.detach().cpu().numpy().copy()

    # Final check of last position
    with torch.no_grad():
        final_wl = _wl_loss_gpu(pos, gd_full, FD_GAMMA_LO).item()
        final_den = _electrostatic_density_loss_gpu(pos, gd_full, hard_util).item()
        final_smooth = final_wl + 0.5 * final_den
    if final_smooth < best_smooth:
        best_np = pos.detach().cpu().numpy().copy()

    result = best_np.astype(np.float64)
    if not np.isfinite(result).all():
        raise ValueError("gradient produced non-finite positions")
    return result


# ── Multi-strategy legalization ───────────────────────────────────────────────

def _legalize_multi(grad_raw, movable, sizes_np, hw, hh, cw, ch, n_hard,
                     sx, sy, wl_data, density_data, cong_data, cong_w_calib):
    """
    Try several legalization strategies, return the one with the lowest
    calibrated proxy (post-legalize). This prevents a bad legalizer from
    destroying a good gradient placement.
    """
    device = _get_device()
    candidates = []

    # Strategy 1: GPU spread + minimal fix
    try:
        p1 = _gpu_spread_legalize(grad_raw, movable, sizes_np, hw, hh, cw, ch, n_hard, device)
        if _count_overlaps(p1, n_hard, sizes_np) > 0:
            p1 = _minimal_fix(p1, movable, sizes_np, hw, hh, cw, ch, n_hard)
        if _count_overlaps(p1, n_hard, sizes_np) == 0:
            candidates.append(("gpu_spread", p1))
    except Exception:
        pass

    # Strategy 2: Minimal fix only (gentle, preserves gradient as much as possible)
    try:
        p2 = _minimal_fix(grad_raw.copy(), movable, sizes_np, hw, hh, cw, ch, n_hard)
        if _count_overlaps(p2, n_hard, sizes_np) == 0:
            candidates.append(("minimal_fix", p2))
    except Exception:
        pass

    # Strategy 3: Full legalize (most aggressive, last resort)
    try:
        p3 = _legalize(grad_raw.copy(), movable, sizes_np, hw, hh, cw, ch, n_hard,
                       sx, sy, gap=0.002)
        if _count_overlaps(p3, n_hard, sizes_np) == 0:
            candidates.append(("full_legalize", p3))
    except Exception:
        pass

    if not candidates:
        return None, None

    # Pick best by calibrated proxy
    best = None
    best_cost = float('inf')
    for name, p in candidates:
        c, _, _, _ = _calibrated_proxy(p, wl_data, density_data, cong_data, n_hard, cong_w_calib)
        if c < best_cost:
            best_cost = c
            best = (name, p)
    return best[1], best[0]


# ── Main placer ───────────────────────────────────────────────────────────────

class FastDreamPlacer:
    """
    Multi-start DREAMPlace-faithful gradient + multi-legalize + heavy LNS.
    """

    def __init__(self, seed=42):
        self.seed = seed

    def place(self, benchmark: Benchmark) -> torch.Tensor:
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

        n_mov = len(movable_idx)
        sx = (sizes_np[:, 0:1] + sizes_np[:, 0:1].T) / 2
        sy = (sizes_np[:, 1:2] + sizes_np[:, 1:2].T) / 2

        plc = _load_plc(benchmark.name)
        if plc is None:
            return benchmark.macro_positions.clone()
        hard_plc_indices = plc.hard_macro_indices

        wl_data, density_data, cong_data = _build_cost_data(benchmark, plc)
        neighbors = _build_neighbors(wl_data, n_hard)

        init_pos = benchmark.macro_positions[:n_hard].numpy().copy().astype(np.float64)
        leg0 = _minimal_fix(init_pos.copy(), movable, sizes_np, hw, hh, cw, ch, n_hard)

        # Calibrate cong weight from leg0
        leg0_wl = _wl_cost(leg0, wl_data)
        leg0_den = _top_density(_build_density_grid(leg0, density_data, n_hard),
                                density_data['n_top'])
        leg0_cong = _cong_cost(*_build_cong_grid(leg0, cong_data), cong_data)
        # cong_w_calib: prefer fixed 0.5 (matches true proxy = WL + 0.5·den + 0.5·cong).
        # Set FD_CONG_W_CALIB=-1 to revert to the old auto-balanced calibration.
        if FD_CONG_W_CALIB < 0:
            cong_w_calib = (0.5 * (leg0_wl + 0.5 * leg0_den) / leg0_cong
                            if leg0_cong > 1e-9 else 0.5)
        else:
            cong_w_calib = FD_CONG_W_CALIB
        leg0_calib, _, _, _ = _calibrated_proxy(
            leg0, wl_data, density_data, cong_data, n_hard, cong_w_calib)
        if FD_VERBOSE:
            print(f"[FD] leg0_calib={leg0_calib:.4f}  cong_w_calib={cong_w_calib:.4f}  n_mov={n_mov}")

        total_budget = FD_TOTAL_BUDGET
        t0 = time.time()

        best_post_grad = leg0
        best_post_grad_cost = leg0_calib
        best_grad_label = "leg0"

        skip_grad = bool(FD_SKIP_GRAD or FD_POLISH_ONLY)
        if skip_grad:
            if FD_VERBOSE:
                print("[FD] skipping gradient; using leg0 as polish start")
        else:
            # ── Multi-start gradient ────────────────────────────────────────
            per_start = min(FD_GRAD_BUDGET, total_budget * 0.20) / max(1, FD_N_STARTS)
            init_positions = _build_initial_positions(
                FD_N_STARTS, sizes_np, hw, hh, cw, ch, n_hard, movable, leg0)

            # Saddle escape state: count of consecutive rounds without improvement.
            # When >=1 and FD_SADDLE_NOISE > 0, next round inits from perturbed best.
            rounds_since_improve = 0
            sad_rng = np.random.RandomState(self.seed + 7777)

            for idx, init in enumerate(init_positions):
                # If previous round didn't improve and saddle-escape is enabled,
                # restart from a perturbed copy of the current best instead of
                # the planned init. The first round always uses its planned init.
                if (idx > 0 and rounds_since_improve >= 1
                        and FD_SADDLE_NOISE > 0.0 and best_grad_label != "leg0"):
                    noise_scale = FD_SADDLE_NOISE * min(cw, ch)
                    kick = sad_rng.normal(0, noise_scale, size=best_post_grad.shape)
                    # Only perturb movable macros; fixed stay put
                    init = best_post_grad.copy()
                    init[movable] += kick[movable]
                    if FD_VERBOSE:
                        print(f"[FD-grad #{idx}] saddle escape: perturbing "
                              f"best ({best_grad_label}) by σ={noise_scale:.2f}")

                t_grad_start = time.time()
                try:
                    raw = _dreamplace_gradient(
                        init, wl_data, density_data, movable, n_hard,
                        cw, ch, per_start, seed=self.seed + idx * 991)
                except Exception as e:
                    if FD_VERBOSE:
                        print(f"[FD-grad #{idx}] exception: {e}")
                    rounds_since_improve += 1
                    continue

                legal_pos, leg_strategy = _legalize_multi(
                    raw, movable, sizes_np, hw, hh, cw, ch, n_hard, sx, sy,
                    wl_data, density_data, cong_data, cong_w_calib)
                if legal_pos is None:
                    if FD_VERBOSE:
                        print(f"[FD-grad #{idx}] could not legalize")
                    rounds_since_improve += 1
                    continue

                cal, _, _, _ = _calibrated_proxy(
                    legal_pos, wl_data, density_data, cong_data, n_hard, cong_w_calib)
                elapsed = time.time() - t_grad_start
                label = f"grad#{idx}_{leg_strategy}"
                if FD_VERBOSE:
                    print(f"[FD-grad #{idx}] {label} calib={cal:.4f}  "
                          f"(leg0={leg0_calib:.4f}, in {elapsed:.0f}s)")

                if cal < best_post_grad_cost or (FD_FORCE_ACCEPT and best_grad_label == "leg0"):
                    best_post_grad = legal_pos
                    best_post_grad_cost = cal
                    best_grad_label = label
                    rounds_since_improve = 0
                else:
                    rounds_since_improve += 1

        if FD_VERBOSE:
            print(f"[FD] best after gradient: {best_grad_label} calib={best_post_grad_cost:.4f}")

        best_pos = best_post_grad

        # ── Pre-LNS coordinate descent (vmallela-style) ───────────────────────
        cd_frac = 0.10 if skip_grad else 0.04
        cd_cap = 240.0 if skip_grad else 120.0
        cd_budget = max(20.0, min(cd_cap, (total_budget - (time.time() - t0)) * cd_frac))
        try:
            pre_cd = _coordinate_descent(
                best_pos, wl_data, movable_idx, n_hard, sizes_np, sx, sy,
                hw, hh, cw, ch, time_budget_secs=cd_budget, max_sweeps=8)
            pcd_calib, _, _, _ = _calibrated_proxy(
                pre_cd, wl_data, density_data, cong_data, n_hard, cong_w_calib)
            if pcd_calib < best_post_grad_cost:
                if FD_VERBOSE:
                    print(f"[FD-preCD] accepted: {pcd_calib:.4f} < {best_post_grad_cost:.4f}")
                best_pos = pre_cd
                best_post_grad_cost = pcd_calib
            else:
                if FD_VERBOSE:
                    print(f"[FD-preCD] rejected: {pcd_calib:.4f} >= {best_post_grad_cost:.4f}")
        except Exception as e:
            if FD_VERBOSE:
                print(f"[FD-preCD] exception: {e}")

        # ── True-proxy baseline for the LNS acceptance test ───────────────────
        best_true_proxy, _, _, _ = _true_proxy_plc(best_pos, plc, hard_plc_indices, n_hard)
        if best_true_proxy is None:
            best_true_proxy = float('inf')

        # ── Aggressive LNS ────────────────────────────────────────────────────
        K = max(10, min(95 if skip_grad else 80, n_mov // (5 if skip_grad else 6)))
        mini_iters = max(5_000, min(30_000 if skip_grad else 25_000,
                                    (5_000_000 if skip_grad else 4_000_000) // max(1, n_mov)))
        lns_frac = max(FD_LNS_FRAC, 0.78) if skip_grad else FD_LNS_FRAC
        lns_budget = max(60.0, (total_budget - (time.time() - t0)) * lns_frac)
        if FD_VERBOSE:
            print(f"[FD-LNS] K={K} mini_iters={mini_iters} budget={lns_budget:.0f}s")

        best_pos_lns, _, _ = _lns_loop(
            best_pos, wl_data, density_data, cong_data, movable, movable_idx,
            sizes_np, sx, sy, hw, hh, cw, ch, n_hard, neighbors,
            K=K, mini_iters=mini_iters,
            time_budget_secs=lns_budget, seed=self.seed + 17,
            cong_w_calib=cong_w_calib,
        )
        true_lns, _, _, _ = _true_proxy_plc(best_pos_lns, plc, hard_plc_indices, n_hard)
        if true_lns is not None and true_lns < best_true_proxy:
            best_pos = best_pos_lns
            best_true_proxy = true_lns

        # ── Pairwise swap ─────────────────────────────────────────────────────
        swap_budget = min(40.0, max(15.0, total_budget - (time.time() - t0) - 200.0))
        if swap_budget > 10:
            try:
                p = _pairwise_swap_search(
                    best_pos, wl_data, density_data, cong_data, movable_idx,
                    sizes_np, sx, sy, hw, hh, cw, ch, n_hard,
                    time_budget_secs=swap_budget)
                t_s, _, _, _ = _true_proxy_plc(p, plc, hard_plc_indices, n_hard)
                if t_s is not None and t_s < best_true_proxy:
                    best_pos = p
                    best_true_proxy = t_s
            except Exception:
                pass

        # ── Full-proxy CD finishing pass ──────────────────────────────────────
        cd_budget2 = max(30.0, total_budget - (time.time() - t0) - 50.0)
        if cd_budget2 > 20:
            try:
                p = _cd_full_proxy(
                    best_pos, wl_data, density_data, cong_data, movable_idx, n_hard,
                    sizes_np, hw, hh, cw, ch,
                    time_budget_secs=cd_budget2, cong_w_calib=cong_w_calib)
                t_c, _, _, _ = _true_proxy_plc(p, plc, hard_plc_indices, n_hard)
                if t_c is not None and t_c < best_true_proxy:
                    best_pos = p
                    best_true_proxy = t_c
            except Exception:
                pass

        # Final overlap sanity — CRITICAL: check at float32 precision.
        # The returned tensor is float32 (matching benchmark.macro_positions),
        # so we must verify at float32. _minimal_fix(gap=1e-8) is below
        # float32 precision (~3e-6) → touching macros become overlaps after
        # cast. _legalize(gap=0.002) is safe but moves macros aggressively
        # (destroys gradient quality). Sweet spot: _minimal_fix with a
        # slightly larger gap (0.005). Try escalating sequence of fixes.
        def _check_f32_overlaps(p):
            return _count_overlaps(p.astype(np.float32).astype(np.float64),
                                     n_hard, sizes_np)

        # Snapshot the current pre-fix position and its proxy
        best_candidate = None
        best_candidate_proxy = float('inf')

        for attempt, (fn, kwargs) in enumerate([
            (None, None),                               # try the placement as-is first
            (_minimal_fix, {"gap": 5e-3}),              # gentle, preserves WL
            (_minimal_fix, {"gap": 1e-2}),
            (_legalize, {"gap": 2e-3}),                 # more aggressive
            (_legalize, {"gap": 1e-2}),
            (_legalize, {"gap": 5e-2}),                 # last resort
        ]):
            if fn is None:
                candidate = best_pos
            else:
                if fn is _minimal_fix:
                    candidate = fn(best_pos, movable, sizes_np, hw, hh, cw, ch, n_hard, **kwargs)
                else:
                    candidate = fn(best_pos, movable, sizes_np, hw, hh, cw, ch,
                                    n_hard, sx, sy, **kwargs)
            n_ov = _check_f32_overlaps(candidate)
            if n_ov > 0:
                if FD_VERBOSE:
                    label = "no-op" if fn is None else f"{fn.__name__}({kwargs})"
                    print(f"[FD] final attempt {attempt}: {label} → {n_ov} f32-overlaps")
                continue
            # Valid at f32. Compute proxy.
            true_p, _, _, _ = _true_proxy_plc(candidate, plc, hard_plc_indices, n_hard)
            if true_p is None:
                continue
            if FD_VERBOSE:
                label = "no-op" if fn is None else f"{fn.__name__}({kwargs})"
                print(f"[FD] final attempt {attempt}: {label} VALID, true_proxy={true_p:.4f}")
            if true_p < best_candidate_proxy:
                best_candidate = candidate
                best_candidate_proxy = true_p

        if best_candidate is not None:
            best_pos = best_candidate
            best_true_proxy = best_candidate_proxy
        else:
            if FD_VERBOSE:
                print(f"[FD] FALLBACK: returning leg0")
            best_pos = leg0

        if FD_VERBOSE:
            print(f"[FD] DONE. final_true_proxy={best_true_proxy:.4f} "
                  f"elapsed={time.time()-t0:.0f}s")

        out = benchmark.macro_positions.clone()
        for i in range(n_hard):
            out[i, 0] = best_pos[i, 0]
            out[i, 1] = best_pos[i, 1]
        return out


def get_placer(seed=42):
    return FastDreamPlacer(seed=seed)
