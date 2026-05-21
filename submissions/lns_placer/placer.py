"""
LNSPlacer v34 — DreamPlace-style Gradient + LNS + Pairwise Swap + Full-Proxy CD

Algorithm:
0. Gradient placement (DreamPlace-style, ~15% of budget):
   - Log-sum-exp WL + bell-kernel density overflow on GPU (CUDA/MPS/CPU)
   - Phase a: WL warmup (cluster by connectivity)
   - Phase b: WL + density co-optimization (spread while preserving topology)
   - GPU force-spread legalization to remove overlaps
1. Conflict-Driven LNS (~75% of budget): select K macros by conflict score, mini-SA, outer SA
2. Pairwise swap (~1%): greedy O(n²), accept if WL+0.5*density improves
3. Full-Proxy CD (~8%): per-macro WL+density+cong optimization via candidate evaluation
4. Post-CD swap (~1%): same as step 2

Phase transitions use true evaluator proxy (plc.compute_proxy_cost) for acceptance.
Gradient phase replaces initial SA — provides far better starting point for LNS.

Usage:
    uv run evaluate submissions/lns_placer/placer.py
    uv run evaluate submissions/lns_placer/placer.py --all
"""

import math
import os
import random
import time
from pathlib import Path


# ── Experiment flags (set via env vars; defaults preserve current behavior) ──
def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default

def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default

# Number of gradient restarts with different seeds (default 1 = no multistart)
PLACER_MULTISTART      = max(1, _env_int("PLACER_MULTISTART", 1))
# 1 = use adaptive λ schedule (balances WL/density gradient norms, DreamPlace-style)
PLACER_ADAPTIVE_LAMBDA = _env_int("PLACER_ADAPTIVE_LAMBDA", 0)
# 1 = use ELECTROSTATIC density loss (FFT Poisson). 0 = legacy relu² overflow.
PLACER_ELECTROSTATIC   = _env_int("PLACER_ELECTROSTATIC", 1)
# Maximum λ for density penalty (default 500 for relu², ~50 for electrostatic).
PLACER_LAMBDA_MAX      = _env_float("PLACER_LAMBDA_MAX",
                                     50.0 if _env_int("PLACER_ELECTROSTATIC", 1) else 500.0)
# 1 = track best position by smooth-WL+0.5*density during Phase b, return that
#     instead of the final (which can overshoot when λ saturates).
PLACER_BEST_TRACK      = _env_int("PLACER_BEST_TRACK", 1)
# Seconds to run pre-LNS coordinate-descent refinement after gradient (0 = off).
PLACER_CD_PRE_LNS      = _env_float("PLACER_CD_PRE_LNS", 0.0)
# Gradient phase budget in seconds (default: 15% of total budget capped at 400s).
PLACER_GRAD_BUDGET     = _env_float("PLACER_GRAD_BUDGET", -1.0)
# 1 = force gradient even on CPU/MPS (slow but useful for local correctness tests).
PLACER_FORCE_GRAD      = _env_int("PLACER_FORCE_GRAD", 0)
# 1 = bypass the calibrated-proxy guardrail and ALWAYS accept the gradient
# placement (as long as it legalizes). Use to test whether gradient is
# secretly useful but being rejected because it doesn't optimize congestion.
PLACER_FORCE_ACCEPT    = _env_int("PLACER_FORCE_ACCEPT", 0)
# Override total time budget (default 2400s competition spec). Useful for sweeps.
PLACER_TOTAL_BUDGET    = _env_float("PLACER_TOTAL_BUDGET", -1.0)
# 1 = congestion-aware density target: bins with high net demand get lower
# target density, forcing macros away from likely-congested regions.
PLACER_CONG_AWARE      = _env_int("PLACER_CONG_AWARE", 0)
# Strength of congestion-aware target modulation [0=off, 1=strong].
PLACER_CONG_WEIGHT     = _env_float("PLACER_CONG_WEIGHT", 0.5)
# Override gradient learning rate (default: scaled by canvas size, ~0.01-0.05).
PLACER_LR              = _env_float("PLACER_LR", -1.0)
# Override HPWL log-sum-exp smoothing parameter (default 2.0).
# Lower γ = smoother (loose approximation); higher γ = sharper (close to true HPWL).
PLACER_GAMMA           = _env_float("PLACER_GAMMA", -1.0)
# Multiplier on density target. Default 1.0 (target = actual utilization).
# < 1.0 forces tighter spread (apparent overflow even at correct util).
# > 1.0 allows looser packing.
PLACER_DENSITY_TARGET_MULT = _env_float("PLACER_DENSITY_TARGET_MULT", 1.0)


import numpy as np
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
from macro_place.benchmark import Benchmark


# ── PLC loader ────────────────────────────────────────────────────────────────

def _load_plc(name):
    from macro_place.loader import load_benchmark_from_dir, load_benchmark
    root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    if root.exists():
        _, plc = load_benchmark_from_dir(str(root))
        return plc
    ng45 = {
        "ariane133_ng45": "ariane133",
        "ariane136_ng45": "ariane136",
        "nvdla_ng45": "nvdla",
        "mempool_tile_ng45": "mempool_tile",
    }
    d = ng45.get(name)
    if d:
        base = Path("external/MacroPlacement/Flows/NanGate45") / d / "netlist" / "output_CT_Grouping"
        if (base / "netlist.pb.txt").exists():
            _, plc = load_benchmark(str(base / "netlist.pb.txt"), str(base / "initial.plc"))
            return plc
    return None


# ── Cost data extraction ──────────────────────────────────────────────────────

def _build_cost_data(benchmark, plc):
    n_hard = benchmark.num_hard_macros
    cw, ch = plc.get_canvas_width_height()
    net_cnt = max(1, plc.net_cnt)

    plc_idx_to_bidx = {}
    for bidx, plc_idx in enumerate(plc.hard_macro_indices):
        plc_idx_to_bidx[plc_idx] = bidx
    mod_name_to_idx = plc.mod_name_to_indices

    wl_hard_bidxs, wl_hard_ox, wl_hard_oy, wl_hard_net = [], [], [], []
    wl_net_fminx, wl_net_fmaxx, wl_net_fminy, wl_net_fmaxy, wl_net_wt = [], [], [], [], []
    wl_macro_to_nets = [[] for _ in range(n_hard)]
    INF = float('inf')
    net_idx = 0

    for driver_pin_name, sink_pin_names in plc.nets.items():
        all_pins = [driver_pin_name] + list(sink_pin_names)
        driver_idx = mod_name_to_idx[driver_pin_name]
        weight = plc.modules_w_pins[driver_idx].get_weight()
        hard_pins, fxs, fys = [], [], []
        for pin_name in all_pins:
            pin_idx = mod_name_to_idx[pin_name]
            pin = plc.modules_w_pins[pin_idx]
            ptype = pin.get_type()
            if ptype == 'PORT':
                x, y = pin.get_pos(); fxs.append(x); fys.append(y)
            elif ptype == 'MACRO_PIN':
                parent_name = pin.get_macro_name()
                parent_idx = mod_name_to_idx.get(parent_name, -1)
                if parent_idx == -1:
                    continue
                if parent_idx in plc_idx_to_bidx:
                    hard_pins.append((plc_idx_to_bidx[parent_idx], pin.x_offset, pin.y_offset))
                else:
                    px, py = plc.modules_w_pins[parent_idx].get_pos()
                    fxs.append(px + pin.x_offset); fys.append(py + pin.y_offset)
        if len(hard_pins) + len(fxs) < 2:
            continue
        wl_net_fminx.append(min(fxs) if fxs else INF)
        wl_net_fmaxx.append(max(fxs) if fxs else -INF)
        wl_net_fminy.append(min(fys) if fys else INF)
        wl_net_fmaxy.append(max(fys) if fys else -INF)
        wl_net_wt.append(weight)
        for bidx, ox, oy in hard_pins:
            wl_hard_bidxs.append(bidx); wl_hard_ox.append(ox)
            wl_hard_oy.append(oy); wl_hard_net.append(net_idx)
            wl_macro_to_nets[bidx].append(net_idx)
        net_idx += 1

    n_wl_nets = net_idx
    hard_bidxs = np.array(wl_hard_bidxs, dtype=np.int32)
    hard_ox = np.array(wl_hard_ox, dtype=np.float64)
    hard_oy = np.array(wl_hard_oy, dtype=np.float64)
    hard_net = np.array(wl_hard_net, dtype=np.int32)
    si = np.argsort(hard_net, kind='stable')
    hard_bidxs, hard_ox, hard_oy = hard_bidxs[si], hard_ox[si], hard_oy[si]
    hard_net_sorted = hard_net[si]
    unique_nets, starts, _ = np.unique(hard_net_sorted, return_index=True, return_counts=True)

    wl_data = {
        'hard_bidxs': hard_bidxs, 'hard_ox': hard_ox, 'hard_oy': hard_oy,
        'hard_net_sorted': hard_net_sorted,
        'unique_nets': unique_nets, 'starts': starts, 'n_wl_nets': n_wl_nets,
        'fixed_min_x': np.array(wl_net_fminx), 'fixed_max_x': np.array(wl_net_fmaxx),
        'fixed_min_y': np.array(wl_net_fminy), 'fixed_max_y': np.array(wl_net_fmaxy),
        'net_weights': np.array(wl_net_wt),
        'norm': (cw + ch) * net_cnt,
        'macro_to_nets': [list(set(nl)) for nl in wl_macro_to_nets],
    }

    gr, gc = plc.grid_row, plc.grid_col
    gw, gh = cw / gc, ch / gr
    fixed_density = np.zeros((gr, gc), dtype=np.float64)
    for soft_idx in plc.soft_macro_indices:
        mod = plc.modules_w_pins[soft_idx]
        mx, my = mod.get_pos()
        _grid_add_sparse(fixed_density, mx, my, mod.get_width(), mod.get_height(),
                         gr, gc, gw, gh)

    density_data = {
        'fixed_density': fixed_density,
        'hard_sizes': benchmark.macro_sizes[:n_hard].numpy().astype(np.float64),
        'grid_rows': gr, 'grid_cols': gc,
        'grid_cell_w': gw, 'grid_cell_h': gh,
        'n_top': max(1, int(gr * gc * 0.1)),
    }

    hroutes, vroutes = plc.get_routes_per_micron()
    v_cap = max(gw * vroutes, 1e-9)
    h_cap = max(gh * hroutes, 1e-9)

    n_un = len(unique_nets)
    net_hard_pins = [[] for _ in range(n_un)]
    for k, nidx in enumerate(unique_nets):
        lo = starts[k]
        hi = starts[k + 1] if k + 1 < n_un else len(hard_bidxs)
        for p in range(lo, hi):
            net_hard_pins[k].append((int(hard_bidxs[p]), float(hard_ox[p]), float(hard_oy[p])))

    def _pin_to_grid(x, y):
        return (max(0, min(gr - 1, int(y / gh))), max(0, min(gc - 1, int(x / gw))))

    net_fixed_c_min = np.full(n_un, gc, dtype=np.int32)
    net_fixed_c_max = np.full(n_un, -1, dtype=np.int32)
    net_fixed_r_min = np.full(n_un, gr, dtype=np.int32)
    net_fixed_r_max = np.full(n_un, -1, dtype=np.int32)
    for k, nidx in enumerate(unique_nets):
        fx_min, fx_max = wl_data['fixed_min_x'][nidx], wl_data['fixed_max_x'][nidx]
        fy_min, fy_max = wl_data['fixed_min_y'][nidx], wl_data['fixed_max_y'][nidx]
        if fx_min < INF:
            r0, c0 = _pin_to_grid(fx_min, fy_min)
            r1, c1 = _pin_to_grid(fx_max, fy_max)
            net_fixed_c_min[k] = min(c0, c1)
            net_fixed_c_max[k] = max(c0, c1)
            net_fixed_r_min[k] = min(r0, r1)
            net_fixed_r_max[k] = max(r0, r1)

    macro_net_k = [[] for _ in range(n_hard)]
    for k in range(n_un):
        for bidx, _, _ in net_hard_pins[k]:
            macro_net_k[bidx].append(k)
    macro_net_k = [list(set(kl)) for kl in macro_net_k]

    cong_data = {
        'n_un': n_un, 'unique_nets': unique_nets,
        'net_hard_pins': net_hard_pins,
        'net_fixed_c_min': net_fixed_c_min, 'net_fixed_c_max': net_fixed_c_max,
        'net_fixed_r_min': net_fixed_r_min, 'net_fixed_r_max': net_fixed_r_max,
        'net_weights': wl_data['net_weights'],
        'macro_net_k': macro_net_k,
        'grid_rows': gr, 'grid_cols': gc,
        'grid_cell_w': gw, 'grid_cell_h': gh,
        'v_cap': v_cap, 'h_cap': h_cap,
    }
    return wl_data, density_data, cong_data


def _grid_add_sparse(grid, mx, my, mw, mh, gr, gc, gw, gh):
    xl, xh = mx - mw / 2, mx + mw / 2
    yl, yh = my - mh / 2, my + mh / 2
    c_lo = max(0, int(math.floor(xl / gw)))
    c_hi = min(gc - 1, int(math.floor(xh / gw)))
    r_lo = max(0, int(math.floor(yl / gh)))
    r_hi = min(gr - 1, int(math.floor(yh / gh)))
    c_arr = np.arange(c_lo, c_hi + 1)
    r_arr = np.arange(r_lo, r_hi + 1)
    ox = np.maximum(0, np.minimum(xh, (c_arr + 1) * gw) - np.maximum(xl, c_arr * gw)) / gw
    oy = np.maximum(0, np.minimum(yh, (r_arr + 1) * gh) - np.maximum(yl, r_arr * gh)) / gh
    grid[r_lo:r_hi + 1, c_lo:c_hi + 1] += np.outer(oy, ox)


def _macro_contrib(x, y, w, h, gw, gh, gr, gc):
    xl, xh = x - w / 2, x + w / 2
    yl, yh = y - h / 2, y + h / 2
    c_lo = max(0, int(math.floor(xl / gw)))
    c_hi = min(gc - 1, int(math.floor(xh / gw)))
    r_lo = max(0, int(math.floor(yl / gh)))
    r_hi = min(gr - 1, int(math.floor(yh / gh)))
    c_arr = np.arange(c_lo, c_hi + 1)
    r_arr = np.arange(r_lo, r_hi + 1)
    ox = np.maximum(0, np.minimum(xh, (c_arr + 1) * gw) - np.maximum(xl, c_arr * gw)) / gw
    oy = np.maximum(0, np.minimum(yh, (r_arr + 1) * gh) - np.maximum(yl, r_arr * gh)) / gh
    return r_lo, r_hi + 1, c_lo, c_hi + 1, np.outer(oy, ox)


# ── Cost functions ────────────────────────────────────────────────────────────

def _wl_cost(pos, d):
    if len(d['hard_bidxs']) == 0:
        return 0.0
    px = pos[d['hard_bidxs'], 0] + d['hard_ox']
    py = pos[d['hard_bidxs'], 1] + d['hard_oy']
    un, st = d['unique_nets'], d['starts']
    mxx = np.maximum(np.maximum.reduceat(px, st), d['fixed_max_x'][un])
    mnx = np.minimum(np.minimum.reduceat(px, st), d['fixed_min_x'][un])
    mxy = np.maximum(np.maximum.reduceat(py, st), d['fixed_max_y'][un])
    mny = np.minimum(np.minimum.reduceat(py, st), d['fixed_min_y'][un])
    return float((d['net_weights'][un] * ((mxx - mnx) + (mxy - mny))).sum() / d['norm'])


def _top_density(grid, n_top):
    flat = grid.ravel()
    return 0.5 * float(np.mean(np.partition(flat, -n_top)[-n_top:]))


def _build_density_grid(pos, density_data, n_hard):
    gr = density_data['grid_rows']; gc = density_data['grid_cols']
    gw = density_data['grid_cell_w']; gh = density_data['grid_cell_h']
    grid = density_data['fixed_density'].copy()
    sizes = density_data['hard_sizes']
    for i in range(n_hard):
        _grid_add_sparse(grid, pos[i, 0], pos[i, 1], sizes[i, 0], sizes[i, 1], gr, gc, gw, gh)
    return grid


def _net_grid_bbox(k, pos, cong_data):
    gw = cong_data['grid_cell_w']; gh = cong_data['grid_cell_h']
    gr = cong_data['grid_rows']; gc = cong_data['grid_cols']
    c_min = cong_data['net_fixed_c_min'][k]
    c_max = cong_data['net_fixed_c_max'][k]
    r_min = cong_data['net_fixed_r_min'][k]
    r_max = cong_data['net_fixed_r_max'][k]
    for bidx, ox, oy in cong_data['net_hard_pins'][k]:
        px, py = pos[bidx, 0] + ox, pos[bidx, 1] + oy
        c = max(0, min(gc - 1, int(px / gw)))
        r = max(0, min(gr - 1, int(py / gh)))
        c_min = min(c_min, c); c_max = max(c_max, c)
        r_min = min(r_min, r); r_max = max(r_max, r)
    return c_min, c_max, r_min, r_max


def _build_cong_grid(pos, cong_data):
    gc = cong_data['grid_cols']; gr = cong_data['grid_rows']
    V_col = np.zeros(gc, dtype=np.float64)
    H_row = np.zeros(gr, dtype=np.float64)
    weights = cong_data['net_weights']
    unique_nets = cong_data['unique_nets']
    for k in range(cong_data['n_un']):
        c_min, c_max, r_min, r_max = _net_grid_bbox(k, pos, cong_data)
        w = weights[unique_nets[k]]
        if c_max > c_min: V_col[c_min:c_max] += w
        if r_max > r_min: H_row[r_min:r_max] += w
    return V_col, H_row


def _cong_cost(V_col, H_row, cong_data):
    return float(np.max(V_col) / cong_data['v_cap'] + np.max(H_row) / cong_data['h_cap'])


def _proxy(pos, wl_data, density_data, cong_data, n_hard):
    wl = _wl_cost(pos, wl_data)
    den = _top_density(_build_density_grid(pos, density_data, n_hard), density_data['n_top'])
    cong = _cong_cost(*_build_cong_grid(pos, cong_data), cong_data)
    return wl + 0.5 * den + 0.5 * cong, wl, den, cong


def _calibrated_proxy(pos, wl_data, density_data, cong_data, n_hard, cong_w_calib):
    """Same as _proxy but with calibrated cong weight matching SA's auto-calibration.
    1D cong is ~12× larger than true cong, so raw 0.5 weight makes _proxy cong-dominated.
    Using the SA-calibrated weight makes comparisons track true proxy changes correctly."""
    wl = _wl_cost(pos, wl_data)
    den = _top_density(_build_density_grid(pos, density_data, n_hard), density_data['n_top'])
    cong = _cong_cost(*_build_cong_grid(pos, cong_data), cong_data)
    return wl + 0.5 * den + cong_w_calib * cong, wl, den, cong


def _true_proxy_plc(pos, plc, hard_plc_indices, n_hard):
    """Sync pos to plc and evaluate true evaluator proxy (WL + 0.5*den + 0.5*cong).
    Imports macro_place.objective to ensure the boundary monkey-patch is active.
    Returns (proxy, wl, den, cong) or (None, None, None, None) on failure."""
    from macro_place.objective import compute_proxy_cost  # noqa: F401 — triggers monkey-patch
    for bidx in range(n_hard):
        node_idx = hard_plc_indices[bidx]
        x = float(pos[bidx, 0])
        y = float(pos[bidx, 1])
        node = plc.modules_w_pins[node_idx]
        if hasattr(node, "set_pos"):
            node.set_pos(x, y)
        else:
            plc.update_node_coords(node_idx, x, y)

        if not hasattr(plc, "_lns_macro_pin_map"):
            pin_map = {}
            for idx, mod in enumerate(plc.modules_w_pins):
                if mod.get_type() == "MACRO_PIN" and hasattr(mod, "get_macro_name"):
                    pin_map.setdefault(mod.get_macro_name(), []).append(idx)
            plc._lns_macro_pin_map = pin_map
        for pin_idx in plc._lns_macro_pin_map.get(node.get_name(), []):
            pin = plc.modules_w_pins[pin_idx]
            pin.set_pos(x + pin.x_offset, y + pin.y_offset)
    try:
        expected_size = plc.grid_col * plc.grid_row
        if len(plc.H_routing_cong) != expected_size:
            plc.V_routing_cong = [0] * expected_size
            plc.H_routing_cong = [0] * expected_size
            plc.V_macro_routing_cong = [0] * expected_size
            plc.H_macro_routing_cong = [0] * expected_size
        plc.FLAG_UPDATE_WIRELENGTH = True
        plc.FLAG_UPDATE_DENSITY = True
        plc.FLAG_UPDATE_CONGESTION = True
        wl = plc.get_cost()
        den = plc.get_density_cost()
        cong = plc.get_congestion_cost()
        return wl + 0.5 * den + 0.5 * cong, wl, den, cong
    except Exception:
        return None, None, None, None


# ── Neighbour list ────────────────────────────────────────────────────────────

def _build_neighbors(wl_data, n_hard):
    net_to_macros = {}
    hard_bidxs = wl_data['hard_bidxs']
    hard_net_sorted = wl_data['hard_net_sorted']
    for i in range(len(hard_bidxs)):
        ni, bi = int(hard_net_sorted[i]), int(hard_bidxs[i])
        net_to_macros.setdefault(ni, []).append(bi)
    adj = [set() for _ in range(n_hard)]
    for macros in net_to_macros.values():
        for a in macros:
            for b in macros:
                if a != b:
                    adj[a].add(b)
    return [list(s) for s in adj]


# ── Legalization utilities ────────────────────────────────────────────────────

def _minimal_fix(pos, movable, sizes, hw, hh, cw, ch, n_hard, gap=1e-8):
    pos = pos.copy().astype(np.float64)
    sizes_f32 = sizes.astype(np.float32)
    sx = (sizes_f32[:, 0:1] + sizes_f32[:, 0:1].T) / 2.0
    sy = (sizes_f32[:, 1:2] + sizes_f32[:, 1:2].T) / 2.0
    for _ in range(300):
        forces = np.zeros_like(pos)
        any_overlap = False
        for i in range(n_hard):
            for j in range(i + 1, n_hard):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                ox = sx[i, j] + gap - abs(dx)
                oy = sy[i, j] + gap - abs(dy)
                if ox <= 0 or oy <= 0:
                    continue
                any_overlap = True
                if ox <= oy:
                    push = ox / 2 + 1e-8
                    sign = 1.0 if dx >= 0 else -1.0
                    if movable[i]: forces[i, 0] += sign * push
                    if movable[j]: forces[j, 0] -= sign * push
                else:
                    push = oy / 2 + 1e-8
                    sign = 1.0 if dy >= 0 else -1.0
                    if movable[i]: forces[i, 1] += sign * push
                    if movable[j]: forces[j, 1] -= sign * push
        if not any_overlap:
            break
        pos[:, 0] = np.clip(pos[:, 0] + forces[:, 0], hw, cw - hw)
        pos[:, 1] = np.clip(pos[:, 1] + forces[:, 1], hh, ch - hh)
    return pos


def _legalize(pos, movable, sizes, hw, hh, cw, ch, n, sx, sy, gap=0.02):
    order = sorted(range(n), key=lambda i: -sizes[i, 0] * sizes[i, 1])
    placed = np.zeros(n, dtype=bool)
    legal = pos.copy()
    ia = np.arange(n)
    for idx in order:
        if not movable[idx]:
            placed[idx] = True; continue
        if placed.any():
            dx = np.abs(legal[idx, 0] - legal[:, 0])
            dy = np.abs(legal[idx, 1] - legal[:, 1])
            if not ((dx < sx[idx] + gap) & (dy < sy[idx] + gap) & placed & (ia != idx)).any():
                placed[idx] = True; continue
        step = max(sizes[idx, 0], sizes[idx, 1]) * 0.5
        best_p = legal[idx].copy(); best_d = float('inf')
        for r in range(1, 200):
            found = False
            for dxm in range(-r, r + 1):
                for dym in range(-r, r + 1):
                    if abs(dxm) != r and abs(dym) != r:
                        continue
                    cx_ = np.clip(pos[idx, 0] + dxm * step, hw[idx], cw - hw[idx])
                    cy_ = np.clip(pos[idx, 1] + dym * step, hh[idx], ch - hh[idx])
                    if placed.any():
                        if ((np.abs(cx_ - legal[:, 0]) < sx[idx] + gap) &
                                (np.abs(cy_ - legal[:, 1]) < sy[idx] + gap) &
                                placed & (ia != idx)).any():
                            continue
                    d = (cx_ - pos[idx, 0]) ** 2 + (cy_ - pos[idx, 1]) ** 2
                    if d < best_d:
                        best_d = d; best_p = np.array([cx_, cy_]); found = True
            if found:
                break
        legal[idx] = best_p; placed[idx] = True
    return legal


def _count_overlaps(pos, n_hard, sizes):
    sizes_f32 = sizes.astype(np.float32)
    count = 0
    for i in range(n_hard):
        for j in range(i + 1, n_hard):
            ox = (sizes_f32[i, 0] + sizes_f32[j, 0]) / 2.0 - abs(pos[i, 0] - pos[j, 0])
            oy = (sizes_f32[i, 1] + sizes_f32[j, 1]) / 2.0 - abs(pos[i, 1] - pos[j, 1])
            if ox > 0 and oy > 0:
                count += 1
    return count


# ── SA (used for both initial SA and mini-SA within LNS) ─────────────────────

def _sa(pos, wl_data, density_data, cong_data, movable, movable_idx,
        sizes, sx, sy, hw, hh, cw, ch, n_hard, neighbors,
        n_iters, seed, T_start_frac=0.05, T_end_frac=0.00005,
        wl_w=1.0, den_w=0.5, cong_w=0.5, cong_w_override=None):
    """
    SA that can operate on a restricted movable set (for mini-SA in LNS/ILS).
    cong_w_override: if set, skips auto-calibration and uses this fixed weight.
    This enables cross-run cost comparisons when all runs share the same calibration.
    """
    rng = random.Random(seed)
    pos = pos.copy()
    movable_list = list(movable_idx)

    gr = density_data['grid_rows']; gc = density_data['grid_cols']
    gw = density_data['grid_cell_w']; gh = density_data['grid_cell_h']
    n_top = density_data['n_top']
    hard_sizes = density_data['hard_sizes']

    grid = _build_density_grid(pos, density_data, n_hard)
    cur_density = _top_density(grid, n_top)
    cur_wl = _wl_cost(pos, wl_data)
    V_col, H_row = _build_cong_grid(pos, cong_data)
    cur_cong = _cong_cost(V_col, H_row, cong_data)
    if cong_w_override is not None:
        cong_w = cong_w_override
    elif cur_cong > 1e-9:
        cong_w = cong_w * (wl_w * cur_wl + den_w * cur_density) / cur_cong
    cur_cost = wl_w * cur_wl + den_w * cur_density + cong_w * cur_cong
    best_pos = pos.copy(); best_cost = cur_cost

    macro_net_k = cong_data['macro_net_k']
    net_weights = cong_data['net_weights']
    cong_unique_nets = cong_data['unique_nets']

    net_bbox = np.zeros((cong_data['n_un'], 4), dtype=np.int32)
    for k in range(cong_data['n_un']):
        c0, c1, r0, r1 = _net_grid_bbox(k, pos, cong_data)
        net_bbox[k] = (c0, c1, r0, r1)

    T_start = max(cw, ch) * T_start_frac
    T_end = max(cw, ch) * T_end_frac
    log_ratio = math.log(T_end / T_start)

    sizes_f32 = hard_sizes.astype(np.float32)
    hs_x = (sizes_f32[:, 0:1] + sizes_f32[:, 0:1].T) / 2.0
    hs_y = (sizes_f32[:, 1:2] + sizes_f32[:, 1:2].T) / 2.0

    def ok(idx):
        ox = hs_x[idx] - np.abs(pos[idx, 0] - pos[:, 0])
        oy = hs_y[idx] - np.abs(pos[idx, 1] - pos[:, 1])
        has_overlap = (ox > 0) & (oy > 0)
        has_overlap[idx] = False
        return not has_overlap.any()

    for step in range(n_iters):
        frac = step / n_iters
        T = T_start * math.exp(log_ratio * frac)
        i = rng.choice(movable_list)
        old_x, old_y = pos[i, 0], pos[i, 1]
        w_i, h_i = float(hard_sizes[i, 0]), float(hard_sizes[i, 1])
        move = rng.random()

        do_swap = False
        j = -1
        if move < 0.45:
            shift = T * (0.4 + 0.6 * (1 - frac))
            new_x = float(np.clip(old_x + rng.gauss(0, shift), hw[i], cw - hw[i]))
            new_y = float(np.clip(old_y + rng.gauss(0, shift), hh[i], ch - hh[i]))
        elif move < 0.80:
            cands = [k for k in neighbors[i] if movable[k]] if neighbors[i] else []
            j = rng.choice(cands) if (cands and rng.random() < 0.65) else rng.choice(movable_list)
            if i == j: continue
            do_swap = True
            old_jx, old_jy = pos[j, 0], pos[j, 1]
            new_x = float(np.clip(old_jx, hw[i], cw - hw[i]))
            new_y = float(np.clip(old_jy, hh[i], ch - hh[i]))
            new_jx = float(np.clip(old_x, hw[j], cw - hw[j]))
            new_jy = float(np.clip(old_y, hh[j], ch - hh[j]))
        else:
            if not neighbors[i]: continue
            j_idx = rng.choice(neighbors[i])
            alpha = rng.uniform(0.05, 0.30)
            new_x = float(np.clip(old_x + alpha * (pos[j_idx, 0] - old_x), hw[i], cw - hw[i]))
            new_y = float(np.clip(old_y + alpha * (pos[j_idx, 1] - old_y), hh[i], ch - hh[i]))

        pos[i, 0] = new_x; pos[i, 1] = new_y
        if do_swap:
            pos[j, 0] = new_jx; pos[j, 1] = new_jy
            if not ok(i) or not ok(j):
                pos[i, 0] = old_x; pos[i, 1] = old_y
                pos[j, 0] = old_jx; pos[j, 1] = old_jy; continue
        else:
            if not ok(i):
                pos[i, 0] = old_x; pos[i, 1] = old_y; continue

        r_lo_o, r_hi_o, c_lo_o, c_hi_o, old_c = _macro_contrib(old_x, old_y, w_i, h_i, gw, gh, gr, gc)
        r_lo_n, r_hi_n, c_lo_n, c_hi_n, new_c = _macro_contrib(new_x, new_y, w_i, h_i, gw, gh, gr, gc)
        grid[r_lo_o:r_hi_o, c_lo_o:c_hi_o] -= old_c
        grid[r_lo_n:r_hi_n, c_lo_n:c_hi_n] += new_c

        if do_swap:
            w_j, h_j = float(hard_sizes[j, 0]), float(hard_sizes[j, 1])
            r_lo_jo, r_hi_jo, c_lo_jo, c_hi_jo, old_jc = _macro_contrib(
                old_jx, old_jy, w_j, h_j, gw, gh, gr, gc)
            r_lo_jn, r_hi_jn, c_lo_jn, c_hi_jn, new_jc = _macro_contrib(
                new_jx, new_jy, w_j, h_j, gw, gh, gr, gc)
            grid[r_lo_jo:r_hi_jo, c_lo_jo:c_hi_jo] -= old_jc
            grid[r_lo_jn:r_hi_jn, c_lo_jn:c_hi_jn] += new_jc

        new_density = _top_density(grid, n_top)
        new_wl = _wl_cost(pos, wl_data)

        affected_nets = list(macro_net_k[i])
        if do_swap:
            for k2 in macro_net_k[j]:
                if k2 not in macro_net_k[i]:
                    affected_nets.append(k2)
        old_net_bbox = {}
        for k2 in affected_nets:
            old_net_bbox[k2] = tuple(net_bbox[k2])
            old_c_min, old_c_max, old_r_min, old_r_max = net_bbox[k2]
            w = net_weights[cong_unique_nets[k2]]
            if old_c_max > old_c_min: V_col[old_c_min:old_c_max] -= w
            if old_r_max > old_r_min: H_row[old_r_min:old_r_max] -= w
            new_c_min2, new_c_max2, new_r_min2, new_r_max2 = _net_grid_bbox(k2, pos, cong_data)
            if new_c_max2 > new_c_min2: V_col[new_c_min2:new_c_max2] += w
            if new_r_max2 > new_r_min2: H_row[new_r_min2:new_r_max2] += w
            net_bbox[k2] = (new_c_min2, new_c_max2, new_r_min2, new_r_max2)
        new_cong = _cong_cost(V_col, H_row, cong_data)

        delta_cost = (wl_w * (new_wl - cur_wl) + den_w * (new_density - cur_density)
                      + cong_w * (new_cong - cur_cong))

        if delta_cost < 0 or rng.random() < math.exp(-delta_cost / max(T, 1e-12)):
            cur_wl = new_wl; cur_density = new_density; cur_cong = new_cong
            cur_cost += delta_cost
            if cur_cost < best_cost:
                best_cost = cur_cost; best_pos = pos.copy()
        else:
            pos[i, 0] = old_x; pos[i, 1] = old_y
            grid[r_lo_n:r_hi_n, c_lo_n:c_hi_n] -= new_c
            grid[r_lo_o:r_hi_o, c_lo_o:c_hi_o] += old_c
            if do_swap:
                pos[j, 0] = old_jx; pos[j, 1] = old_jy
                grid[r_lo_jn:r_hi_jn, c_lo_jn:c_hi_jn] -= new_jc
                grid[r_lo_jo:r_hi_jo, c_lo_jo:c_hi_jo] += old_jc
            for k2 in affected_nets:
                new_c_min2, new_c_max2, new_r_min2, new_r_max2 = tuple(net_bbox[k2])
                w = net_weights[cong_unique_nets[k2]]
                if new_c_max2 > new_c_min2: V_col[new_c_min2:new_c_max2] -= w
                if new_r_max2 > new_r_min2: H_row[new_r_min2:new_r_max2] -= w
                old_c_min, old_c_max, old_r_min, old_r_max = old_net_bbox[k2]
                if old_c_max > old_c_min: V_col[old_c_min:old_c_max] += w
                if old_r_max > old_r_min: H_row[old_r_min:old_r_max] += w
                net_bbox[k2] = old_net_bbox[k2]

    return best_pos, best_cost


# ── Conflict scoring ──────────────────────────────────────────────────────────

def _conflict_scores(pos, wl_data, density_data, cong_data, n_hard, movable):
    """
    Score each movable macro by its contribution to WL, density, and congestion.
    WL dominates by design (layout units vs [0,1] density/cong) because the true
    proxy weights WL at 1.0 — WL-heavy macros are the right targets for LNS.
    """
    wl_sc  = np.zeros(n_hard)
    den_sc = np.zeros(n_hard)
    cong_sc = np.zeros(n_hard)

    # WL: per-net WL, then sum over nets connected to each macro
    if len(wl_data['hard_bidxs']) > 0:
        px = pos[wl_data['hard_bidxs'], 0] + wl_data['hard_ox']
        py = pos[wl_data['hard_bidxs'], 1] + wl_data['hard_oy']
        un, st = wl_data['unique_nets'], wl_data['starts']
        mxx = np.maximum(np.maximum.reduceat(px, st), wl_data['fixed_max_x'][un])
        mnx = np.minimum(np.minimum.reduceat(px, st), wl_data['fixed_min_x'][un])
        mxy = np.maximum(np.maximum.reduceat(py, st), wl_data['fixed_max_y'][un])
        mny = np.minimum(np.minimum.reduceat(py, st), wl_data['fixed_min_y'][un])
        net_wl_un = wl_data['net_weights'][un] * ((mxx - mnx) + (mxy - mny))
        net_wl_full = np.zeros(wl_data['n_wl_nets'])
        net_wl_full[un] = net_wl_un
        macro_to_nets = wl_data['macro_to_nets']
        for i in range(n_hard):
            if movable[i] and macro_to_nets[i]:
                wl_sc[i] = sum(net_wl_full[ni] for ni in macro_to_nets[i])

    # Density: occupancy of the cell(s) under each macro
    gr = density_data['grid_rows']; gc = density_data['grid_cols']
    gw = density_data['grid_cell_w']; gh = density_data['grid_cell_h']
    grid = _build_density_grid(pos, density_data, n_hard)
    for i in range(n_hard):
        if movable[i]:
            r = max(0, min(gr - 1, int(pos[i, 1] / gh)))
            c = max(0, min(gc - 1, int(pos[i, 0] / gw)))
            den_sc[i] = grid[r, c]

    # Congestion: peak V/H demand in channels spanned by connected nets
    V_col, H_row = _build_cong_grid(pos, cong_data)
    macro_net_k = cong_data['macro_net_k']
    v_cap = cong_data['v_cap']; h_cap = cong_data['h_cap']
    for i in range(n_hard):
        if movable[i] and macro_net_k[i]:
            cong_contrib = 0.0
            for k in macro_net_k[i]:
                c_min, c_max, r_min, r_max = int(cong_data['net_fixed_c_min'][k]), int(cong_data['net_fixed_c_max'][k]), int(cong_data['net_fixed_r_min'][k]), int(cong_data['net_fixed_r_max'][k])
                gw_ = cong_data['grid_cell_w']; gh_ = cong_data['grid_cell_h']
                gc_ = cong_data['grid_cols']; gr_ = cong_data['grid_rows']
                for bidx, ox, oy in cong_data['net_hard_pins'][k]:
                    px2, py2 = pos[bidx, 0] + ox, pos[bidx, 1] + oy
                    c2 = max(0, min(gc_ - 1, int(px2 / gw_)))
                    r2 = max(0, min(gr_ - 1, int(py2 / gh_)))
                    c_min = min(c_min, c2); c_max = max(c_max, c2)
                    r_min = min(r_min, r2); r_max = max(r_max, r2)
                if c_max > c_min:
                    cong_contrib += float(np.max(V_col[c_min:c_max])) / v_cap
                if r_max > r_min:
                    cong_contrib += float(np.max(H_row[r_min:r_max])) / h_cap
            cong_sc[i] = cong_contrib

    # WL-dominated selection: wl_sc is raw canvas units (much larger than den/cong [0,3]).
    # Empirically better than normalizing: WL-heavy macros are naturally the right targets
    # because WL minimization aligns with cong minimization (packing = less routing demand).
    # Normalizing to select congested macros hurt ibm04/ibm06 (cong topology-constrained).
    return wl_sc + den_sc + cong_sc


# ── LNS outer loop ────────────────────────────────────────────────────────────

def _lns(pos, wl_data, density_data, cong_data, movable, movable_idx,
         sizes_np, sx, sy, hw, hh, cw, ch, n_hard, neighbors,
         K, mini_iters, time_budget_secs, seed, cong_w_calib=0.5,
         T_outer_start=0.02):
    """
    Conflict-Driven LNS with simulated annealing outer acceptance.

    Outer SA acceptance allows escaping local minima that greedy LNS gets stuck in.
    At high T_outer: aggressively explore by accepting worse solutions + scattering
    selected macros randomly (ruin-and-recreate). At low T_outer: fine-tune greedily.

    T_outer schedule: T_outer_start → 1e-5 over the time budget (1000× reduction).
    cong_w_calib: SA-calibrated cong weight; makes comparisons track true proxy correctly.
    T_outer_start: initial outer acceptance temperature (default 0.02).
    """
    rng = random.Random(seed)
    pos = pos.copy()
    movable_list = list(movable_idx)

    cur_proxy, _, _, _ = _calibrated_proxy(pos, wl_data, density_data, cong_data, n_hard, cong_w_calib)
    best_proxy = cur_proxy
    best_pos = pos.copy()

    # Outer SA temperature: controls acceptance of worse solutions between LNS steps
    T_outer_end = 1e-5
    log_ratio_outer = math.log(T_outer_end / T_outer_start)

    t0 = time.time()
    outer_iter = 0
    iters_no_best_improve = 0
    MAX_STAGNATION = max(60, K * 5)   # force scatter when stuck this long

    while time.time() - t0 < time_budget_secs:
        outer_iter += 1
        frac = min(1.0, (time.time() - t0) / time_budget_secs)
        T_outer = T_outer_start * math.exp(log_ratio_outer * frac)

        scores = _conflict_scores(pos, wl_data, density_data, cong_data, n_hard, movable)
        scored = sorted(movable_list, key=lambda i: -scores[i])

        K_conflict = max(1, K * 3 // 5)
        K_random = K - K_conflict
        top_pool = scored[:min(K_conflict * 3, len(scored))]
        selected = list(rng.sample(top_pool, min(K_conflict, len(top_pool))))
        sel_set = set(selected)
        remaining = [i for i in movable_list if i not in sel_set]
        if remaining and K_random > 0:
            selected += rng.sample(remaining, min(K_random, len(remaining)))

        # Build mini-movable mask
        mini_movable = np.zeros(n_hard, dtype=bool)
        for i in selected:
            mini_movable[i] = True
        mini_movable_idx = np.array(selected, dtype=np.int32)

        # Ruin-and-recreate: scatter selected macros randomly.
        # Normal: scatter probability ∝ T_outer (explore early, exploit late).
        # Stagnated: 40% chance to scatter; T_frac-mixing chooses global vs local.
        # Early stagnations: global scatter toward chip center (escape bad global arrangements).
        # Late stagnations: local scatter around best_pos (sigma=0.08, fine-tuning).
        # T_frac = T_outer/T_outer_start ≈ 0 for 96% of run → almost always local scatter.
        is_stagnated = iters_no_best_improve > MAX_STAGNATION
        scatter_prob = 0.4 if is_stagnated else min(0.8, T_outer * 30)
        start_pos = pos.copy()
        if rng.random() < scatter_prob:
            if is_stagnated:
                iters_no_best_improve = 0
                T_frac_cur = T_outer / T_outer_start
                use_global = (rng.random() < T_frac_cur)
                if use_global:
                    # Global scatter: toward chip center (escape bad global arrangements)
                    for i in selected:
                        start_pos[i, 0] = float(np.clip(
                            rng.gauss(cw * 0.5, cw * 0.12), hw[i], cw - hw[i]))
                        start_pos[i, 1] = float(np.clip(
                            rng.gauss(ch * 0.5, ch * 0.12), hh[i], ch - hh[i]))
                else:
                    # Local scatter: best_pos ± 8% chip size
                    for i in selected:
                        start_pos[i, 0] = float(np.clip(
                            rng.gauss(best_pos[i, 0], cw * 0.08), hw[i], cw - hw[i]))
                        start_pos[i, 1] = float(np.clip(
                            rng.gauss(best_pos[i, 1], ch * 0.08), hh[i], ch - hh[i]))
            else:
                for i in selected:
                    start_pos[i, 0] = float(np.clip(
                        rng.gauss(cw * 0.5, cw * 0.3), hw[i], cw - hw[i]))
                    start_pos[i, 1] = float(np.clip(
                        rng.gauss(ch * 0.5, ch * 0.3), hh[i], ch - hh[i]))

        mini_T_floor = 0.005
        mini_T_start = max(mini_T_floor, min(0.15, T_outer * 6))
        mini_T_end = max(0.00005, mini_T_start * 0.001)

        # Mini-SA uses cong_w_calib for consistency with outer LNS acceptance.
        # Reduce mini-SA iterations when T_outer is low (greedy phase): fewer iterations
        # suffice for convergence, freeing budget for more outer LNS iterations.
        # min_frac=0.15 gives more outer iterations in the greedy phase.
        # 0.5 tested worse than 0.25, continuing trend toward more outer iters.
        T_frac = T_outer / T_outer_start  # 1.0 at start, ~0 at end
        eff_mini_iters = max(1000, int(mini_iters * max(0.15, min(1.0, T_frac * 4))))
        new_pos, _ = _sa(
            start_pos, wl_data, density_data, cong_data,
            mini_movable, mini_movable_idx,
            sizes_np, sx, sy, hw, hh, cw, ch, n_hard, neighbors,
            n_iters=eff_mini_iters, seed=rng.randint(0, 2**31),
            T_start_frac=mini_T_start,
            T_end_frac=mini_T_end,
            cong_w_override=cong_w_calib,
        )

        new_proxy, _, _, _ = _calibrated_proxy(new_pos, wl_data, density_data, cong_data, n_hard, cong_w_calib)
        delta = new_proxy - cur_proxy

        # Outer SA acceptance: accept improvement always, worse solutions with exp(-delta/T)
        if delta < 0 or rng.random() < math.exp(-delta / max(T_outer, 1e-9)):
            pos = new_pos.copy()
            cur_proxy = new_proxy
            if new_proxy < best_proxy:
                best_proxy = new_proxy
                best_pos = new_pos.copy()
                iters_no_best_improve = 0
            else:
                iters_no_best_improve += 1
        else:
            iters_no_best_improve += 1

    return best_pos, best_proxy, outer_iter


# ── Pairwise swap search ──────────────────────────────────────────────────────

def _pairwise_swap_search(pos, wl_data, density_data, cong_data, movable_idx,
                          sizes_np, sx, sy, hw, hh, cw, ch, n_hard,
                          time_budget_secs):
    """
    Greedy pairwise swap search with incremental density and WL evaluation.
    Per-pair cost: O(affected_nets) instead of O(n_hard) full rebuild → ~100× speedup.
    Accepts swap if WL + 0.5*density strictly decreases; true-proxy guardrail runs externally.
    """
    t0 = time.time()
    pos = pos.copy()
    movable_list = list(movable_idx)
    n_mov = len(movable_list)

    # Density incremental state
    gr = density_data['grid_rows']; gc = density_data['grid_cols']
    gw = density_data['grid_cell_w']; gh = density_data['grid_cell_h']
    n_top = density_data['n_top']
    hard_sizes = density_data['hard_sizes']

    # WL incremental state
    hard_bidxs = wl_data['hard_bidxs']
    hard_ox = wl_data['hard_ox']
    hard_oy = wl_data['hard_oy']
    unique_nets = wl_data['unique_nets']
    starts = wl_data['starts']
    fixed_min_x = wl_data['fixed_min_x']; fixed_max_x = wl_data['fixed_max_x']
    fixed_min_y = wl_data['fixed_min_y']; fixed_max_y = wl_data['fixed_max_y']
    net_weights = wl_data['net_weights']
    norm = wl_data['norm']
    n_un = len(unique_nets)
    # macro_net_k[bidx] = list of k-indices into unique_nets containing this macro
    macro_net_k = cong_data['macro_net_k']

    def _net_bbox_k(k, cur_pos):
        """Recompute bbox for single net k (used after accepted swaps)."""
        lo = starts[k]
        hi = starts[k + 1] if k + 1 < n_un else len(hard_bidxs)
        nidx = unique_nets[k]
        slc_bx = cur_pos[hard_bidxs[lo:hi], 0] + hard_ox[lo:hi]
        slc_by = cur_pos[hard_bidxs[lo:hi], 1] + hard_oy[lo:hi]
        return (min(float(slc_bx.min()), float(fixed_min_x[nidx])),
                max(float(slc_bx.max()), float(fixed_max_x[nidx])),
                min(float(slc_by.min()), float(fixed_min_y[nidx])),
                max(float(slc_by.max()), float(fixed_max_y[nidx])))

    def _build_all_bboxes(cur_pos):
        """Vectorized: build all net bboxes at once using reduceat (same as _wl_cost)."""
        px = cur_pos[hard_bidxs, 0] + hard_ox
        py = cur_pos[hard_bidxs, 1] + hard_oy
        mxx = np.maximum(np.maximum.reduceat(px, starts), fixed_max_x[unique_nets])
        mnx = np.minimum(np.minimum.reduceat(px, starts), fixed_min_x[unique_nets])
        mxy = np.maximum(np.maximum.reduceat(py, starts), fixed_max_y[unique_nets])
        mny = np.minimum(np.minimum.reduceat(py, starts), fixed_min_y[unique_nets])
        return list(zip(mnx.tolist(), mxx.tolist(), mny.tolist(), mxy.tolist()))

    # Vectorized overlap check (numpy, avoids Python loop over n_hard)
    def overlap_free_v(idx, nx, ny, skip):
        dxv = np.abs(nx - pos[:, 0])
        dyv = np.abs(ny - pos[:, 1])
        mask = np.ones(n_hard, dtype=bool)
        mask[idx] = False
        mask[skip] = False
        return not bool(np.any((dxv < sx[idx]) & (dyv < sy[idx]) & mask))

    any_pass_improved = True
    while any_pass_improved and (time.time() - t0) < time_budget_secs:
        any_pass_improved = False

        # Build grid and net bboxes once per pass (vectorized, fast)
        grid = _build_density_grid(pos, density_data, n_hard)
        net_bboxes = _build_all_bboxes(pos)
        cur_wl = _wl_cost(pos, wl_data)
        cur_den = _top_density(grid, n_top)
        cur_score = cur_wl + 0.5 * cur_den

        for a in range(n_mov):
            if time.time() - t0 >= time_budget_secs:
                break
            i = movable_list[a]
            for b in range(a + 1, n_mov):
                if time.time() - t0 >= time_budget_secs:
                    break
                j = movable_list[b]
                old_xi, old_yi = pos[i, 0], pos[i, 1]
                old_xj, old_yj = pos[j, 0], pos[j, 1]

                new_xi = float(np.clip(old_xj, hw[i], cw - hw[i]))
                new_yi = float(np.clip(old_yj, hh[i], ch - hh[i]))
                new_xj = float(np.clip(old_xi, hw[j], cw - hw[j]))
                new_yj = float(np.clip(old_yi, hh[j], ch - hh[j]))

                if (abs(new_xi - new_xj) < sx[i, j] and
                        abs(new_yi - new_yj) < sy[i, j]):
                    continue
                if not overlap_free_v(i, new_xi, new_yi, j):
                    continue
                if not overlap_free_v(j, new_xj, new_yj, i):
                    continue

                # Tentative swap
                pos[i, 0] = new_xi; pos[i, 1] = new_yi
                pos[j, 0] = new_xj; pos[j, 1] = new_yj

                # Incremental WL: only recompute affected nets
                affected = list(set(macro_net_k[i]) | set(macro_net_k[j]))
                old_contrib = sum(
                    float(net_weights[unique_nets[k]]) * ((net_bboxes[k][1] - net_bboxes[k][0]) +
                                                           (net_bboxes[k][3] - net_bboxes[k][2]))
                    for k in affected)
                new_bb = {k: _net_bbox_k(k, pos) for k in affected}
                new_contrib = sum(
                    float(net_weights[unique_nets[k]]) * ((new_bb[k][1] - new_bb[k][0]) +
                                                           (new_bb[k][3] - new_bb[k][2]))
                    for k in affected)
                new_wl = cur_wl + (new_contrib - old_contrib) / norm

                # Incremental density: 4 _macro_contrib updates
                wi, hi_m = float(hard_sizes[i, 0]), float(hard_sizes[i, 1])
                wj, hj_m = float(hard_sizes[j, 0]), float(hard_sizes[j, 1])
                rlo_oi, rhi_oi, clo_oi, chi_oi, oc_i = _macro_contrib(
                    old_xi, old_yi, wi, hi_m, gw, gh, gr, gc)
                rlo_ni, rhi_ni, clo_ni, chi_ni, nc_i = _macro_contrib(
                    new_xi, new_yi, wi, hi_m, gw, gh, gr, gc)
                rlo_oj, rhi_oj, clo_oj, chi_oj, oc_j = _macro_contrib(
                    old_xj, old_yj, wj, hj_m, gw, gh, gr, gc)
                rlo_nj, rhi_nj, clo_nj, chi_nj, nc_j = _macro_contrib(
                    new_xj, new_yj, wj, hj_m, gw, gh, gr, gc)
                grid[rlo_oi:rhi_oi, clo_oi:chi_oi] -= oc_i
                grid[rlo_ni:rhi_ni, clo_ni:chi_ni] += nc_i
                grid[rlo_oj:rhi_oj, clo_oj:chi_oj] -= oc_j
                grid[rlo_nj:rhi_nj, clo_nj:chi_nj] += nc_j
                new_den = _top_density(grid, n_top)
                new_score = new_wl + 0.5 * new_den

                if new_score < cur_score - 1e-9:
                    cur_score = new_score
                    cur_wl = new_wl
                    cur_den = new_den
                    for k, bb in new_bb.items():
                        net_bboxes[k] = bb
                    any_pass_improved = True
                else:
                    pos[i, 0] = old_xi; pos[i, 1] = old_yi
                    pos[j, 0] = old_xj; pos[j, 1] = old_yj
                    # Revert grid
                    grid[rlo_ni:rhi_ni, clo_ni:chi_ni] -= nc_i
                    grid[rlo_oi:rhi_oi, clo_oi:chi_oi] += oc_i
                    grid[rlo_nj:rhi_nj, clo_nj:chi_nj] -= nc_j
                    grid[rlo_oj:rhi_oj, clo_oj:chi_oj] += oc_j

    return pos


# ── Coordinate Descent ───────────────────────────────────────────────────────

def _coordinate_descent(pos, wl_data, movable_idx, n_hard, sizes_np, sx, sy, hw, hh, cw, ch,
                         time_budget_secs, max_sweeps=20):
    """
    Coordinate descent: for each movable macro find the WL-optimal (x, y) via 1D event scan
    (weighted median of connected-net extents), then move there if no overlap results.
    Multiple sweeps until convergence or time budget exhausted.
    """
    pos = pos.copy()
    movable_list = list(movable_idx)

    # Use float32 half-sizes for overlap checks — matches SA's ok() and _count_overlaps.
    sizes_f32 = sizes_np.astype(np.float32)
    sx32 = (sizes_f32[:, 0:1] + sizes_f32[:, 0:1].T) / 2.0
    sy32 = (sizes_f32[:, 1:2] + sizes_f32[:, 1:2].T) / 2.0

    hard_bidxs = wl_data['hard_bidxs']
    hard_ox = wl_data['hard_ox']
    hard_oy = wl_data['hard_oy']
    unique_nets_arr = wl_data['unique_nets']
    starts = wl_data['starts']
    fixed_min_x = wl_data['fixed_min_x']
    fixed_max_x = wl_data['fixed_max_x']
    fixed_min_y = wl_data['fixed_min_y']
    fixed_max_y = wl_data['fixed_max_y']
    net_weights = wl_data['net_weights']
    macro_to_nets = wl_data['macro_to_nets']
    n_un = len(unique_nets_arr)
    n_hard_entries = len(hard_bidxs)

    # Map original net index → position k in unique_nets_arr
    net_idx_to_k = {int(unique_nets_arr[k]): k for k in range(n_un)}

    t0 = time.time()

    for _sweep in range(max_sweeps):
        if time.time() - t0 >= time_budget_secs:
            break
        any_moved = False

        for bidx in movable_list:
            if time.time() - t0 >= time_budget_secs:
                break

            net_list = macro_to_nets[bidx]
            if not net_list:
                continue

            events_x = []
            events_y = []
            total_wt_x = 0.0
            total_wt_y = 0.0

            for net_idx in net_list:
                k = net_idx_to_k.get(net_idx, -1)
                if k == -1:
                    continue
                nidx = int(unique_nets_arr[k])
                w = float(net_weights[nidx])

                lo = int(starts[k])
                hi_end = int(starts[k + 1]) if k + 1 < n_un else n_hard_entries

                other_min_x = float(fixed_min_x[nidx])
                other_max_x = float(fixed_max_x[nidx])
                other_min_y = float(fixed_min_y[nidx])
                other_max_y = float(fixed_max_y[nidx])
                self_ox = 0.0
                self_oy = 0.0

                for p in range(lo, hi_end):
                    mb = int(hard_bidxs[p])
                    ox_p = float(hard_ox[p])
                    oy_p = float(hard_oy[p])
                    if mb == bidx:
                        self_ox = ox_p
                        self_oy = oy_p
                    else:
                        px = pos[mb, 0] + ox_p
                        py = pos[mb, 1] + oy_p
                        if px < other_min_x:
                            other_min_x = px
                        if px > other_max_x:
                            other_max_x = px
                        if py < other_min_y:
                            other_min_y = py
                        if py > other_max_y:
                            other_max_y = py

                if other_min_x <= other_max_x:
                    total_wt_x += w
                    events_x.append((other_min_x - self_ox, w))
                    events_x.append((other_max_x - self_ox, w))
                if other_min_y <= other_max_y:
                    total_wt_y += w
                    events_y.append((other_min_y - self_oy, w))
                    events_y.append((other_max_y - self_oy, w))

            old_x, old_y = pos[bidx, 0], pos[bidx, 1]
            opt_x = old_x
            opt_y = old_y

            if events_x and total_wt_x > 0:
                events_x.sort()
                G = -total_wt_x
                for ev_x, ev_w in events_x:
                    G += ev_w
                    if G >= 0:
                        opt_x = float(np.clip(ev_x, hw[bidx], cw - hw[bidx]))
                        break

            if events_y and total_wt_y > 0:
                events_y.sort()
                G = -total_wt_y
                for ev_y, ev_w in events_y:
                    G += ev_w
                    if G >= 0:
                        opt_y = float(np.clip(ev_y, hh[bidx], ch - hh[bidx]))
                        break

            if abs(opt_x - old_x) < 1e-10 and abs(opt_y - old_y) < 1e-10:
                continue

            # Tentatively apply move and check macro overlaps (float32 sizes, same as SA)
            pos[bidx, 0] = opt_x
            pos[bidx, 1] = opt_y
            overlap = False
            for k2 in range(n_hard):
                if k2 == bidx:
                    continue
                if (abs(opt_x - pos[k2, 0]) < sx32[bidx, k2] and
                        abs(opt_y - pos[k2, 1]) < sy32[bidx, k2]):
                    overlap = True
                    break

            if overlap:
                pos[bidx, 0] = old_x
                pos[bidx, 1] = old_y
            else:
                any_moved = True

        if not any_moved:
            break

    return pos


def _cd_full_proxy(pos, wl_data, density_data, cong_data, movable_idx, n_hard,
                   sizes_np, hw, hh, cw, ch, time_budget_secs,
                   max_sweeps=3000, cong_w_calib=0.5, den_w_cd=0.5):
    """
    Full-proxy coordinate descent: per-macro candidate search minimizing
    WL + 0.5*density + cong_w*cong with incremental delta computation.

    Candidates per macro (up to 10):
      1. WL-optimal position (weighted median event scan)
      2-4. Intermediate steps at alpha=0.75, 0.5, 0.25 toward WL-optimal
      5-6. Partial moves: WL-optimal X only (opt_x, old_y) and Y only (old_x, opt_y)
      7. Center of the lowest-congestion cell in 5-cell-radius neighbourhood
      8. Center of the lowest-density cell in 5-cell-radius neighbourhood
      9. Global low-cong: best V column × best H row (argmin V_col, argmin H_row)
      10. Global low-density: argmin over entire density grid

    All density and congestion updates are incremental (O(nets × bandwidth) per eval).
    Accepts a move only if the full calibrated proxy strictly decreases.
    """
    pos = pos.copy()
    movable_list = list(movable_idx)

    gr = density_data['grid_rows'];  gc = density_data['grid_cols']
    gw = density_data['grid_cell_w']; gh = density_data['grid_cell_h']
    n_top = density_data['n_top']
    hard_sizes = density_data['hard_sizes']

    grid = _build_density_grid(pos, density_data, n_hard)
    V_col, H_row = _build_cong_grid(pos, cong_data)
    cur_den  = _top_density(grid, n_top)
    cur_cong = _cong_cost(V_col, H_row, cong_data)

    # Per-unique-net cong bboxes — maintained incrementally (same as SA).
    n_un_c = cong_data['n_un']
    net_bbox = np.zeros((n_un_c, 4), dtype=np.int32)
    for k in range(n_un_c):
        c0, c1, r0, r1 = _net_grid_bbox(k, pos, cong_data)
        net_bbox[k] = (c0, c1, r0, r1)

    # float32 half-sizes for overlap check (mirrors SA's ok()).
    sizes_f32 = sizes_np.astype(np.float32)
    sx32 = (sizes_f32[:, 0:1] + sizes_f32[:, 0:1].T) / 2.0
    sy32 = (sizes_f32[:, 1:2] + sizes_f32[:, 1:2].T) / 2.0

    # Vectorized CD overlap check: O(n_hard) numpy instead of O(n_hard) Python loop.
    def _cd_ok(bidx_v, cx, cy):
        bad = (np.abs(cx - pos[:, 0]) < sx32[bidx_v]) & (np.abs(cy - pos[:, 1]) < sy32[bidx_v])
        bad[bidx_v] = False
        return not bad.any()

    # WL shortcuts
    hard_bidxs   = wl_data['hard_bidxs']
    hard_ox      = wl_data['hard_ox']
    hard_oy      = wl_data['hard_oy']
    unique_nets_arr = wl_data['unique_nets']
    starts       = wl_data['starts']
    fixed_min_x  = wl_data['fixed_min_x']
    fixed_max_x  = wl_data['fixed_max_x']
    fixed_min_y  = wl_data['fixed_min_y']
    fixed_max_y  = wl_data['fixed_max_y']
    net_weights  = wl_data['net_weights']
    macro_to_nets = wl_data['macro_to_nets']
    wl_norm      = float(wl_data['norm'])
    n_un_w       = len(unique_nets_arr)
    n_hard_entries = len(hard_bidxs)
    net_idx_to_k = {int(unique_nets_arr[k]): k for k in range(n_un_w)}

    # Cong shortcuts
    macro_net_k     = cong_data['macro_net_k']
    net_hard_pins   = cong_data['net_hard_pins']
    net_fixed_c_min = cong_data['net_fixed_c_min']
    net_fixed_c_max = cong_data['net_fixed_c_max']
    net_fixed_r_min = cong_data['net_fixed_r_min']
    net_fixed_r_max = cong_data['net_fixed_r_max']
    v_cap           = cong_data['v_cap']
    h_cap           = cong_data['h_cap']
    cong_uniq       = cong_data['unique_nets']

    t0 = time.time()
    rng_cd = np.random.default_rng(42)
    movable_arr = np.array(movable_list, dtype=np.int32)

    for _sweep in range(max_sweeps):
        if time.time() - t0 >= time_budget_secs:
            break
        any_moved = False
        rng_cd.shuffle(movable_arr)

        for bidx in movable_arr:
            bidx = int(bidx)
            if time.time() - t0 >= time_budget_secs:
                break

            old_x, old_y = pos[bidx, 0], pos[bidx, 1]
            w_i = float(hard_sizes[bidx, 0])
            h_i = float(hard_sizes[bidx, 1])

            # ── Build WL "other-extent" for each net containing bidx ──────────
            net_list = macro_to_nets[bidx]
            wl_nets = []   # (w, soxw, soyw, A_min_x, A_max_x, A_min_y, A_max_y)
            total_wt_x = 0.0;  total_wt_y = 0.0
            events_x = [];     events_y   = []

            for net_idx_orig in net_list:
                k_wl = net_idx_to_k.get(net_idx_orig, -1)
                if k_wl == -1:
                    continue
                nidx = int(unique_nets_arr[k_wl])
                w    = float(net_weights[nidx])
                lo   = int(starts[k_wl])
                hi_e = int(starts[k_wl + 1]) if k_wl + 1 < n_un_w else n_hard_entries

                A_min_x = float(fixed_min_x[nidx])
                A_max_x = float(fixed_max_x[nidx])
                A_min_y = float(fixed_min_y[nidx])
                A_max_y = float(fixed_max_y[nidx])
                soxw = 0.0;  soyw = 0.0

                for p in range(lo, hi_e):
                    mb  = int(hard_bidxs[p])
                    oxp = float(hard_ox[p]);  oyp = float(hard_oy[p])
                    if mb == bidx:
                        soxw = oxp;  soyw = oyp
                    else:
                        px = pos[mb, 0] + oxp;  py = pos[mb, 1] + oyp
                        if px < A_min_x: A_min_x = px
                        if px > A_max_x: A_max_x = px
                        if py < A_min_y: A_min_y = py
                        if py > A_max_y: A_max_y = py

                if A_min_x <= A_max_x:
                    total_wt_x += w
                    events_x.append((A_min_x - soxw, w))
                    events_x.append((A_max_x - soxw, w))
                if A_min_y <= A_max_y:
                    total_wt_y += w
                    events_y.append((A_min_y - soyw, w))
                    events_y.append((A_max_y - soyw, w))

                wl_nets.append((w, soxw, soyw, A_min_x, A_max_x, A_min_y, A_max_y))

            # Old WL contribution of bidx's nets
            old_wl_i = 0.0
            for (w, soxw, soyw, Amnx, Amxx, Amny, Amxy) in wl_nets:
                px = old_x + soxw;  py = old_y + soyw
                old_wl_i += w * (max(px, Amxx) - min(px, Amnx) +
                                  max(py, Amxy) - min(py, Amny))

            # WL-optimal position via weighted median
            opt_x = old_x;  opt_y = old_y
            if events_x and total_wt_x > 0:
                events_x.sort()
                G = -total_wt_x
                for ev_x, ev_w in events_x:
                    G += ev_w
                    if G >= 0:
                        opt_x = float(np.clip(ev_x, hw[bidx], cw - hw[bidx]))
                        break
            if events_y and total_wt_y > 0:
                events_y.sort()
                G = -total_wt_y
                for ev_y, ev_w in events_y:
                    G += ev_w
                    if G >= 0:
                        opt_y = float(np.clip(ev_y, hh[bidx], ch - hh[bidx]))
                        break

            # ── Build cong "base-extent" (excluding bidx) for each net ────────
            cong_nets = macro_net_k[bidx]
            # (k, k_w, self_ox, self_oy, base_c_min, base_c_max, base_r_min, base_r_max)
            cong_base = []
            for k in cong_nets:
                k_w     = float(net_weights[cong_uniq[k]])
                bc_min  = int(net_fixed_c_min[k]); bc_max = int(net_fixed_c_max[k])
                br_min  = int(net_fixed_r_min[k]); br_max = int(net_fixed_r_max[k])
                soxc    = 0.0;  soyc = 0.0
                for (mb, ox, oy) in net_hard_pins[k]:
                    if mb == bidx:
                        soxc = ox;  soyc = oy
                    else:
                        c2 = max(0, min(gc - 1, int((pos[mb, 0] + ox) / gw)))
                        r2 = max(0, min(gr - 1, int((pos[mb, 1] + oy) / gh)))
                        if c2 < bc_min: bc_min = c2
                        if c2 > bc_max: bc_max = c2
                        if r2 < br_min: br_min = r2
                        if r2 > br_max: br_max = r2
                cong_base.append((k, k_w, soxc, soyc, bc_min, bc_max, br_min, br_max))

            # ── Lowest-congestion and lowest-density cells within radius 5 ────
            r_curr = max(0, min(gr - 1, int(old_y / gh)))
            c_curr = max(0, min(gc - 1, int(old_x / gw)))
            best_cong_v = V_col[c_curr] / v_cap + H_row[r_curr] / h_cap
            best_den_v  = float(grid[r_curr, c_curr])
            cong_cx = None;  cong_cy = None
            den_cx  = None;  den_cy  = None
            for dr in range(-5, 6):
                for dc in range(-5, 6):
                    rc = r_curr + dr;  cc = c_curr + dc
                    if 0 <= rc < gr and 0 <= cc < gc:
                        cv = V_col[cc] / v_cap + H_row[rc] / h_cap
                        if cv < best_cong_v - 1e-9:
                            best_cong_v = cv
                            cong_cx = float(np.clip(cc * gw + gw / 2, hw[bidx], cw - hw[bidx]))
                            cong_cy = float(np.clip(rc * gh + gh / 2, hh[bidx], ch - hh[bidx]))
                        dv = float(grid[rc, cc])
                        if dv < best_den_v - 1e-9:
                            best_den_v = dv
                            den_cx = float(np.clip(cc * gw + gw / 2, hw[bidx], cw - hw[bidx]))
                            den_cy = float(np.clip(rc * gh + gh / 2, hh[bidx], ch - hh[bidx]))

            # ── Global low-cong: best V column + best H row (escapes radius-5 search) ─
            glob_vc = int(np.argmin(V_col))
            glob_hr = int(np.argmin(H_row))
            glob_cong_cx = float(np.clip(glob_vc * gw + gw / 2, hw[bidx], cw - hw[bidx]))
            glob_cong_cy = float(np.clip(glob_hr * gh + gh / 2, hh[bidx], ch - hh[bidx]))
            # ── Global low-density: argmin over entire grid ─────────────────────
            flat_idx = int(np.argmin(grid))
            g_den_r, g_den_c = flat_idx // gc, flat_idx % gc
            glob_den_cx = float(np.clip(g_den_c * gw + gw / 2, hw[bidx], cw - hw[bidx]))
            glob_den_cy = float(np.clip(g_den_r * gh + gh / 2, hh[bidx], ch - hh[bidx]))

            # ── Candidate list ────────────────────────────────────────────────
            dx = opt_x - old_x;  dy = opt_y - old_y
            cands = []
            if abs(dx) > 1e-10 or abs(dy) > 1e-10:
                cands.append((opt_x, opt_y))
                for alpha in (0.75, 0.5, 0.25):
                    cands.append((old_x + alpha * dx, old_y + alpha * dy))
                # Partial moves: WL-optimal on one axis only (helps when full move is blocked)
                if abs(dx) > 1e-10:
                    cands.append((opt_x, old_y))
                if abs(dy) > 1e-10:
                    cands.append((old_x, opt_y))
            if cong_cx is not None:
                cands.append((cong_cx, cong_cy))
            if den_cx is not None and (den_cx, den_cy) not in cands:
                cands.append((den_cx, den_cy))
            # Add global candidates (not already in cands)
            if (glob_cong_cx, glob_cong_cy) not in cands:
                cands.append((glob_cong_cx, glob_cong_cy))
            if (glob_den_cx, glob_den_cy) not in cands:
                cands.append((glob_den_cx, glob_den_cy))
            if not cands:
                continue

            # ── Evaluate each candidate ───────────────────────────────────────
            best_delta_proxy = 0.0   # only accept strict improvements
            best_cand_xy     = None

            for (cx_raw, cy_raw) in cands:
                cx = float(np.clip(cx_raw, hw[bidx], cw - hw[bidx]))
                cy = float(np.clip(cy_raw, hh[bidx], ch - hh[bidx]))
                if abs(cx - old_x) < 1e-10 and abs(cy - old_y) < 1e-10:
                    continue

                # Vectorized overlap check
                if not _cd_ok(bidx, cx, cy):
                    continue

                # WL delta
                new_wl_i = 0.0
                for (w, soxw, soyw, Amnx, Amxx, Amny, Amxy) in wl_nets:
                    px = cx + soxw;  py = cy + soyw
                    new_wl_i += w * (max(px, Amxx) - min(px, Amnx) +
                                      max(py, Amxy) - min(py, Amny))
                delta_wl = (new_wl_i - old_wl_i) / wl_norm

                # Density delta (temp grid update → revert)
                rlo_o, rhi_o, clo_o, chi_o, oc = _macro_contrib(old_x, old_y, w_i, h_i, gw, gh, gr, gc)
                rlo_n, rhi_n, clo_n, chi_n, nc = _macro_contrib(cx,    cy,    w_i, h_i, gw, gh, gr, gc)
                grid[rlo_o:rhi_o, clo_o:chi_o] -= oc
                grid[rlo_n:rhi_n, clo_n:chi_n] += nc
                new_den   = _top_density(grid, n_top)
                delta_den = new_den - cur_den
                grid[rlo_n:rhi_n, clo_n:chi_n] -= nc
                grid[rlo_o:rhi_o, clo_o:chi_o] += oc

                # Cong delta (temp V_col/H_row update → revert)
                for (k, k_w, soxc, soyc, bc_min, bc_max, br_min, br_max) in cong_base:
                    oc_min, oc_max, or_min, or_max = net_bbox[k]
                    if oc_max > oc_min: V_col[oc_min:oc_max] -= k_w
                    if or_max > or_min: H_row[or_min:or_max] -= k_w
                    npc = max(0, min(gc - 1, int((cx + soxc) / gw)))
                    npr = max(0, min(gr - 1, int((cy + soyc) / gh)))
                    nc_min = min(bc_min, npc); nc_max = max(bc_max, npc)
                    nr_min = min(br_min, npr); nr_max = max(br_max, npr)
                    if nc_max > nc_min: V_col[nc_min:nc_max] += k_w
                    if nr_max > nr_min: H_row[nr_min:nr_max] += k_w

                new_cong   = _cong_cost(V_col, H_row, cong_data)
                delta_cong = new_cong - cur_cong

                # Revert cong
                for (k, k_w, soxc, soyc, bc_min, bc_max, br_min, br_max) in cong_base:
                    npc = max(0, min(gc - 1, int((cx + soxc) / gw)))
                    npr = max(0, min(gr - 1, int((cy + soyc) / gh)))
                    nc_min = min(bc_min, npc); nc_max = max(bc_max, npc)
                    nr_min = min(br_min, npr); nr_max = max(br_max, npr)
                    if nc_max > nc_min: V_col[nc_min:nc_max] -= k_w
                    if nr_max > nr_min: H_row[nr_min:nr_max] -= k_w
                    oc_min, oc_max, or_min, or_max = net_bbox[k]
                    if oc_max > oc_min: V_col[oc_min:oc_max] += k_w
                    if or_max > or_min: H_row[or_min:or_max] += k_w

                delta_proxy = delta_wl + den_w_cd * delta_den + cong_w_calib * delta_cong
                if delta_proxy < best_delta_proxy:
                    best_delta_proxy = delta_proxy
                    best_cand_xy     = (cx, cy)

            # ── Apply best move ───────────────────────────────────────────────
            if best_cand_xy is not None:
                cx, cy = best_cand_xy
                pos[bidx, 0] = cx;  pos[bidx, 1] = cy

                # Density
                rlo_o, rhi_o, clo_o, chi_o, oc = _macro_contrib(old_x, old_y, w_i, h_i, gw, gh, gr, gc)
                rlo_n, rhi_n, clo_n, chi_n, nc = _macro_contrib(cx,    cy,    w_i, h_i, gw, gh, gr, gc)
                grid[rlo_o:rhi_o, clo_o:chi_o] -= oc
                grid[rlo_n:rhi_n, clo_n:chi_n] += nc
                cur_den = _top_density(grid, n_top)

                # Cong
                for (k, k_w, soxc, soyc, bc_min, bc_max, br_min, br_max) in cong_base:
                    oc_min, oc_max, or_min, or_max = net_bbox[k]
                    if oc_max > oc_min: V_col[oc_min:oc_max] -= k_w
                    if or_max > or_min: H_row[or_min:or_max] -= k_w
                    npc = max(0, min(gc - 1, int((cx + soxc) / gw)))
                    npr = max(0, min(gr - 1, int((cy + soyc) / gh)))
                    nc_min = min(bc_min, npc); nc_max = max(bc_max, npc)
                    nr_min = min(br_min, npr); nr_max = max(br_max, npr)
                    if nc_max > nc_min: V_col[nc_min:nc_max] += k_w
                    if nr_max > nr_min: H_row[nr_min:nr_max] += k_w
                    net_bbox[k] = (nc_min, nc_max, nr_min, nr_max)
                cur_cong = _cong_cost(V_col, H_row, cong_data)
                any_moved = True

        if not any_moved:
            break

    return pos


# ── GPU Gradient Placement (DreamPlace-style) ────────────────────────────────

def _get_device():
    """Auto-select best available device: CUDA > MPS > CPU."""
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available — gradient placement disabled")
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def _build_gpu_tensors(wl_data, density_data, n_hard, device):
    """Convert pre-built CPU cost arrays into float32 GPU tensors."""
    def t32(a):  return torch.tensor(np.asarray(a, dtype=np.float32), device=device)
    def tlong(a): return torch.tensor(np.asarray(a, dtype=np.int64),  device=device)

    INF = 1e30
    fix_max_x = np.asarray(wl_data['fixed_max_x'], dtype=np.float32)
    fix_min_x = np.asarray(wl_data['fixed_min_x'], dtype=np.float32)
    fix_max_y = np.asarray(wl_data['fixed_max_y'], dtype=np.float32)
    fix_min_y = np.asarray(wl_data['fixed_min_y'], dtype=np.float32)
    # Replace ±INF sentinels with finite GPU-safe values
    fix_max_x = np.where(fix_max_x < -1e8, -INF, fix_max_x)
    fix_min_x = np.where(fix_min_x >  1e8,  INF, fix_min_x)
    fix_max_y = np.where(fix_max_y < -1e8, -INF, fix_max_y)
    fix_min_y = np.where(fix_min_y >  1e8,  INF, fix_min_y)

    gr = density_data['grid_rows']
    gc = density_data['grid_cols']
    gw = float(density_data['grid_cell_w'])
    gh = float(density_data['grid_cell_h'])

    out = {
        'n_wl_nets':     wl_data['n_wl_nets'],
        'hard_bidxs':    tlong(wl_data['hard_bidxs']),
        'hard_ox':       t32(wl_data['hard_ox']),
        'hard_oy':       t32(wl_data['hard_oy']),
        'hard_net':      tlong(wl_data['hard_net_sorted']),
        'fixed_max_x':   t32(fix_max_x),
        'fixed_min_x':   t32(fix_min_x),
        'fixed_max_y':   t32(fix_max_y),
        'fixed_min_y':   t32(fix_min_y),
        'net_weights':   t32(wl_data['net_weights']),
        'hard_sizes':    t32(density_data['hard_sizes']),
        'fixed_density': t32(density_data['fixed_density']),
        'gr': gr, 'gc': gc, 'gw': gw, 'gh': gh,
    }
    if PLACER_CONG_AWARE:
        cw = gc * gw
        ch = gr * gh
        out['net_demand'] = _build_net_demand_map(wl_data, gr, gc, gw, gh, cw, ch, device)
    return out


def _wl_loss_gpu(pos, gd, gamma):
    """
    Differentiable WL via stable log-sum-exp HPWL approximation.
    Handles movable pins (gradient-bearing) + fixed bounding boxes (constants).
    pos: [n_hard, 2] float32 on device.
    """
    n    = gd['n_wl_nets']
    bidx = gd['hard_bidxs']   # [n_pins] long
    ox   = gd['hard_ox']      # [n_pins] f32
    oy   = gd['hard_oy']      # [n_pins] f32
    nets = gd['hard_net']     # [n_pins] long
    dev  = pos.device

    px = pos[bidx, 0] + ox    # movable pin x   [n_pins]
    py = pos[bidx, 1] + oy    # movable pin y

    def _lse(pin_vals, fixed_bound):
        # stable: per-net max(movable, fixed) → shift → sum → log
        net_max = torch.full((n,), -1e30, device=dev)
        net_max.scatter_reduce_(0, nets, pin_vals, reduce='amax', include_self=True)
        net_max = torch.maximum(net_max, fixed_bound)
        # Movable contribution
        exp_mov = torch.exp((gamma * (pin_vals - net_max[nets])).clamp(-80.0, 0.0))
        net_sum = torch.zeros(n, device=dev)
        net_sum.scatter_add_(0, nets, exp_mov)
        # Fixed contribution (exp(0)=1 when fixed_bound==net_max, 0 when -1e30)
        shift_fix = (gamma * (fixed_bound - net_max)).clamp(-80.0, 0.0)
        has_fix = fixed_bound > -1e29
        net_sum = net_sum + torch.where(has_fix, torch.exp(shift_fix),
                                        torch.zeros_like(shift_fix))
        return net_max + torch.log(net_sum.clamp(min=1e-30)) / gamma

    wl_x = _lse(px,  gd['fixed_max_x']) + _lse(-px, -gd['fixed_min_x'])
    wl_y = _lse(py,  gd['fixed_max_y']) + _lse(-py, -gd['fixed_min_y'])
    return ((wl_x + wl_y) * gd['net_weights']).sum()


def _density_loss_gpu(pos, gd, target_util):
    """
    Bell-kernel density overflow: Σ relu(D(r,c) − target)².
    Density computed as efficient matmul: [gr,gc] = phi_y.T @ (phi_x * weight).
    pos: [n_hard, 2] float32 on device.
    """
    gr, gc   = gd['gr'], gd['gc']
    gw, gh   = gd['gw'], gd['gh']
    cw, ch   = gc * gw, gr * gh
    sizes    = gd['hard_sizes']   # [n, 2]
    dev      = pos.device

    cx = torch.linspace(gw / 2, cw - gw / 2, gc, device=dev)   # [gc]
    cy = torch.linspace(gh / 2, ch - gh / 2, gr, device=dev)   # [gr]

    ax = sizes[:, 0:1] / 2 + gw   # bell half-width  [n,1]
    ay = sizes[:, 1:2] / 2 + gh   # bell half-height [n,1]

    phi_x = (1.0 - (pos[:, 0:1] - cx).abs() / ax).clamp(0.0, 1.0)   # [n, gc]
    phi_y = (1.0 - (pos[:, 1:2] - cy).abs() / ay).clamp(0.0, 1.0)   # [n, gr]

    weight  = sizes[:, 0] * sizes[:, 1] / (gw * gh)            # [n]
    density = phi_y.T @ (phi_x * weight.unsqueeze(1))           # [gr, gc]
    density = density + gd['fixed_density']

    return torch.relu(density - target_util).pow(2).sum()


def _build_net_demand_map(wl_data, gr, gc, gw, gh, cw, ch, device):
    """
    Precompute a per-bin 'net demand' map [gr, gc] in [0, 1].

    For each WL net, smear unit weight uniformly over the bins covered by the
    net's fixed-pin bounding box. Bins under many nets get higher demand,
    indicating they're more likely to be congested.

    Computed ONCE at gradient setup — does not depend on macro positions.
    Used to bias the density target: target_per_bin = target * (1 - β * demand[bin]).
    """
    fmin_x = np.asarray(wl_data['fixed_min_x'], dtype=np.float32)
    fmax_x = np.asarray(wl_data['fixed_max_x'], dtype=np.float32)
    fmin_y = np.asarray(wl_data['fixed_min_y'], dtype=np.float32)
    fmax_y = np.asarray(wl_data['fixed_max_y'], dtype=np.float32)
    n_nets = len(fmin_x)

    demand = np.zeros((gr, gc), dtype=np.float32)
    for k in range(n_nets):
        x_lo = max(0.0, fmin_x[k] if fmin_x[k] > -1e8 else 0.0)
        x_hi = min(cw,  fmax_x[k] if fmax_x[k] <  1e8 else cw)
        y_lo = max(0.0, fmin_y[k] if fmin_y[k] > -1e8 else 0.0)
        y_hi = min(ch,  fmax_y[k] if fmax_y[k] <  1e8 else ch)
        if x_hi <= x_lo or y_hi <= y_lo:
            continue
        bx_lo = max(0, int(x_lo / gw))
        bx_hi = min(gc, int(x_hi / gw) + 1)
        by_lo = max(0, int(y_lo / gh))
        by_hi = min(gr, int(y_hi / gh) + 1)
        demand[by_lo:by_hi, bx_lo:bx_hi] += 1.0

    m = demand.max()
    if m > 0:
        demand /= m
    return torch.tensor(demand, device=device)


def _electrostatic_density_loss_gpu(pos, gd, target_util):
    """
    ePlace/RePlAce/DreamPlace-style electrostatic density energy via FFT Poisson solve.

    Treats (density − target) as a 2D charge field ρ on the canvas grid, then
    solves Poisson's equation -∇²ψ = ρ in Fourier space (ψ̂ = ρ̂ / |k|²).
    Energy E = ½ Σ ρ·ψ·dA. Gradient w.r.t. macro positions IS the
    electrostatic force — globally aware: overcrowded regions repel,
    underfilled regions attract. The local relu² overflow penalty in
    `_density_loss_gpu` only feels nearby crowding; this feels the whole
    canvas. This is THE single biggest algorithmic upgrade vs the naive
    overflow loss and is what separates "SA-territory" placers (~1.45)
    from "gradient-territory" placers (~1.30) on the leaderboard.

    pos: [n_hard, 2] float32 on device.
    """
    gr, gc   = gd['gr'], gd['gc']
    gw, gh   = gd['gw'], gd['gh']
    cw, ch   = gc * gw, gr * gh
    sizes    = gd['hard_sizes']   # [n, 2]
    dev      = pos.device
    dtype    = pos.dtype

    # 1. Bell-kernel density deposition (same as _density_loss_gpu).
    cx = torch.linspace(gw / 2, cw - gw / 2, gc, device=dev, dtype=dtype)
    cy = torch.linspace(gh / 2, ch - gh / 2, gr, device=dev, dtype=dtype)
    ax = sizes[:, 0:1] / 2 + gw
    ay = sizes[:, 1:2] / 2 + gh
    phi_x = (1.0 - (pos[:, 0:1] - cx).abs() / ax).clamp(0.0, 1.0)
    phi_y = (1.0 - (pos[:, 1:2] - cy).abs() / ay).clamp(0.0, 1.0)
    weight  = sizes[:, 0] * sizes[:, 1] / (gw * gh)
    density = phi_y.T @ (phi_x * weight.unsqueeze(1)) + gd['fixed_density']

    # 2. Charge field. Optionally use a per-bin congestion-aware target.
    if 'net_demand' in gd and PLACER_CONG_AWARE:
        # target_per_bin = target_util * (1 - β * net_demand[bin]) * target_mult
        # Bins with high demand have LOWER target, so even moderate density there
        # registers as 'overflow' → macros pushed away.
        beta = float(PLACER_CONG_WEIGHT)
        target_grid = target_util * (1.0 - beta * gd['net_demand']) * PLACER_DENSITY_TARGET_MULT
        rho = density - target_grid
    else:
        rho = density - target_util * PLACER_DENSITY_TARGET_MULT
    rho = rho - rho.mean()

    # 3. FFT Poisson solve. Wavenumbers (angular, rad/unit):
    kx = 2.0 * math.pi * torch.fft.fftfreq(gc, d=gw).to(device=dev, dtype=dtype)
    ky = 2.0 * math.pi * torch.fft.fftfreq(gr, d=gh).to(device=dev, dtype=dtype)
    KY, KX = torch.meshgrid(ky, kx, indexing='ij')   # [gr, gc]
    K2 = KX * KX + KY * KY
    # K2[0,0] is exactly 0; replace with 1 to avoid div-by-zero. ρ̂[0,0]=0
    # already (zero-mean rho), so ψ̂[0,0] = 0/1 = 0 — correct DC handling.
    K2_safe = torch.where(K2 > 0, K2, torch.ones_like(K2))

    rho_fft = torch.fft.fft2(rho)
    psi_fft = rho_fft / K2_safe
    psi     = torch.fft.ifft2(psi_fft).real

    # 4. Energy = ½ ∫ ρ·ψ dA  (positive semi-definite, ≡ ½ ∫|∇ψ|² dA).
    return 0.5 * (rho * psi).sum() * (gw * gh)


def _gpu_spread_legalize(pos_np, movable_np, sizes_np, hw, hh, cw, ch,
                          n_hard, device, max_iters=600):
    """
    GPU force-spread legalization: vectorised O(n²) pairwise push, very fast on GPU.
    Returns np.ndarray [n_hard, 2].
    """
    pos   = torch.tensor(pos_np,    dtype=torch.float32, device=device)
    sizes = torch.tensor(sizes_np,  dtype=torch.float32, device=device)
    hw_t  = torch.tensor(hw,        dtype=torch.float32, device=device)
    hh_t  = torch.tensor(hh,        dtype=torch.float32, device=device)
    mov   = torch.tensor(movable_np, dtype=torch.bool,  device=device)

    sx = (sizes[:, 0:1] + sizes[:, 0]) / 2   # [n,n] pairwise sum of half-widths
    sy = (sizes[:, 1:2] + sizes[:, 1]) / 2

    for _ in range(max_iters):
        dx = pos[:, 0:1] - pos[:, 0]    # [n,n]
        dy = pos[:, 1:2] - pos[:, 1]

        ox = sx - dx.abs()
        oy = sy - dy.abs()
        overlap = (ox > 0) & (oy > 0)
        overlap.fill_diagonal_(False)

        if not overlap.any():
            break

        push_x = (ox <= oy) & overlap
        push_y = (~push_x)  & overlap

        fx = (ox / 2 + 1e-7) * dx.sign() * push_x.float()
        fy = (oy / 2 + 1e-7) * dy.sign() * push_y.float()

        fx_sum = fx.sum(dim=1)
        fy_sum = fy.sum(dim=1)

        new_x = (pos[:, 0] + fx_sum).clamp(hw_t, cw - hw_t)
        new_y = (pos[:, 1] + fy_sum).clamp(hh_t, ch - hh_t)
        pos[:, 0] = torch.where(mov, new_x, pos[:, 0])
        pos[:, 1] = torch.where(mov, new_y, pos[:, 1])

    return pos.cpu().numpy().astype(np.float64)


def _gradient_placement(pos_init_np, wl_data, density_data, movable_np,
                          n_hard, cw, ch, time_budget_secs, seed=42):
    """
    DreamPlace-style gradient placement with GPU acceleration.

    Phase a (~8% of budget): WL-only warmup — clusters macros by connectivity.
    Phase b (~92% of budget): WL + density co-optimisation — spreads macros while
      preserving topology.  Lambda (density weight) grows 1e-4 → 500 exponentially.

    Returns np.ndarray [n_hard, 2] float64 — positions before legalization.
    """
    device = _get_device()
    if device.type in ('cpu', 'mps') and not PLACER_FORCE_GRAD:
        raise RuntimeError(f"{device.type.upper()} gradient not cost-effective (<20K iters/360s) — use SA fallback")
    torch.manual_seed(seed)

    gd      = _build_gpu_tensors(wl_data, density_data, n_hard, device)
    sizes_t = gd['hard_sizes']   # [n, 2] f32

    # Target utilisation for density penalty (hard macros only)
    hard_util = float((sizes_t[:, 0] * sizes_t[:, 1]).sum().item()) / (cw * ch)

    mov_t  = torch.tensor(movable_np, dtype=torch.bool, device=device)
    hw_t   = sizes_t[:, 0] / 2
    hh_t   = sizes_t[:, 1] / 2
    lo     = torch.stack([hw_t, hh_t], dim=1)      # canvas lower bounds
    hi     = torch.stack([cw - hw_t, ch - hh_t], dim=1)  # canvas upper bounds
    pos0   = torch.tensor(pos_init_np, dtype=torch.float32, device=device)

    pos = pos0.clone().detach().requires_grad_(True)
    gamma = PLACER_GAMMA if PLACER_GAMMA > 0 else 2.0
    # DreamPlace uses lr≈0.01; scale slightly with chip size but keep in [0.01, 0.05]
    default_lr = max(0.01, min(0.05, (cw + ch) * 6e-4))
    lr    = PLACER_LR if PLACER_LR > 0 else default_lr
    opt   = torch.optim.Adam([pos], lr=lr, betas=(0.9, 0.999))

    t0            = time.time()
    warmup_budget = time_budget_secs * 0.08
    best_wl       = float('inf')
    best_np       = pos_init_np.copy()

    # ── Phase a: WL warmup ──────────────────────────────────────────────────
    while time.time() - t0 < warmup_budget:
        opt.zero_grad()
        wl = _wl_loss_gpu(pos, gd, gamma)
        wl.backward()
        torch.nn.utils.clip_grad_norm_([pos], 1.0)
        opt.step()
        with torch.no_grad():
            pos.data.clamp_(lo, hi)
            pos.data[~mov_t] = pos0[~mov_t]
        wl_v = wl.item()
        if wl_v < best_wl:
            best_wl = wl_v
            best_np = pos.detach().cpu().numpy().copy()

    # Start Phase b from best warmup position (avoids oscillation at end of warmup)
    with torch.no_grad():
        pos.data = torch.tensor(best_np, dtype=torch.float32, device=device)
    opt = torch.optim.Adam([pos], lr=lr, betas=(0.9, 0.999))  # fresh momentum

    # Estimate GPU throughput for lambda schedule
    steps_warmup  = max(1, round((time.time() - t0) /
                                  max(warmup_budget, 1e-6) * warmup_budget))
    sps            = max(10.0, steps_warmup / max(time.time() - t0, 1e-6))
    main_budget    = time_budget_secs - (time.time() - t0)
    total_main     = max(1, int(sps * main_budget))

    lambda_d       = 1e-4
    lambda_d_max   = PLACER_LAMBDA_MAX
    growth_steps   = max(1, int(total_main * 0.75))
    lambda_growth  = (lambda_d_max / lambda_d) ** (1.0 / growth_steps)

    # Density choice: electrostatic (FFT Poisson energy, global field) vs
    # legacy relu² overflow (local). Electrostatic ≈ DreamPlace, recommended.
    density_fn = (_electrostatic_density_loss_gpu if PLACER_ELECTROSTATIC
                  else _density_loss_gpu)

    # Best-position tracking: λ ramps up during Phase b, so the final position
    # can be worse than an intermediate one. Track best by smooth WL+0.5·den.
    best_proxy = float('inf')
    best_np    = pos.detach().cpu().numpy().copy()
    check_every = max(1, total_main // 100)  # ~100 checkpoints

    # ── Phase b: WL + density ───────────────────────────────────────────────
    t1 = time.time()
    step = 0
    while time.time() - t1 < main_budget:
        opt.zero_grad()
        wl  = _wl_loss_gpu(pos, gd, gamma)
        den = density_fn(pos, gd, hard_util)
        total_loss = wl + lambda_d * den

        if PLACER_ADAPTIVE_LAMBDA:
            # DreamPlace-style: balance gradient-norm contributions of WL and density.
            # Compute the two gradients via autograd.grad, then renormalize
            # lambda_d toward the target ratio (||grad_WL||/||grad_den|| ≈ 1).
            wl_grad = torch.autograd.grad(wl, pos, retain_graph=True, create_graph=False)[0]
            den_grad = torch.autograd.grad(den, pos, retain_graph=False, create_graph=False)[0]
            wl_norm = wl_grad.norm().item() + 1e-12
            den_norm = den_grad.norm().item() + 1e-12
            target_lambda = wl_norm / den_norm
            # Smooth toward target lambda, but respect the growth schedule as upper bound.
            lambda_d = min(lambda_d_max, 0.95 * lambda_d + 0.05 * target_lambda)
            # Set pos.grad directly. Calling .backward() on a freshly-constructed
            # (wl + lambda_d * den) here would crash with "Trying to backward
            # through the graph a second time" because both wl's and den's
            # graphs were already consumed by the autograd.grad calls above.
            pos.grad = (wl_grad + lambda_d * den_grad).detach()
        else:
            total_loss.backward()

        torch.nn.utils.clip_grad_norm_([pos], 1.0)
        opt.step()
        with torch.no_grad():
            pos.data.clamp_(lo, hi)
            pos.data[~mov_t] = pos0[~mov_t]

        if PLACER_BEST_TRACK and (step % check_every == 0):
            # Use smooth proxy (WL + 0.5·density) — true proxy is too expensive here.
            smooth_proxy = wl.item() + 0.5 * den.item()
            if smooth_proxy < best_proxy:
                best_proxy = smooth_proxy
                best_np = pos.detach().cpu().numpy().copy()

        if not PLACER_ADAPTIVE_LAMBDA:
            lambda_d = min(lambda_d * lambda_growth, lambda_d_max)
        step += 1

    if PLACER_BEST_TRACK:
        result = best_np.astype(np.float64)
    else:
        result = pos.detach().cpu().numpy().astype(np.float64)

    if not np.isfinite(result).all():
        raise ValueError("Gradient placement produced non-finite positions")
    return result


def _gradient_placement_multistart(pos_init_np, wl_data, density_data, movable_np,
                                    n_hard, cw, ch, time_budget_secs, seed=42,
                                    n_starts=None):
    """
    Run _gradient_placement N times with different seeds, return the one
    with the lowest smooth WL+0.5·density. Each run gets time_budget / N.

    With CUDA's ~1M iters/360s on the competition spec, each run still gets
    plenty of iterations. Different seeds → different Adam trajectories →
    different local minima. The best across runs typically improves over
    single-start by 0.02-0.05.
    """
    n_starts = n_starts if n_starts is not None else PLACER_MULTISTART
    if n_starts <= 1:
        return _gradient_placement(pos_init_np, wl_data, density_data, movable_np,
                                    n_hard, cw, ch, time_budget_secs, seed=seed)

    per_run = time_budget_secs / n_starts
    best_result = None
    best_score  = float('inf')

    # Evaluate smooth proxy on a result by computing WL + 0.5·density on device.
    device = _get_device()
    gd = _build_gpu_tensors(wl_data, density_data, n_hard, device)
    sizes_t = gd['hard_sizes']
    hard_util = float((sizes_t[:, 0] * sizes_t[:, 1]).sum().item()) / (cw * ch)
    density_fn = (_electrostatic_density_loss_gpu if PLACER_ELECTROSTATIC
                  else _density_loss_gpu)

    for i in range(n_starts):
        try:
            r = _gradient_placement(pos_init_np, wl_data, density_data, movable_np,
                                     n_hard, cw, ch, per_run, seed=seed + i * 7919)
        except Exception:
            continue
        with torch.no_grad():
            pos_t = torch.tensor(r, dtype=torch.float32, device=device)
            score = (_wl_loss_gpu(pos_t, gd, 2.0).item()
                     + 0.5 * density_fn(pos_t, gd, hard_util).item())
        if score < best_score:
            best_score, best_result = score, r

    if best_result is None:
        # All runs failed; let caller handle SA fallback.
        raise RuntimeError("All multistart gradient runs failed")
    return best_result


# ── Main placer ───────────────────────────────────────────────────────────────

class LNSPlacer:
    """SA + LNS + Pairwise Swap + Full-Proxy CD placer. (v14)
    Phase transitions use true evaluator proxy for acceptance decisions.
    CD optimizes true proxy (WL + 0.5*den + cong_w*cong) with true-proxy guardrail; runs ~200s.
    Runtime: ~40 min/benchmark (target).
    """

    def __init__(self, seed=42):
        self.seed = seed

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        random.seed(self.seed)
        np.random.seed(self.seed)

        n_hard = benchmark.num_hard_macros
        sizes_np = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
        cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
        hw, hh = sizes_np[:, 0] / 2, sizes_np[:, 1] / 2
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
        search0 = _minimal_fix(init_pos.copy(), movable, sizes_np, hw, hh, cw, ch, n_hard)
        leg0 = _minimal_fix(init_pos.copy(), movable, sizes_np, hw, hh, cw, ch, n_hard, gap=0.01)
        if _count_overlaps(leg0.astype(np.float32).astype(np.float64), n_hard, sizes_np) > 0:
            leg0 = _legalize(
                leg0, movable, sizes_np, hw, hh, cw, ch, n_hard, sx, sy, gap=0.05,
            )

        # Compute SA calibration from leg0 ONCE.
        leg0_wl = _wl_cost(leg0, wl_data)
        leg0_den = _top_density(_build_density_grid(leg0, density_data, n_hard), density_data['n_top'])
        leg0_cong = _cong_cost(*_build_cong_grid(leg0, cong_data), cong_data)
        cong_w_calib = 0.5 * (leg0_wl + 0.5 * leg0_den) / leg0_cong if leg0_cong > 1e-9 else 0.5

        # Competition hard cap is 3600s/bench; leave ~100s margin for setup/eval overhead.
        total_budget = PLACER_TOTAL_BUDGET if PLACER_TOTAL_BUDGET > 0 else 3500.0
        t0_place = time.time()

        # Phase 0: Gradient-based placement (DreamPlace-style).
        # Uses ~15% of the budget (≤400s) to find a much better starting point than SA.
        # Falls back to SA-from-leg0 if gradient raises, can't legalize, or is worse than leg0.
        grad_budget = (PLACER_GRAD_BUDGET if PLACER_GRAD_BUDGET > 0
                       else min(400.0, total_budget * 0.15))
        best_pos = leg0.copy()
        leg0_calib, _, _, _ = _calibrated_proxy(
            leg0, wl_data, density_data, cong_data, n_hard, cong_w_calib)
        leg0_true_proxy, leg0_true_wl, leg0_true_den, leg0_true_cong = _true_proxy_plc(
            leg0, plc, hard_plc_indices, n_hard)
        if leg0_true_proxy is None:
            leg0_true_proxy = float("inf")

        # Only use an invalid SA placement as an exploratory basin when the
        # official evaluator says the legalized init is genuinely hard.
        # The internal 1D congestion proxy can be high on benign cases like
        # ibm04, where invalid-SA exploration wastes time and hurts quality.
        explore_from_sa = (
            leg0_true_den is not None and leg0_true_cong is not None
            and (leg0_true_den >= 1.0 or leg0_true_cong >= 2.35)
        )

        def _guarded_sa_fallback():
            sa_start = search0 if explore_from_sa else leg0
            sa_pos = _sa(
                sa_start, wl_data, density_data, cong_data, movable, movable_idx,
                sizes_np, sx, sy, hw, hh, cw, ch, n_hard, neighbors,
                n_iters=max(200_000, n_mov * 300), seed=self.seed,
                cong_w_override=cong_w_calib,
            )[0]
            sa_true, _, _, _ = _true_proxy_plc(sa_pos, plc, hard_plc_indices, n_hard)
            sa_ov = _count_overlaps(sa_pos.astype(np.float32).astype(np.float64), n_hard, sizes_np)
            if explore_from_sa:
                if sa_true is None:
                    print("[Phase0] SA exploratory start accepted; true proxy unavailable")
                else:
                    print(f"[Phase0] SA exploratory start accepted: proxy={sa_true:.4f} ov={sa_ov} leg0={leg0_true_proxy:.4f}")
                return sa_pos
            if sa_true is not None and sa_ov == 0 and sa_true < leg0_true_proxy:
                print(f"[Phase0] SA fallback accepted: {sa_true:.4f} < leg0={leg0_true_proxy:.4f}")
                return sa_pos
            if sa_true is None:
                print("[Phase0] SA fallback rejected: true proxy unavailable")
            else:
                print(f"[Phase0] SA fallback rejected: proxy={sa_true:.4f} ov={sa_ov} leg0={leg0_true_proxy:.4f}")
            return leg0.copy()
        try:
            grad_raw = _gradient_placement_multistart(
                leg0, wl_data, density_data, movable, n_hard,
                cw, ch, grad_budget, seed=self.seed,
            )
            # GPU force-spread legalization
            device_grad = _get_device()
            grad_legal = _gpu_spread_legalize(
                grad_raw, movable, sizes_np, hw, hh, cw, ch, n_hard, device_grad,
            )
            # Final overlap check — minimal_fix first (preserves WL), then full legalize
            if _count_overlaps(grad_legal, n_hard, sizes_np) > 0:
                grad_legal = _minimal_fix(
                    grad_legal, movable, sizes_np, hw, hh, cw, ch, n_hard,
                )
            if _count_overlaps(grad_legal, n_hard, sizes_np) > 0:
                grad_legal = _legalize(
                    grad_legal, movable, sizes_np, hw, hh, cw, ch, n_hard, sx, sy, gap=0.002,
                )
            if _count_overlaps(grad_legal, n_hard, sizes_np) > 0:
                # Still overlapping after all fixups — SA fallback
                best_pos = _guarded_sa_fallback()
            else:
                # Proxy guardrail: only accept gradient if it beats leg0 calibrated proxy
                grad_calib, _, _, _ = _calibrated_proxy(
                    grad_legal, wl_data, density_data, cong_data, n_hard, cong_w_calib)
                if grad_calib < leg0_calib or PLACER_FORCE_ACCEPT:
                    tag = "FORCE-accepted" if (grad_calib >= leg0_calib and PLACER_FORCE_ACCEPT) else "accepted"
                    print(f"[Phase0] gradient {tag}: grad_calib={grad_calib:.4f}  leg0={leg0_calib:.4f}")
                    best_pos = grad_legal
                else:
                    print(f"[Phase0] gradient rejected by guardrail: grad_calib={grad_calib:.4f} >= leg0={leg0_calib:.4f} — SA fallback")
                    best_pos = _guarded_sa_fallback()
        except Exception as _e:
            print(f"[Phase0] gradient exception ({_e}) — SA fallback")
            best_pos = _guarded_sa_fallback()

        # Optional pre-LNS coordinate-descent refinement (WL-optimal 1D moves).
        # Gradient gives a topology-correct but not WL-tight placement; CD tightens
        # WL with overlap-safe moves before LNS attempts global perturbations.
        # This is what vmallela does (their pipeline name is literally "Incremental CD+LNS").
        if PLACER_CD_PRE_LNS > 0:
            cd_budget = min(PLACER_CD_PRE_LNS,
                            max(10.0, total_budget - (time.time() - t0_place) - 1500.0))
            if cd_budget > 5.0:
                pre_cd_pos = _coordinate_descent(
                    best_pos, wl_data, movable_idx, n_hard, sizes_np, sx, sy,
                    hw, hh, cw, ch, time_budget_secs=cd_budget, max_sweeps=10,
                )
                # Accept by calibrated proxy guardrail (consistent with Phase 0 logic).
                cd_calib, _, _, _ = _calibrated_proxy(
                    pre_cd_pos, wl_data, density_data, cong_data, n_hard, cong_w_calib)
                bp_calib, _, _, _ = _calibrated_proxy(
                    best_pos, wl_data, density_data, cong_data, n_hard, cong_w_calib)
                if cd_calib < bp_calib:
                    print(f"[PreLNS-CD] accepted: {cd_calib:.4f} < {bp_calib:.4f}")
                    best_pos = pre_cd_pos
                else:
                    print(f"[PreLNS-CD] rejected: {cd_calib:.4f} >= {bp_calib:.4f}")

        # cong_w_calib stays at leg0-calibrated value — empirically optimal for ibm04/ibm06
        # (gradient-legalized placement has different WL/cong ratio than true optimum,
        #  recalibrating from it could increase cong_w and trade WL for cong: harmful).

        # True proxy after Phase 0 — baseline for all subsequent acceptance decisions.
        best_true_proxy, _wl_t, _den_t, _cong_t = _true_proxy_plc(best_pos, plc, hard_plc_indices, n_hard)
        if best_true_proxy is None:
            best_true_proxy = float('inf')
            _cong_t = None

        # Phase 2: Conflict-Driven LNS — calibrated cong weight aligns WL→cong naturally.
        # Empirically: WL-optimal positions also minimize cong for ibm04/ibm06, so using
        # cong_w=0.5 (explicit cong pressure) trades WL for 1D cong reduction without
        # improving 2D cong (topology-constrained), worsening true proxy.
        K = max(10, min(60, n_mov // 7))        # larger K for big benchmarks (ibm17: 13→19, ibm18: 17→24)
        mini_iters = max(4000, min(20000, 3_000_000 // n_mov))
        lns_budget = max(60.0, total_budget - (time.time() - t0_place) - 200.0)

        best_pos_lns, best_proxy_lns, _ = _lns(
            best_pos, wl_data, density_data, cong_data, movable, movable_idx,
            sizes_np, sx, sy, hw, hh, cw, ch, n_hard, neighbors,
            K=K, mini_iters=mini_iters,
            time_budget_secs=lns_budget, seed=self.seed + 1,
            cong_w_calib=cong_w_calib,
        )

        # Accept LNS result via true evaluator proxy — eliminates 1D≠2D cong guardrail issues.
        true_lns, _, _, _ = _true_proxy_plc(best_pos_lns, plc, hard_plc_indices, n_hard)
        if true_lns is not None and true_lns < best_true_proxy:
            best_pos = best_pos_lns
            best_true_proxy = true_lns
        elif true_lns is None:
            # Fallback if plc proxy unavailable: calibrated proxy
            proxy_cur_calib, _, _, _ = _calibrated_proxy(
                best_pos, wl_data, density_data, cong_data, n_hard, cong_w_calib)
            if best_proxy_lns < proxy_cur_calib:
                best_pos = best_pos_lns

        # Phase 3: Pairwise swap search — greedy O(n²), accept if WL+0.5*den improves.
        # Cap at 30s to preserve budget for full-proxy CD (phase 4).
        swap_budget = min(30.0, max(15.0, total_budget - (time.time() - t0_place) - 175.0))
        if swap_budget > 10.0:
            pos_after_swap = _pairwise_swap_search(
                best_pos, wl_data, density_data, cong_data, movable_idx,
                sizes_np, sx, sy, hw, hh, cw, ch, n_hard,
                time_budget_secs=swap_budget,
            )
            true_swap, _, _, _ = _true_proxy_plc(pos_after_swap, plc, hard_plc_indices, n_hard)
            if true_swap is not None and true_swap < best_true_proxy:
                best_pos = pos_after_swap
                best_true_proxy = true_swap
            elif true_swap is None:
                pre_wl = _wl_cost(best_pos, wl_data)
                pre_den = _top_density(
                    _build_density_grid(best_pos, density_data, n_hard), density_data['n_top'])
                post_wl = _wl_cost(pos_after_swap, wl_data)
                post_den = _top_density(
                    _build_density_grid(pos_after_swap, density_data, n_hard), density_data['n_top'])
                if post_wl + 0.5 * post_den < pre_wl + 0.5 * pre_den:
                    best_pos = pos_after_swap

        # Phase 4: WL+density CD — cong_w=0 keeps the objective decoupled from congestion
        # so CD can find strong WL+density optima; true-proxy guardrail rejects regressions.
        cd_budget = min(250.0, max(30.0, total_budget - (time.time() - t0_place) - 20.0))
        if cd_budget > 10.0:
            best_pos_cd = _cd_full_proxy(
                best_pos, wl_data, density_data, cong_data, movable_idx, n_hard,
                sizes_np, hw, hh, cw, ch, time_budget_secs=cd_budget,
                cong_w_calib=0.0, den_w_cd=0.5)
            true_cd, _, _, _ = _true_proxy_plc(best_pos_cd, plc, hard_plc_indices, n_hard)
            if true_cd is not None and true_cd < best_true_proxy:
                best_pos = best_pos_cd
                best_true_proxy = true_cd

        # Phase 5: Post-CD pairwise swap — CD creates new swap opportunities.
        swap2_budget = min(30.0, max(0.0, total_budget - (time.time() - t0_place) - 5.0))
        if swap2_budget > 5.0:
            pos_after_swap2 = _pairwise_swap_search(
                best_pos, wl_data, density_data, cong_data, movable_idx,
                sizes_np, sx, sy, hw, hh, cw, ch, n_hard,
                time_budget_secs=swap2_budget,
            )
            true_swap2, _, _, _ = _true_proxy_plc(pos_after_swap2, plc, hard_plc_indices, n_hard)
            if true_swap2 is not None and true_swap2 < best_true_proxy:
                best_pos = pos_after_swap2
                best_true_proxy = true_swap2

        # Phase 6: Bonus LNS — reclaims unused time when earlier phases converged early.
        # Large benchmarks (phases 3-5 fill the budget) get 0s here; small ones get ~140s.
        bonus_lns_budget = total_budget - (time.time() - t0_place) - 5.0
        if bonus_lns_budget > 30.0:
            bonus_pos, _, _ = _lns(
                best_pos, wl_data, density_data, cong_data, movable, movable_idx,
                sizes_np, sx, sy, hw, hh, cw, ch, n_hard, neighbors,
                K=K, mini_iters=mini_iters,
                time_budget_secs=bonus_lns_budget, seed=self.seed + 2,
                cong_w_calib=cong_w_calib,
            )
            true_bonus, _, _, _ = _true_proxy_plc(bonus_pos, plc, hard_plc_indices, n_hard)
            if true_bonus is not None and true_bonus < best_true_proxy:
                best_pos = bonus_pos
                best_true_proxy = true_bonus

        def _f32_overlaps(p):
            return _count_overlaps(p.astype(np.float32).astype(np.float64), n_hard, sizes_np)

        if _f32_overlaps(best_pos) > 0:
            best_fix = None
            best_fix_proxy = float("inf")
            for fn, kwargs in [
                (_minimal_fix, {"gap": 0.005}),
                (_minimal_fix, {"gap": 0.01}),
                (_legalize, {"gap": 0.002}),
                (_legalize, {"gap": 0.01}),
                (_legalize, {"gap": 0.05}),
            ]:
                if fn is _minimal_fix:
                    cand = fn(best_pos, movable, sizes_np, hw, hh, cw, ch, n_hard, **kwargs)
                else:
                    cand = fn(best_pos, movable, sizes_np, hw, hh, cw, ch, n_hard, sx, sy, **kwargs)
                if _f32_overlaps(cand) > 0:
                    continue
                cand_true, _, _, _ = _true_proxy_plc(cand, plc, hard_plc_indices, n_hard)
                if cand_true is not None and cand_true < best_fix_proxy:
                    best_fix = cand
                    best_fix_proxy = cand_true
            if best_fix is not None:
                best_pos = best_fix
                best_true_proxy = best_fix_proxy
            else:
                best_pos = _legalize(best_pos, movable, sizes_np, hw, hh,
                                     cw, ch, n_hard, sx, sy, gap=0.05)

        final_true, _, _, _ = _true_proxy_plc(best_pos, plc, hard_plc_indices, n_hard)
        if final_true is not None and leg0_true_proxy < final_true:
            print(f"[Final] reverting to leg0 guardrail: leg0={leg0_true_proxy:.4f} < final={final_true:.4f}")
            best_pos = leg0.copy()

        full_pos = benchmark.macro_positions.double()
        full_pos[:n_hard] = torch.tensor(best_pos, dtype=torch.float64)
        return full_pos
