"""
tri_safe_lns — Best of three: LegOnly vs LegSwap vs full LNSPlacer.

Adds a LegSwap candidate (legalize + pairwise_swap, no SA) on top of
SafeLNS. LegSwap is a cheap WL-improving refinement that can't blow up
density/cong the way full LNS sometimes does. On benchmarks where LNS
regresses but LegOnly is also leaving WL on the table, LegSwap may win.

Knobs:
    PLACER_TOTAL_BUDGET — total seconds. LegOnly gets ~10s, LegSwap gets
                         budget/4, LNS gets the rest.
"""

import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from macro_place.objective import compute_proxy_cost

_HERE = Path(__file__).resolve().parent
_LNS_PATH = _HERE.parent / "lns_placer" / "placer.py"
_spec = importlib.util.spec_from_file_location("_tri_inner", str(_LNS_PATH))
_lns = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lns)


def _legalize_only(benchmark):
    n_hard = benchmark.num_hard_macros
    if n_hard == 0:
        return benchmark.macro_positions.clone()
    sizes_np = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
    cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
    hw = sizes_np[:, 0] / 2
    hh = sizes_np[:, 1] / 2
    movable = benchmark.get_movable_mask()[:n_hard].numpy()
    sx = (sizes_np[:, 0:1] + sizes_np[:, 0:1].T) / 2
    sy = (sizes_np[:, 1:2] + sizes_np[:, 1:2].T) / 2

    init_pos = benchmark.macro_positions[:n_hard].numpy().copy().astype(np.float64)
    legal = _lns._minimal_fix(init_pos, movable, sizes_np, hw, hh, cw, ch,
                               n_hard, gap=0.01)
    if _lns._count_overlaps(legal, n_hard, sizes_np) > 0:
        legal = _lns._legalize(legal, movable, sizes_np, hw, hh, cw, ch,
                                n_hard, sx, sy, gap=0.05)
    return legal


def _legswap(benchmark, plc, wl_data, density_data, cong_data, legal_np, budget):
    """LegOnly + pairwise_swap. Returns numpy positions."""
    n_hard = benchmark.num_hard_macros
    sizes_np = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
    cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
    hw = sizes_np[:, 0] / 2
    hh = sizes_np[:, 1] / 2
    movable_idx = np.where(benchmark.get_movable_mask()[:n_hard].numpy())[0]
    sx = (sizes_np[:, 0:1] + sizes_np[:, 0:1].T) / 2
    sy = (sizes_np[:, 1:2] + sizes_np[:, 1:2].T) / 2

    return _lns._pairwise_swap_search(
        legal_np, wl_data, density_data, cong_data, movable_idx,
        sizes_np, sx, sy, hw, hh, cw, ch, n_hard,
        time_budget_secs=budget,
    )


def _proxy(pos_np, benchmark, plc):
    n_hard = benchmark.num_hard_macros
    result = benchmark.macro_positions.clone()
    result[:n_hard] = torch.from_numpy(pos_np).to(result.dtype)
    costs = compute_proxy_cost(result, benchmark, plc)
    return result, float(costs["proxy_cost"]), int(costs["overlap_count"])


class TriSafeLNSPlacer:
    def __init__(self, seed=42):
        self.seed = seed

    def place(self, benchmark) -> torch.Tensor:
        plc = _lns._load_plc(benchmark.name)
        if plc is None:
            return _lns.LNSPlacer(seed=self.seed).place(benchmark)

        budget = float(os.environ.get("PLACER_TOTAL_BUDGET", "120"))
        wl_data, density_data, cong_data = _lns._build_cost_data(benchmark, plc)

        # 1. LegOnly
        t0 = time.time()
        legal = _legalize_only(benchmark)
        leg_t = time.time() - t0
        leg_result, leg_proxy, leg_ov = _proxy(legal, benchmark, plc)
        print(f"[TRI] LegOnly  proxy={leg_proxy:.4f} overlaps={leg_ov}  ({leg_t:.1f}s)")

        # 2. LegSwap (budget/4)
        swap_budget = max(15.0, budget * 0.25)
        t0 = time.time()
        swap_pos = _legswap(benchmark, plc, wl_data, density_data, cong_data,
                            legal, swap_budget)
        swap_t = time.time() - t0
        swap_result, swap_proxy, swap_ov = _proxy(swap_pos, benchmark, plc)
        print(f"[TRI] LegSwap  proxy={swap_proxy:.4f} overlaps={swap_ov}  ({swap_t:.1f}s)")

        # 3. Full LNS (remaining budget; honor PLACER_TOTAL_BUDGET internally)
        # We pass the *remaining* time as the LNS budget via env var override.
        remaining = max(30.0, budget - leg_t - swap_t - 5.0)
        old_budget = os.environ.get("PLACER_TOTAL_BUDGET")
        os.environ["PLACER_TOTAL_BUDGET"] = str(remaining)
        saved = benchmark.macro_positions.clone()
        try:
            lns_pos = _lns.LNSPlacer(seed=self.seed).place(benchmark)
        finally:
            benchmark.macro_positions = saved
            if old_budget is None:
                os.environ.pop("PLACER_TOTAL_BUDGET", None)
            else:
                os.environ["PLACER_TOTAL_BUDGET"] = old_budget
        lns_costs = compute_proxy_cost(lns_pos, benchmark, plc)
        lns_proxy = float(lns_costs["proxy_cost"])
        lns_ov = int(lns_costs["overlap_count"])
        print(f"[TRI] LNS       proxy={lns_proxy:.4f} overlaps={lns_ov}")

        # Pick best valid placement
        cands = [
            ("LegOnly", leg_result, leg_proxy, leg_ov),
            ("LegSwap", swap_result, swap_proxy, swap_ov),
            ("LNS",     lns_pos,    lns_proxy, lns_ov),
        ]
        valids = [(n, p, x, o) for n, p, x, o in cands if o == 0]
        if not valids:
            valids = cands  # all invalid; pick lowest proxy anyway
        valids.sort(key=lambda t: t[2])
        winner_name, winner_pos, winner_proxy, _ = valids[0]
        print(f"[TRI] WINNER = {winner_name} (proxy={winner_proxy:.4f})")
        return winner_pos
