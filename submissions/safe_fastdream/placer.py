"""
safe_fastdream — FastDreamPlacer with the LegOnly regression guardrail.

Same pattern as safe_lns_placer/placer.py, but wraps fastdream instead.
Use this on the VM:

  1. Runs LegOnly (just legalize init.plc with gap=0.01)
  2. Runs the full FastDreamPlacer pipeline (gradient + LNS + CD + swap)
  3. Scores both with official compute_proxy_cost
  4. Returns whichever has the lower valid proxy

This is the right pattern when your inner placer is sometimes-better,
sometimes-worse than init.plc. Top leaderboard placers (DREAMPlace-based)
are reliably better than init.plc, so they don't need a guardrail; but
fastdream/lns_placer on CPU and even some GPU configurations can
regress on some benchmarks, so this catches that.
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
_FD_PATH = _HERE.parent / "fastdream" / "placer.py"

# Load lns_placer (for legalize helpers and _load_plc)
_spec_l = importlib.util.spec_from_file_location("_safe_fd_lns", str(_LNS_PATH))
_lns = importlib.util.module_from_spec(_spec_l)
_spec_l.loader.exec_module(_lns)

# Load fastdream
_spec_fd = importlib.util.spec_from_file_location("_safe_fd_fd", str(_FD_PATH))
_fd = importlib.util.module_from_spec(_spec_fd)
_spec_fd.loader.exec_module(_fd)


def _legalize_only(benchmark) -> torch.Tensor:
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
    result = benchmark.macro_positions.clone()
    result[:n_hard] = torch.from_numpy(legal).to(result.dtype)
    return result


class SafeFastDreamPlacer:
    def __init__(self, seed=42):
        self.seed = seed

    def place(self, benchmark) -> torch.Tensor:
        plc = _lns._load_plc(benchmark.name)
        if plc is None:
            return _fd.FastDreamPlacer(seed=self.seed).place(benchmark)

        # 1. LegOnly baseline
        t0 = time.time()
        leg_pos = _legalize_only(benchmark)
        leg_costs = compute_proxy_cost(leg_pos, benchmark, plc)
        leg_proxy = float(leg_costs["proxy_cost"])
        leg_overlaps = int(leg_costs["overlap_count"])
        print(f"[SAFE-FD] LegOnly proxy={leg_proxy:.4f} overlaps={leg_overlaps} ({time.time()-t0:.1f}s)")

        # 2. Full FastDream pipeline. Snapshot init positions in case it mutates.
        saved = benchmark.macro_positions.clone()
        try:
            fd_pos = _fd.FastDreamPlacer(seed=self.seed).place(benchmark)
        finally:
            benchmark.macro_positions = saved
        fd_costs = compute_proxy_cost(fd_pos, benchmark, plc)
        fd_proxy = float(fd_costs["proxy_cost"])
        fd_overlaps = int(fd_costs["overlap_count"])
        print(f"[SAFE-FD] FastDream proxy={fd_proxy:.4f} overlaps={fd_overlaps}")

        # 3. Pick best valid placement.
        leg_valid = leg_overlaps == 0
        fd_valid = fd_overlaps == 0
        if not fd_valid and leg_valid:
            print("[SAFE-FD] returning LegOnly (FastDream produced overlaps)")
            return leg_pos
        if fd_valid and not leg_valid:
            print("[SAFE-FD] returning FastDream (LegOnly has overlaps)")
            return fd_pos
        if fd_proxy < leg_proxy:
            print(f"[SAFE-FD] FastDream wins by {leg_proxy - fd_proxy:.4f}")
            return fd_pos
        else:
            print(f"[SAFE-FD] LegOnly wins by {fd_proxy - leg_proxy:.4f} — FastDream regressed")
            return leg_pos
