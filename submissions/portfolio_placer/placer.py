"""
portfolio_placer — benchmark-routed best known local strategy.

Default route:
  * ibm08: SafeFastDream, because VM smoke at 600s gave 1.4878 vs
    TriSafe's 1.56-1.61.
  * everything else: TriSafeLNS, which is the most reliable guarded CPU/GPU
    portfolio from current sweeps.

Env:
  PORTFOLIO_FD_BENCHES="ibm08,ibm06" to override the FastDream route list.
  PORTFOLIO_TOTAL_BUDGET sets both PLACER_TOTAL_BUDGET and FD_TOTAL_BUDGET
    when those env vars are not already set.
"""

import importlib.util
import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SUBMISSIONS = _HERE.parent

_budget = os.environ.get("PORTFOLIO_TOTAL_BUDGET", "600")
os.environ.setdefault("PLACER_TOTAL_BUDGET", _budget)
os.environ.setdefault("FD_TOTAL_BUDGET", _budget)
os.environ.setdefault("FD_CONG_W_CALIB", "0.5")
os.environ.setdefault("FD_SADDLE_NOISE", "0.03")


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_tri = _load_module("_portfolio_tri", _SUBMISSIONS / "tri_safe_lns" / "placer.py")
_fd = _load_module("_portfolio_safe_fd", _SUBMISSIONS / "safe_fastdream" / "placer.py")


def _fd_benches():
    raw = os.environ.get("PORTFOLIO_FD_BENCHES", "ibm08")
    return {x.strip() for x in raw.split(",") if x.strip()}


class PortfolioPlacer:
    def __init__(self, seed=42):
        self.seed = seed

    def place(self, benchmark):
        if benchmark.name in _fd_benches():
            print(f"[PORTFOLIO] {benchmark.name}: using SafeFastDream")
            return _fd.SafeFastDreamPlacer(seed=self.seed).place(benchmark)
        print(f"[PORTFOLIO] {benchmark.name}: using TriSafeLNS")
        return _tri.TriSafeLNSPlacer(seed=self.seed).place(benchmark)


def get_placer(seed=42):
    return PortfolioPlacer(seed=seed)
