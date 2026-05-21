"""
multistart_placer — race portfolio_placer and xplace_placer, pick best.

The top scorers (Carrotato 0.97, Shoom 0.98) effectively do multi-start: they
try multiple methods and keep the best. We do the same with what we have:
  * Portfolio (TriSafeLNS for 16 benches, SafeFastDream for ibm08).
    Proven 1.4506 avg overnight.
  * Xplace-based (Xplace GP → CD/LNS/swap refinement). New, varying quality
    per benchmark — wins or loses vs portfolio depending on benchmark.

We run BOTH and return whichever has the lower true proxy. Floor is portfolio
(if Xplace's path doesn't help, we keep portfolio's result). Upside is wins
on benchmarks where Xplace's broader search escapes local minima portfolio
can't.

Env vars:
  MS_TOTAL_BUDGET       = total wall-clock seconds (default 1800)
  MS_PORTFOLIO_FRAC     = fraction of budget for portfolio (default 0.5)
  MS_SKIP_XPLACE        = "1" to skip Xplace path (debug/fallback)
  MS_SKIP_PORTFOLIO     = "1" to skip portfolio (testing only — leaves us with
                           Xplace-only output, possibly worse than leg0)

Sub-placer env vars are derived automatically (PORTFOLIO_TOTAL_BUDGET,
XP_TOTAL_BUDGET) before each sub-placer is loaded.
"""

import importlib.util
import os
import time
from pathlib import Path

import numpy as np
import torch

# Same DREAMPlace v4 numpy-alias patch as in xplace_placer / dreamplace_placer.
for _name, _typ in (("str", str), ("bool", bool), ("int", int),
                     ("float", float), ("object", object), ("long", int)):
    if not hasattr(np, _name):
        setattr(np, _name, _typ)

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost

_HERE = Path(__file__).resolve().parent
_SUBMISSIONS = _HERE.parent


def _ef(name, default):
    try: return float(os.environ.get(name, default))
    except Exception: return default
def _ei(name, default):
    try: return int(os.environ.get(name, default))
    except Exception: return default


MS_TOTAL_BUDGET    = _ef("MS_TOTAL_BUDGET", 1800.0)
MS_PORTFOLIO_FRAC  = _ef("MS_PORTFOLIO_FRAC", 0.5)
MS_SKIP_XPLACE     = _ei("MS_SKIP_XPLACE", 0)
MS_SKIP_PORTFOLIO  = _ei("MS_SKIP_PORTFOLIO", 0)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _true_proxy(positions: torch.Tensor, benchmark) -> float:
    """Score with the official proxy. Returns +inf on any error."""
    try:
        # _lns._load_plc gives us a PlacementCost; compute_proxy_cost expects
        # a torch positions tensor matching benchmark.macro_positions shape.
        # Lazy-import lns module to access _load_plc.
        if not hasattr(_true_proxy, "_lns"):
            _true_proxy._lns = _load_module(
                "_ms_lns_for_score", _SUBMISSIONS / "lns_placer" / "placer.py")
        plc = _true_proxy._lns._load_plc(benchmark.name)
        costs = compute_proxy_cost(positions, benchmark, plc)
        if int(costs.get("overlap_count", 0)) > 0:
            return float("inf")
        return float(costs["proxy_cost"])
    except Exception as exc:
        print(f"[MS] _true_proxy error: {exc}")
        return float("inf")


class MultiStartPlacer:
    def __init__(self, seed: int = 42):
        self.seed = seed

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        results = []  # list of (proxy, positions, label)
        t_global = time.time()

        portfolio_budget = MS_TOTAL_BUDGET * MS_PORTFOLIO_FRAC
        xplace_budget    = MS_TOTAL_BUDGET * (1.0 - MS_PORTFOLIO_FRAC)

        # ── Run 1: Portfolio (TriSafeLNS or SafeFastDream depending on bench) ─
        if not MS_SKIP_PORTFOLIO:
            t0 = time.time()
            try:
                os.environ["PORTFOLIO_TOTAL_BUDGET"] = str(int(portfolio_budget))
                os.environ["PLACER_TOTAL_BUDGET"]   = str(int(portfolio_budget))
                os.environ["FD_TOTAL_BUDGET"]       = str(int(portfolio_budget))
                pf_mod = _load_module(
                    "_ms_portfolio", _SUBMISSIONS / "portfolio_placer" / "placer.py")
                pos_pf = pf_mod.PortfolioPlacer(seed=self.seed).place(benchmark)
                pr_pf = _true_proxy(pos_pf, benchmark)
                print(f"[MS] Portfolio finished in {time.time()-t0:.0f}s: "
                      f"proxy={pr_pf:.4f}")
                results.append((pr_pf, pos_pf, "portfolio"))
            except Exception as exc:
                print(f"[MS] Portfolio exception: {exc}")
                import traceback; traceback.print_exc()

        # ── Run 2: Xplace + cong-aware refinement ─────────────────────────────
        if not MS_SKIP_XPLACE:
            t0 = time.time()
            try:
                os.environ["XP_TOTAL_BUDGET"] = str(int(xplace_budget))
                # Xplace's GP itself is fast on GPU; give the refinement more
                # of the budget by capping GP at ~120s.
                os.environ.setdefault("XP_GRAD_BUDGET", "120")
                xp_mod = _load_module(
                    "_ms_xplace", _SUBMISSIONS / "xplace_placer" / "placer.py")
                pos_xp = xp_mod.XplacePlacer(seed=self.seed).place(benchmark)
                pr_xp = _true_proxy(pos_xp, benchmark)
                print(f"[MS] Xplace finished in {time.time()-t0:.0f}s: "
                      f"proxy={pr_xp:.4f}")
                results.append((pr_xp, pos_xp, "xplace"))
            except Exception as exc:
                print(f"[MS] Xplace exception: {exc}")
                import traceback; traceback.print_exc()

        if not results:
            # Both paths failed — fall back to trivial legalize via lns_placer.
            print(f"[MS] BOTH paths failed; falling back to bare LNSPlacer")
            lns_mod = _load_module("_ms_lns_fallback",
                                    _SUBMISSIONS / "lns_placer" / "placer.py")
            return lns_mod.LNSPlacer(seed=self.seed).place(benchmark)

        # Pick the lowest-proxy result.
        results.sort(key=lambda r: r[0])
        best_proxy, best_pos, best_label = results[0]
        gap_str = ""
        if len(results) > 1:
            gap = results[1][0] - results[0][0]
            gap_str = f"  (Δ vs {results[1][2]} = {gap:+.4f})"
        print(f"[MS] WINNER: {best_label}  proxy={best_proxy:.4f}{gap_str}  "
              f"total_elapsed={time.time()-t_global:.0f}s")
        return best_pos


def get_placer(seed=42):
    return MultiStartPlacer(seed=seed)
