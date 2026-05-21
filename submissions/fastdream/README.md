# FastDream — DREAMPlace-faithful gradient + heavy LNS

Target: close the 0.27 gap from MTK (1.28, pure DREAMPlace) to vmallela
(1.0109, DREAMPlace + heavy refinement). Built by getting the DREAMPlace
gradient details right and stacking the best refinement passes from
lns_placer.

## What's different from lns_placer

| Feature | lns_placer | FastDream |
|---------|-----------|-----------|
| HPWL smoothing γ | fixed 2.0 | **annealed 8.0 → 0.5** |
| Density grid resolution | fixed | **annealed coarse→fine** |
| Optimizer | Adam | **Nesterov SGD** |
| λ schedule | exponential growth | **dynamic from \|\|grad_WL\|\|/\|\|grad_den\|\|** |
| Initial placement | leg0 only | **3-start: leg0 + spiral + random** |
| Legalization | single strategy | **try 3, pick best by proxy** |
| Gradient acceptance | calibrated-proxy guardrail | **FORCE_ACCEPT by default** |
| Pre-LNS CD | optional | **always** |
| LNS K (macros per neighborhood) | n_mov / 7 | **n_mov / 6** (larger) |
| LNS budget fraction | ~50% | **65%** |

## Tuning knobs (env vars)

| Variable | Default | What it does |
|----------|---------|--------------|
| `FD_TOTAL_BUDGET` | 3500 | wall-clock seconds total |
| `FD_GRAD_BUDGET` | 600 | seconds for all gradient starts combined |
| `FD_LNS_FRAC` | 0.65 | fraction of remaining budget for LNS |
| `FD_N_STARTS` | 3 | number of gradient restarts |
| `FD_GAMMA_HI` | 8.0 | starting γ (smooth HPWL) |
| `FD_GAMMA_LO` | 0.5 | final γ (sharp HPWL) |
| `FD_LAMBDA_MIN` | 1e-4 | λ floor |
| `FD_LAMBDA_MAX` | 50 | λ ceiling |
| `FD_USE_NESTEROV` | 1 | 1=Nesterov SGD, 0=Adam |
| `FD_GRID_MULT_LO` | 0.5 | coarse grid factor at start of gradient |
| `FD_GRID_MULT_HI` | 1.0 | fine grid factor at end of gradient |
| `FD_FORCE_ACCEPT` | 1 | always use gradient (don't fall back to SA) |
| `FD_VERBOSE` | 1 | print diagnostics |

## Quick test

```bash
# Single benchmark, short budget for development
FD_TOTAL_BUDGET=500 FD_GRAD_BUDGET=200 \
  CUDA_VISIBLE_DEVICES=0 python3 -m macro_place.evaluate \
  submissions/fastdream/placer.py --benchmark ibm04

# Full benchmark, competition budget
CUDA_VISIBLE_DEVICES=0 python3 -m macro_place.evaluate \
  submissions/fastdream/placer.py --benchmark ibm10
```

## What to watch in the output

```
[FD] leg0_calib=0.6190  cong_w_calib=0.21  n_mov=134
  [FD-grad] sps≈3200, total_steps=640000
[FD-grad #0] grad#0_minimal_fix calib=0.6010  (leg0=0.6190, in 200s)   ← gradient helped
[FD-grad #1] grad#1_gpu_spread calib=0.5980   (leg0=0.6190, in 200s)
[FD-grad #2] grad#2_minimal_fix calib=0.7800  (leg0=0.6190, in 200s)   ← spiral start was bad, but ok — we pick best
[FD] best after gradient: grad#1_gpu_spread calib=0.5980
[FD-preCD] accepted: 0.5950 < 0.5980
[FD-LNS] K=22 mini_iters=29850 budget=2300s
[FD] DONE. final_true_proxy=1.2920 elapsed=3450s
```

Key signals:
- **`calib < leg0_calib`**: gradient is helping. We want this on as many benchmarks as possible.
- **multiple legalization strategies in output**: shows multi-leg is picking different winners per start.
- **final_true_proxy much better than v36's number**: the whole pipeline is winning.

## Hypothesized improvement

| Benchmark | v36 (your current) | FastDream target |
|-----------|-------------------|------------------|
| ibm04 (small) | 1.3289 | 1.28–1.31 |
| ibm10 (medium) | 1.3019 | 1.22–1.27 |
| ibm17 (large) | 1.7410 | 1.65–1.71 |
| **avg over 17** | **1.4538** | **1.30–1.40** |

If the targets hit, you land at rank 14-17 — significant jump from current ~25.

If the gradient *really* clicks (γ-annealing + Nesterov), the upper bound is more like 1.20-1.25 (top 8-10). The leader (1.01) is still a stretch — that requires real DREAMPlace + per-benchmark BO.
