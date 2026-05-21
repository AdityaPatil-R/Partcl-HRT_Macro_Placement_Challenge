# KeepDreaming — TriSafeLNS Portfolio

PartCL x HRT Macro Placement Challenge 2026 submission.

## Entry point

```
submissions/portfolio_placer/placer.py
```

This is the placer to evaluate. The other directories under `submissions/`
(`fastdream/`, `lns_placer/`, `safe_fastdream/`, `tri_safe_lns/`) are
**imported by** `portfolio_placer` and are part of the same submission tree.
The directories `dreamplace_placer/`, `xplace_placer/`, and `multistart_placer/`
are alternative pipelines kept in-repo for reference; they are **not** the
submitted entry point.

## How judges run it

The Dockerfile's default `CMD` already points at the right path, so the
canonical invocation is just:

```bash
docker build -t partcl-keepdreaming .

docker run --rm --gpus all --network none \
  -v $PWD/external:/work/external \
  partcl-keepdreaming
# defaults to: submissions/portfolio_placer/placer.py --all
```

For a single benchmark:

```bash
docker run --rm --gpus all --network none \
  -v $PWD/external:/work/external \
  partcl-keepdreaming \
  submissions/portfolio_placer/placer.py --benchmark ibm04
```

## Method

Per-benchmark routing between two CPU placers:

* **TriSafeLNS** (16 of 17 benchmarks). Three guarded passes:
  `LegOnly` (legalize from init), `LegSwap` (legalize + pairwise swap),
  `LNS` (conflict-driven Large Neighborhood Search with simulated-annealing
  outer acceptance). Pick the best by true proxy. Every accept/reject gate
  uses the official `compute_proxy_cost` — no calibrated drift.

* **SafeFastDream** (ibm08 only). Pure-PyTorch electrostatic gradient +
  coordinate descent + LNS. Specializes for ibm08's congestion profile where
  TriSafeLNS plateaus higher.

## Reported results (self-reported)

| Metric | Value |
|--------|-------|
| Avg proxy across 17 IBM benchmarks | **1.4506** |
| Avg runtime | ~48 min/bench (~13.5h total) |
| Overlaps | 0 on all 17 |

## Build notes

The image bundles CUDA 11.8, PyTorch 2.1.2, DREAMPlace 4.1.0, and Xplace 3.0
because the alternative pipelines (`xplace_placer/`, `dreamplace_placer/`) use
them. The submitted `portfolio_placer` itself is pure CPU — it doesn't call
the gradient placers. The image size is therefore larger than it strictly
needs to be for the submission, but it makes everything reproducible from a
single Dockerfile.

CMake quirks handled in the Dockerfile:
* DREAMPlace's `find_package(CUDA)` only works under cmake < 3.27, so we
  invoke `/usr/bin/cmake` (apt's 3.22) explicitly.
* Both DREAMPlace and Xplace patch
  `torch.cuda.is_available()` → `torch.backends.cuda.is_built()` so CUDA
  detection passes during `docker build` (no GPU at build time).

`macro_place/_plc.py` is patched to force-load the Python `plc_client_os.py`
from the TILOS MacroPlacement submodule rather than the prebuilt Cython
extension, which depends on the unavailable `circuit_training` package.
