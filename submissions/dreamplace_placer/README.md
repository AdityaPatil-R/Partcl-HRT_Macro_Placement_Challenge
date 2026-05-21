# DREAMPlace Placer — real DREAMPlace + LNS refinement

This is the **high-risk / high-reward** path. Real DREAMPlace is what the
top leaderboard teams (vmallela 1.0109, DREAMPlaceProMaxUltra 1.0121) use.
If integration works, you're in top-3 territory.

## How it works

1. Convert benchmark to Bookshelf format (`.aux`/`.nodes`/`.nets`/`.pl`/`.wts`/`.scl`).
2. Invoke DREAMPlace via its Python API (`dreamplace.Placer.place(params)`).
3. Read back the optimized placement.
4. Refine with the existing lns_placer LNS + CD + swap pipeline.

If DREAMPlace fails to import or place (any reason — bad install, format
mismatch, runtime crash), the placer automatically falls back to FastDream
or basic SA, so you don't lose your submission.

## The hard part: getting DREAMPlace to install

DREAMPlace is **notoriously hard to install**. The `Dockerfile` at the repo
root builds it from source pinned to v4.1.0. Things that can break:

- DREAMPlace requires `numpy<2.0` (uses `np.string_`, removed in 2.0).
  Shoom hit this on the leaderboard ("ran fallback SA only").
- CUDA toolkit version mismatch with PyTorch.
- libfftw3 / libboost / libgflags / libgoogle-glog version conflicts.
- The build itself takes 5-15 minutes on a fast machine.

## How to test (in order of confidence)

### 1. Build the Docker image locally

```bash
cd ~/partcl-macro-place-challenge
docker build -t partcl-dreamplace . 2>&1 | tee /tmp/docker_build.log
```

This takes 10-30 minutes. Watch for errors. If it fails, fix the Dockerfile
or skip this approach. Successful build → you have a verified DREAMPlace
install.

### 2. Run a single benchmark in the container

```bash
docker run --rm --gpus all \
  -v $PWD/external:/work/external \
  partcl-dreamplace \
  submissions/dreamplace_placer/placer.py --benchmark ibm04 2>&1 | tail -50
```

Watch for:
- `[DP] running DREAMPlace (...)` — DREAMPlace started
- `[DP] DREAMPlace finished in Xs` — DREAMPlace completed
- `[DP] DREAMPlace placement: calib=...` — placement returned
- `proxy=...` — final proxy

### 3. If step 2 works, test on a larger benchmark

```bash
docker run --rm --gpus all \
  -v $PWD/external:/work/external \
  partcl-dreamplace \
  submissions/dreamplace_placer/placer.py --benchmark ibm10
```

### 4. Submission

If everything works, your submission is **the Dockerfile + the dreamplace_placer
submission**. The judges will build the image themselves.

## Env vars

| Variable | Default | What it does |
|----------|---------|-------------|
| `DP_TOTAL_BUDGET` | 3500 | total wall-clock seconds |
| `DP_GRAD_BUDGET` | 600 | DREAMPlace run seconds (the rest goes to refinement) |
| `DP_FALLBACK` | `"fastdream"` | what to do if DREAMPlace fails (`fastdream` or `lns_placer`) |
| `DP_NUM_BINS_X` | auto | density grid resolution X (power of 2) |
| `DP_NUM_BINS_Y` | auto | density grid resolution Y (power of 2) |
| `DP_VERBOSE` | 1 | print diagnostics |

## Failure modes & what to do

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `dreamplace not available` | Import failed (wrong PYTHONPATH or no install) | Check Dockerfile `RUN python3 -c "import dreamplace"` — must succeed |
| `DREAMPlace.place() raised` | Format issue (Bookshelf files malformed) | Read DREAMPlace's error; probably need to adjust `bookshelf_writer.py` |
| `no output .pl found` | DREAMPlace ran but didn't write expected file | Check `work_dir` contents; DREAMPlace may have a different output name |
| Wrong macro positions | Lower-left vs center coord mismatch | Fix in `placer.py` parsing or `bookshelf_writer.py` writer |

## Expected outcomes

| Scenario | Proxy avg | Leaderboard rank |
|----------|-----------|------------------|
| DREAMPlace works perfectly | 1.05–1.15 | top 3-7 |
| DREAMPlace works but Bookshelf format has issues | 1.20-1.40 | top 12-20 |
| DREAMPlace fails, fallback to FastDream | similar to FastDream alone | top 15-22 |

## Caveats

- This is **untested** end-to-end — I designed the integration but couldn't run it
  on a real DREAMPlace install. There WILL be bugs to fix. Allocate 1-2 days of
  debugging time.
- The Bookshelf format conversion is the most likely failure point. ICCAD04
  uses non-standard node naming and pin offsets. If positions come back wrong,
  inspect `bookshelf_writer.py`.
- ICCAD04 macros are heterogeneous in size, but DREAMPlace assumes uniform
  row height. We work around this by using a single giant row covering the
  canvas — should be fine for placement but may produce slightly suboptimal
  results vs a proper LEF/DEF setup.
- If you're short on time, **submit FastDream as your safe baseline** and try
  DREAMPlace as a bonus. Both can be submitted via different Docker tags
  if needed.
