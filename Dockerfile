# ============================================================
# Dockerfile — KeepDreaming submission for the PartCL x HRT Macro Placement
# Challenge 2026. Builds a self-contained image with DREAMPlace 4.1.0 + Xplace
# pre-installed, then runs `submissions/portfolio_placer/placer.py` by default.
#
# Submission entry point:  submissions/portfolio_placer/placer.py
# Method:                  TriSafeLNS Portfolio (avg proxy 1.4506 on IBM)
#
# Build (~30-45 min on first run; subsequent builds use docker layer cache):
#   docker build -t partcl-keepdreaming .
#
# Run on a single benchmark:
#   docker run --rm --gpus all --network none \
#     -v $PWD/external:/work/external partcl-keepdreaming \
#     submissions/portfolio_placer/placer.py --benchmark ibm04
#
# Run on all 17 IBM benchmarks (default behavior):
#   docker run --rm --gpus all --network none \
#     -v $PWD/external:/work/external partcl-keepdreaming
# ============================================================

# Build with CUDA 11.8 / pytorch 2.1.2 — narrow sweet spot:
#   * DREAMPlace 4.1.0's CUB usage breaks on CUDA 12.x's CUB 2.3+
#     (accumulator_pack_base_t removed). Stay on 11.x.
#   * The L4 GPU is Ada (sm_89). CUDA 11.7's cuFFT does NOT have an sm_89
#     kernel path and raises CUFFT_INTERNAL_ERROR on any rfft. CUDA 11.8 is
#     the first version with Ada cuFFT support.
# Hence: CUDA 11.8 is the only CUDA that simultaneously builds DREAMPlace
# AND runs cuFFT on the L4.
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel AS dreamplace-builder

# Build-time deps for DREAMPlace.
# `libfl-dev` provides FlexLexer.h that DREAMPlace's Limbo parsers need.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git wget ca-certificates \
        flex libfl-dev bison libgflags-dev libgoogle-glog-dev \
        libboost-all-dev libfftw3-dev \
        libcairo2-dev libpng-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

# DREAMPlace requires numpy<2.0 (uses np.string_, removed in 2.0)
RUN pip install --no-cache-dir 'numpy<2.0' pybind11 scipy 'matplotlib<3.10'

# Force PyTorch CUDA arch list at build time. The base image has CUDA but
# no GPU during build, so torch.cuda.is_available() returns False. Setting
# TORCH_CUDA_ARCH_LIST tells the build to compile CUDA kernels for these
# architectures anyway — they'll work at runtime when a GPU is present.
# 8.9 = Ada Lovelace (RTX 6000 Ada, competition GPU); 8.6 = RTX 3090 / A6000;
# 8.0 = A100; 7.5 = T4 / RTX 20-series; covers everything plausible.
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9"
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV CMAKE_CUDA_COMPILER=${CUDA_HOME}/bin/nvcc
ENV CUDACXX=${CUDA_HOME}/bin/nvcc
ENV CUDA_TOOLKIT_ROOT_DIR=${CUDA_HOME}

# Build DREAMPlace from source.
# IMPORTANT: use /usr/bin/cmake (apt-installed, ~3.22) NOT /opt/conda/bin/cmake
# (3.30.5). DREAMPlace 4.1.0 uses the legacy `find_package(CUDA)` module, which
# CMake 3.27+ silently ignores — manually-specified CMAKE_CUDA_COMPILER /
# CUDA_TOOLKIT_ROOT_DIR flags get a "manually-specified variables were not used"
# warning, CUDA_FOUND stays blank, and only CPU .so kernels are built. The apt
# cmake 3.22 still honors the legacy FindCUDA module.
WORKDIR /opt
RUN git clone --recursive https://github.com/limbo018/DREAMPlace.git \
    && cd DREAMPlace \
    && git checkout 4.1.0 \
    && \
    # CRITICAL: DREAMPlace's cmake/TorchExtension.cmake detects CUDA via
    # `torch.cuda.is_available()` which returns False during `docker build`
    # (no GPU at build time). That sets TORCH_ENABLE_CUDA=0 and skips compiling
    # all _cuda.so kernels, even though the pytorch binary itself has CUDA
    # support. Patch it to use `torch.backends.cuda.is_built()` instead, which
    # checks whether pytorch was COMPILED with CUDA — true regardless of GPU
    # presence at build time.
    sed -i 's|torch.cuda.is_available()|torch.backends.cuda.is_built()|g' \
        cmake/TorchExtension.cmake \
    && grep -n 'is_built\|is_available' cmake/TorchExtension.cmake \
    && mkdir build \
    && cd build \
    && /usr/bin/cmake .. -DCMAKE_INSTALL_PREFIX=/opt/dreamplace_install \
                 -DPython_EXECUTABLE=$(which python3) \
                 -DCMAKE_CUDA_ARCHITECTURES="7.5;8.0;8.6" \
                 -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
                 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    && make -j4 \
    && make install

# Verify CUDA actually got linked into the build. CUDA_FOUND=TRUE in
# configure.py is the canonical check; missing _cuda.so files is the
# downstream symptom. Fail the build loudly if it didn't take.
RUN python3 -c "g={};exec(open('/opt/dreamplace_install/dreamplace/configure.py').read(),g); \
    cf=str(g['compile_configurations'].get('CUDA_FOUND','')).upper(); \
    print('CUDA_FOUND:', cf); \
    assert cf=='TRUE', 'DREAMPlace built without CUDA — check cmake output above'" \
    && ls /opt/dreamplace_install/dreamplace/ops/electric_potential/*.so

# ── Xplace (CUHK-EDA) — Carrotato's #1 placer uses Xplace + Triton + polish ─
# Xplace optimizes routing congestion (RUDY) directly — fixes the alignment
# issue we hit with DREAMPlace (HPWL+density only). Requires CMake≥3.24 (we
# have conda's 3.30 in PATH), boost (already installed), pytorch (in base).

# Xplace's Python source imports seaborn / igraph / numba / pulp / torchvision.
# Installed AFTER DREAMPlace builds so we don't invalidate that layer's cache.
RUN pip install --no-cache-dir seaborn 'igraph' numba pulp torchvision

WORKDIR /opt
RUN git clone --recursive https://github.com/cuhk-eda/Xplace.git \
    && cd Xplace \
    && \
    # Same root cause as DREAMPlace: CMakeLists.txt:71 invokes
    # `torch.cuda.is_available()` at config time, which is False during
    # `docker build` (no GPU at build time) and causes Xplace's CMakeLists.txt
    # to FATAL_ERROR with "Xplace only supports Torch-CUDA mode". Patch to
    # use `torch.backends.cuda.is_built()` which returns true if the pytorch
    # binary was compiled with CUDA support — independent of runtime GPU.
    sed -i 's|torch.cuda.is_available()|torch.backends.cuda.is_built()|g' \
        CMakeLists.txt \
    && grep -n 'is_built\|is_available' CMakeLists.txt \
    && mkdir build && cd build \
    && cmake .. -DCMAKE_CUDA_ARCHITECTURES="80;86" \
                 -DPYTHON_EXECUTABLE=$(which python3) \
    && make -j4 \
    && make install
RUN python3 -c "import sys; sys.path.insert(0, '/opt/Xplace'); from src import run_placement_main; print('Xplace OK')"

# ── Final stage: runtime image ────────────────────────────────────────────────
# Match the builder's CUDA version — DREAMPlace .so files compiled against
# CUDA 11.8 need the matching runtime libs.
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

# DREAMPlace's compiled .so modules dynamically link against system libs
# (fftw3, gflags, glog, boost). The builder stage had the -dev variants; the
# runtime stage needs the matching runtime variants or `import dreamplace`
# will fail with "cannot open shared object file".
ENV DEBIAN_FRONTEND=noninteractive
# Ubuntu 20.04 (pytorch 2.0.1 base) ships boost 1.71.0, not 1.74.0 (22.04).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libfftw3-double3 libfftw3-single3 \
        libgflags2.2 libgoogle-glog0v5 \
        libboost-system1.71.0 libboost-filesystem1.71.0 \
        libboost-thread1.71.0 libboost-iostreams1.71.0 \
        libcairo2 libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

# Same numpy<2.0 constraint at runtime.
# shapely + networkx are needed by DREAMPlace's fence_region and related ops.
# seaborn/igraph/numba/pulp/torchvision are required by Xplace's Python source.
RUN pip install --no-cache-dir 'numpy<2.0' 'matplotlib<3.10' absl-py tqdm scipy \
        shapely networkx pyunpack patool cairocffi Pillow \
        seaborn 'igraph' numba pulp torchvision

# Carry over DREAMPlace install from builder stage.
# dreamplace/Placer.py uses BARE imports (`import Params`, `import PlaceDB`,
# `import NonLinearPlace`, `import Timer`) for its sibling modules, so we
# need the inner directory on PYTHONPATH as well — not just the parent.
COPY --from=dreamplace-builder /opt/dreamplace_install /opt/dreamplace_install
# Carry over Xplace install (entire source tree, since it's invoked as a
# Python module rooted at /opt/Xplace via `main.py`).
COPY --from=dreamplace-builder /opt/Xplace /opt/Xplace
ENV PYTHONPATH="/opt/dreamplace_install:/opt/dreamplace_install/dreamplace:/opt/Xplace:${PYTHONPATH}"

# Carry over the project code
WORKDIR /work
COPY pyproject.toml requirements.txt* ./
COPY macro_place ./macro_place
COPY submissions ./submissions
# external/MacroPlacement is mounted at runtime (large benchmark data)

# Install our project (no torch — already in base image)
RUN pip install --no-cache-dir -e . 2>/dev/null || pip install --no-cache-dir -e . --no-build-isolation

# Sanity check: can DREAMPlace import AND can its C++ extensions load?
# We fail the build loudly here — a silent import failure would mean every
# `docker run` falls back to FastDream and we'd never know real DREAMPlace
# isn't working.
#
# `import dreamplace.Placer` is the real test: it transitively triggers the
# bare imports (Params, PlaceDB, Timer, NonLinearPlace) that v4.1.0's
# Placer.py uses, and was missed by a simple `import dreamplace`.
RUN python3 -c "import dreamplace; print('DREAMPlace OK:', dreamplace.__file__)"
RUN python3 -c "from dreamplace.ops.hpwl import hpwl; print('hpwl ext OK')"
RUN python3 -c "import dreamplace.Placer; import dreamplace.Params; print('Placer + Params import OK')"

# Unbuffered Python so stdout isn't lost if the process is killed mid-run
# (e.g. OOM). Critical for debugging container crashes via `docker logs`/`tee`.
ENV PYTHONUNBUFFERED=1

# Default command (override with --benchmark)
ENTRYPOINT ["python3", "-m", "macro_place.evaluate"]
# Default to the actual submission: TriSafeLNS Portfolio. Run all 17 IBM
# benchmarks by default. Override with `--benchmark ibmNN` for single runs.
CMD ["submissions/portfolio_placer/placer.py", "--all"]

