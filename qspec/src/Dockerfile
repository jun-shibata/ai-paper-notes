FROM ghcr.io/astral-sh/uv:0.12.3 AS uv_source
FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04 AS qspec

# Initial UV settings
COPY --from=uv_source /uv /uvx /usr/local/bin/
RUN uv --version

# Set environments variables
ENV MOFED_VER 5.3-1.0.0.1
ENV OS_VER ubuntu22.04
ENV PLATFORM x86_64
ENV HOME /root

# Copy QSpec directory (original source code)
COPY . /workspace/QSpec
WORKDIR /workspace/QSpec

# Do not reuse CMake caches created outside this container
RUN rm -rf \
    /workspace/QSpec/.deps \
    /workspace/QSpec/build \
    /workspace/QSpec/dist

ENV MAX_JOBS=4
ENV NVCC_THREADS=1

# NVIDIA L40S: compute capability 8.9
ENV TORCH_CUDA_ARCH_LIST=8.9

# QSpec setup.py invokes pip internally for fast-hadamard-transform
ENV PIP_NO_BUILD_ISOLATION=1

# change data path and number of trainers here
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3.10-venv \
        build-essential ninja-build git git-lfs \
        wget curl ca-certificates kmod && \
    rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/qspec-venv
ENV PATH=/opt/qspec-venv/bin:${PATH}
ENV UV_PYTHON_DOWNLOADS=never

RUN uv venv \
    --python /usr/bin/python3.10 \
    --seed \
    "${VIRTUAL_ENV}"

RUN python --version && uv --version

# Build dependencies
RUN uv pip install -r requirements-build.txt
RUN uv pip install -r requirements-cuda.txt
# Build QSpec's vLLM
ENV MAX_JOBS=4

RUN uv pip install \
    --no-build-isolation \
    --editable .

# QSpec-related packages
RUN uv pip install \
    --no-deps \
    --no-build-isolation \
    --editable third-party/ao

RUN uv pip install \
    --no-deps \
    --no-build-isolation \
    --editable third-party/fast-hadamard-transform

RUN uv pip install \
    --no-deps \
    --no-build-isolation \
    --editable third-party/QuaRot

RUN uv pip install \
    --no-deps \
    --no-build-isolation \
    third-party/kernels

RUN uv pip install "bitblas==0.1.0.post1"

# Restore versions verified with QSpec
RUN uv pip install \
    "transformers==4.47.1" \
    "numpy==1.26.4" \
    "fsspec==2026.6.0"

RUN uv pip check

CMD /bin/bash