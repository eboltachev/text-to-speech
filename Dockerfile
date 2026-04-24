FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_SYSTEM_PYTHON=0 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

ARG INSTALL_FLASH_ATTN=1
ARG FLASH_ATTN_MAX_JOBS=4
ARG FLASH_ATTN_NVCC_THREADS=2

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    ninja-build \
    git \
    curl \
    ca-certificates \
    ffmpeg \
    sox \
 && rm -rf /var/lib/apt/lists/*

# Avoid PEP 668 (externally-managed environment) by using an isolated venv.
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip setuptools wheel uv

WORKDIR /opt
RUN git clone --depth 1 https://github.com/vllm-project/vllm-omni.git

WORKDIR /opt/vllm-omni

# DGX Spark (ARM64) install path from NVIDIA forum guide.
RUN uv pip install \
      https://github.com/vllm-project/vllm/releases/download/v0.16.0/vllm-0.16.0+cu130-cp38-abi3-manylinux_2_35_aarch64.whl \
      --extra-index-url https://download.pytorch.org/whl/cu130 \
      --index-strategy unsafe-best-match

# fa3-fwd has no aarch64 wheel; forum guide recommends removing/commenting this dependency.
RUN sed -i '/^fa3-fwd==0.0.2$/d' requirements/cuda.txt

RUN uv pip install -e .

# Optional Flash Attention 2 build from source (forum-aligned workflow).
RUN if [ "$INSTALL_FLASH_ATTN" = "1" ]; then \
      git clone --depth=1 https://github.com/Dao-AILab/flash-attention /opt/flash-attention && \
      cd /opt/flash-attention && \
      export MAX_JOBS="$FLASH_ATTN_MAX_JOBS" && \
      export NVCC_THREADS="$FLASH_ATTN_NVCC_THREADS" && \
      export FLASH_ATTENTION_FORCE_BUILD="TRUE" && \
      uv pip install -v --no-build-isolation .; \
    else \
      echo "Skipping Flash Attention build (INSTALL_FLASH_ATTN=0)"; \
    fi

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8091

ENTRYPOINT ["/entrypoint.sh"]
