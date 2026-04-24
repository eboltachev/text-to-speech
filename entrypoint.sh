#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice}"
PORT="${PORT:-8091}"
HOST="${HOST:-0.0.0.0}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-FLASH_ATTN}"

# Possible values depend on installed backends (FLASH_ATTN, TORCH_SDPA, etc.)
export VLLM_ATTENTION_BACKEND="${ATTENTION_BACKEND}"

cd /opt/vllm-omni

exec vllm serve "${MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --stage-configs-path vllm_omni/model_executor/stage_configs/qwen3_tts.yaml \
  --omni \
  --trust-remote-code \
  --enforce-eager
