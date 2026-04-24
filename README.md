# DGX Spark GB10: Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice через vLLM-Omni

Репозиторий приведён в соответствие с гайдом NVIDIA Forum по запуску `vllm-omni` на DGX Spark (ARM64), включая шаги для `fa3-fwd` и Flash Attention 2.

## Запуск

```bash
cp .env.example .env
docker compose up --build -d
```

## Проверка

```bash
curl http://localhost:8091/health
curl http://localhost:8091/v1/models
```

## Почему была ошибка при сборке

Если вы видели ошибку вида `externally-managed-environment` / ссылку на PEP 668 на шаге `python3 -m pip install ...`, причина в том, что в Ubuntu 24.04 системный Python защищён от установки пакетов в system-site-packages.

В Dockerfile это исправлено: зависимости ставятся в отдельное виртуальное окружение `/opt/venv`, поэтому установка больше не конфликтует с PEP 668.

## Соответствие гайду NVIDIA (проверочный список)

- Устанавливаются системные зависимости `ffmpeg` и `sox`.
- Устанавливается ARM64 wheel `vllm==0.16.0+cu130`.
- Клонируется `vllm-omni` и ставится `uv pip install -e .`.
- Из `requirements/cuda.txt` удаляется `fa3-fwd==0.0.2` (как рекомендовано для ARM64).
- Flash Attention 2 ставится **из исходников** через `uv pip install -v --no-build-isolation .`.
- Для сборки Flash Attention выставляются параметры:
  - `MAX_JOBS=4`
  - `NVCC_THREADS=2`
  - `FLASH_ATTENTION_FORCE_BUILD=TRUE`
- Сервер стартует командой `vllm serve ... --stage-configs-path .../qwen3_tts.yaml --omni --port 8091 --trust-remote-code --enforce-eager`.

## Flash Attention на DGX Spark

По гайду для Blackwell/ARM64 рекомендуется Flash Attention 2.

В этом репозитории по умолчанию:

- `INSTALL_FLASH_ATTN=1`
- `ATTENTION_BACKEND=FLASH_ATTN`
- `FLASH_ATTN_MAX_JOBS=4`
- `FLASH_ATTN_NVCC_THREADS=2`

Если Flash Attention не собрался (например, нехватка RAM/времени сборки), можно временно переключиться на SDPA:

```bash
INSTALL_FLASH_ATTN=0
ATTENTION_BACKEND=TORCH_SDPA
```

И затем снова выполнить:

```bash
docker compose up --build -d
```

## Пример генерации речи

```bash
curl -X POST "http://localhost:8091/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "input": "Привет! Это прямой запуск vLLM-Omni на DGX Spark.",
    "voice": "calm"
  }' --output speech.wav
```

## Voice clone пример

```bash
curl -X POST "http://localhost:8091/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "input": "Тест клонирования голоса",
    "reference_audio_path": "/workspace/samples/ref.wav",
    "reference_text": "Оригинальный текст референсной записи"
  }' --output clone.wav
```

Оригинальная инструкция:
https://forums.developer.nvidia.com/t/running-vllm-omni-for-qwen3-tts-voice-design-voice-clone-on-dgx-spark/361255
