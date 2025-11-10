ARG BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

FROM ${BASE_IMAGE} AS base-utils

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 필수 시스템 패키지와 공통 유틸
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev git ffmpeg sox libsox-dev libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# python 명령어 기본값 설정
RUN ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /workspace

FROM base-utils AS python-common

COPY requirements /tmp/requirements/
ENV PIP_CONSTRAINT=/tmp/requirements/pins.txt

# 공통 파이썬 의존성
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir -r /tmp/requirements/base.txt && \
    python3 -m pip install --no-cache-dir -r /tmp/requirements/python.txt

# 🔧 [추가] cuDNN 9.1 런타임 설치 + 로더 경로 노출
RUN python3 -m pip install --no-cache-dir "nvidia-cudnn-cu12>=9.1,<9.2"
ENV LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# PyTorch 2.8.0 CUDA 12.8(+cu128) 휠 사용
# ARG TORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu128
# RUN python3 -m pip install --no-cache-dir --index-url $TORCH_CUDA_INDEX \
#     torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

FROM python-common AS stt-whisperx


# STT/WhisperX 계층
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements/whisperx.txt

FROM stt-whisperx AS tts-cosyvoice

# CosyVoice 및 모델 관련 의존성
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements/cosyvoice.txt

ARG CLONE_COSYVOICE=true
ENV COSYVOICE_DIR=/opt/CosyVoice
RUN if [ "$CLONE_COSYVOICE" = "true" ]; then \
        git clone https://github.com/FunAudioLLM/CosyVoice.git ${COSYVOICE_DIR} && \
        cd ${COSYVOICE_DIR} && git submodule update --init --recursive && \
        python3 -m pip install --no-cache-dir -r requirements.txt || true; \
    fi

# CosyVoice 모델 다운로드
ARG DOWNLOAD_COSYVOICE_MODELS=false

RUN if [ "$DOWNLOAD_COSYVOICE_MODELS" = "true" ]; then \
        python3 -c "from modelscope import snapshot_download; \
snapshot_download('iic/CosyVoice2-0.5B', local_dir='/opt/CosyVoice/pretrained_models/CosyVoice2-0.5B')" && \
        python3 -c "from modelscope import snapshot_download; \
snapshot_download('iic/CosyVoice-ttsfrd', local_dir='/opt/CosyVoice/pretrained_models/CosyVoice-ttsfrd')"; \
    fi

FROM tts-cosyvoice AS extra

# 기타 의존성
RUN if [ -s /tmp/requirements/extra.txt ]; then \
        python3 -m pip install --no-cache-dir -r /tmp/requirements/extra.txt; \
    fi

FROM extra AS app-base

WORKDIR /app

# 애플리케이션 코드 복사
COPY . /app

ENV APP_MODULE=main:app \
    APP_PORT=8000

EXPOSE 8000

FROM app-base AS runtime

ENV UVICORN_RELOAD=

CMD ["sh", "-c", "uvicorn ${APP_MODULE:-main:app} --host 0.0.0.0 --port ${APP_PORT:-8000} ${UVICORN_RELOAD:-}"]

FROM app-base AS dev

ENV UVICORN_RELOAD=--reload

CMD ["sh", "-c", "uvicorn ${APP_MODULE:-main:app} --host 0.0.0.0 --port ${APP_PORT:-8000} ${UVICORN_RELOAD:-}"]

FROM runtime
