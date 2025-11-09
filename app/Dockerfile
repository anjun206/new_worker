# 기반 이미지: CUDA 13.0.2 + Ubuntu 22.04
FROM nvidia/cuda:13.0.2-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 필수 시스템 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev git ffmpeg sox libsox-dev libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# python 명령어 기본값 설정
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Python 의존성 설치
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

# (선택) CosyVoice 2.0 설치 및 모델 다운로드
RUN git clone https://github.com/FunAudioLLM/CosyVoice.git /opt/CosyVoice && \
    cd /opt/CosyVoice && git submodule update --init --recursive && \
    python3 -m pip install --no-cache-dir -r requirements.txt || true

RUN python3 -m pip install --no-cache-dir modelscope && \
    python3 -c "from modelscope import snapshot_download; \
snapshot_download('iic/CosyVoice2-0.5B', local_dir='/opt/CosyVoice/pretrained_models/CosyVoice2-0.5B')" && \
    python3 -c "from modelscope import snapshot_download; \
snapshot_download('iic/CosyVoice-ttsfrd', local_dir='/opt/CosyVoice/pretrained_models/CosyVoice-ttsfrd')"

# 애플리케이션 코드 복사
COPY . /app

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
