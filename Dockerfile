# Use an official CUDA base image with Ubuntu 22.04
FROM nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04

# Install system dependencies: Python, FFmpeg, SoX, and others
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev git ffmpeg sox libsox-dev libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# Set Python3 as default
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install Python dependencies (GPU-enabled Torch, FastAPI, etc.)
RUN pip install --no-cache-dir torch==2.1.0 torchvision==0.12.0 torchaudio==2.1.0 \
    fastapi uvicorn pydantic ffmpeg-python whisperx demucs==4.0.1 \
    googletrans==4.0.0-rc1 pydub

# (Optional) Install CosyVoice 2.0 and download model weights
# Clone CosyVoice repository
RUN git clone https://github.com/FunAudioLLM/CosyVoice.git /app/CosyVoice && \
    cd /app/CosyVoice && git submodule update --init --recursive
# Install CosyVoice dependencies (using requirements.txt if available)
RUN cd /app/CosyVoice && pip install -r requirements.txt || true 
# Download pre-trained CosyVoice models (if not already packaged)
RUN pip install modelscope && python -c "from modelscope import snapshot_download; \
    snapshot_download('iic/CosyVoice2-0.5B', local_dir='/app/CosyVoice/pretrained_models/CosyVoice2-0.5B')" && \
    python -c "from modelscope import snapshot_download; \
    snapshot_download('iic/CosyVoice-ttsfrd', local_dir='/app/CosyVoice/pretrained_models/CosyVoice-ttsfrd')"

# Copy application code
WORKDIR /app
COPY stt.py demucs_split.py translate.py tts.py mux.py main.py /app/

# Expose port for API
EXPOSE 8000

# Start the FastAPI server with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
