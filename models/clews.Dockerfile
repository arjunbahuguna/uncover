FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1

WORKDIR /app

# System deps for audio decoding and Python package builds.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install the dependencies listed by the project README/install script.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    lightning==2.3.0 \
    tensorboard==2.17.0 \
    einops==0.8.0 \
    torchinfo==1.8.0 \
    omegaconf==2.3.0 \
    tqdm==4.66.4 \
    joblib==1.4.2 \
    soundfile==0.12.1 \
    soxr==0.3.7 \
    nnAudio==0.3.3 \
    numpy==1.26.4 \
    julius==0.2.7

COPY models/clews/ /app/

CMD ["bash"]
