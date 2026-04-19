FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu

COPY . .

RUN mkdir -p /app/results

ENTRYPOINT ["python", "master_eval.py"]
CMD ["--name", "cloud_run", "--time", "3600", "--seed", "42"]
