FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    default-jre-headless \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml LICENSE README.md /app/
COPY src /app/src
COPY docs /app/docs
COPY configs /app/configs
COPY data /app/data
COPY models /app/models

RUN pip install --upgrade pip && pip install -e .

CMD ["aml-serve"]
