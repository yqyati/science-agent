FROM registry.h.pjlab.org.cn/ailab/ml-base:22.04-pjlab-20260323
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git poppler-utils \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/science-requirements.txt
RUN python3 -m pip --version \
    && python3 -m pip install --no-cache-dir --only-binary=:all: -r /tmp/science-requirements.txt \
    && python3 -m pip check \
    && rm /tmp/science-requirements.txt
RUN mkdir -p /workspace/pages
WORKDIR /workspace
