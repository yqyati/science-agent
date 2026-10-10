FROM registry.h.pjlab.org.cn/ailab/ml-base:22.04-pjlab-20260323
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git poppler-utils python3-cairo \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/science-requirements.txt
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_EXTRA_INDEX_URL
RUN python3 -m pip --version \
    && PIP_CONFIG_FILE=/dev/null python3 -m pip install --no-cache-dir --only-binary=:all: -r /tmp/science-requirements.txt \
    && python3 -m pip check \
    && rm /tmp/science-requirements.txt
RUN mkdir -p /workspace/pages
WORKDIR /workspace
