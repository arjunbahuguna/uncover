FROM continuumio/miniconda3:24.7.1-0

WORKDIR /app

# Create the same conda env as in the project setup instructions.
COPY models/Discogs-VINet/env.yaml /tmp/env.yaml
RUN sed '/^prefix:/d' /tmp/env.yaml > /tmp/env-clean.yaml && \
    conda env create -f /tmp/env-clean.yaml && \
    conda clean -afy

SHELL ["/bin/bash", "-c"]
