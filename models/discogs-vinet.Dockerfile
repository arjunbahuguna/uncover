FROM continuumio/miniconda3:24.7.1-0

WORKDIR /app

# Step 1: Copy the environment configuration file and remove the hardcoded prefix.
# The 'prefix' field in conda export files is machine-specific and must be removed
# to allow successful environment creation in a new container.
COPY models/Discogs-VINet/env.yaml /tmp/env.yaml
RUN sed '/^prefix:/d' /tmp/env.yaml > /tmp/env-clean.yaml && \
    conda env create -f /tmp/env-clean.yaml && \
    conda clean -afy

# Step 2: Critical Fix - Prioritize the Python path of the 'discogs-vinet' environment.
# By prepending this path to the system PATH variable, running 'python' directly
# will invoke the interpreter from this specific environment, ensuring access
# to the correct dependencies (e.g., torch) without needing to activate the env manually.
ENV PATH /opt/conda/envs/discogs-vinet/bin:$PATH

# Step 3: Patch - Explicitly install GPU-enabled PyTorch packages.
# This ensures that the specific CUDA-compatible versions of torch, torchvision,
# and torchaudio are installed within the target environment, overriding any
# CPU-only versions that might have been defined in the original YAML file.
RUN conda run -n discogs-vinet pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

SHELL ["/bin/bash", "-c"]
CMD ["bash"]