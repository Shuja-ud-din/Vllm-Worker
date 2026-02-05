FROM nvidia/cuda:12.1.0-base-ubuntu22.04

RUN apt-get update -y && apt-get install -y python3-pip git

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install dependencies
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade -r requirements.txt

# Install vLLM and FlashInfer optimized attention
ARG VLLM_VERSION=0.9.1
RUN python3 -m pip install vllm==${VLLM_VERSION}

# Copy source code
COPY src /src
WORKDIR /src
ENV PYTHONPATH="/:/src"

# Default command
CMD ["python3", "handler.py"]
