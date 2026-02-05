FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

RUN apt-get update -y && apt-get install -y \
    python3-pip git build-essential ninja-build \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

COPY requirements.txt /requirements.txt

# Install dependencies (NO CACHE to save disk)
RUN pip install --no-cache-dir -r /requirements.txt

# Install vLLM separately (NO CACHE)
ARG VLLM_VERSION=0.9.1
RUN pip install --no-cache-dir vllm==${VLLM_VERSION}

# Optional: Flash Attention (skip if build fails)
# RUN pip install --no-cache-dir flash-attn --no-build-isolation

COPY src /src
WORKDIR /src

ENV PYTHONPATH="/src"

CMD ["python3", "-m", "handler"]
