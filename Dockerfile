FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

RUN apt-get update -y && apt-get install -y \
    python3-pip git build-essential ninja-build

RUN python3 -m pip install --upgrade pip

COPY requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /requirements.txt

# Install vLLM
ARG VLLM_VERSION=0.9.1
RUN pip install vllm==${VLLM_VERSION}

# Install Flash Attention 2 (BEST for H100/A100)
RUN pip install flash-attn --no-build-isolation

# Copy source
COPY src /src
WORKDIR /src

ENV PYTHONPATH="/src"

CMD ["python3", "-m", "handler"]
