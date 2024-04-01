# Base image
FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    unzip \
    build-essential

# Install Python dependencies
RUN pip install --no-cache-dir \
    transformers \
    datasets \
    tokenizers \
    accelerate \
    nltk \
    sentencepiece \
    wandb

# Download and install Apex
RUN git clone https://github.com/NVIDIA/apex.git \
    && cd apex \
    && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ \
    && cd ..

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Copy your fine-tuning script and dataset
COPY fine_tune_llm.py /app/
COPY dataset/ /app/dataset/

# Set the entry point command
CMD ["python", "fine_tune_llm.py"]