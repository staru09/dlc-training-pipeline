# Use PyTorch with CUDA 12.1 and cuDNN 8 as the base image for GPU inference.
# (Cloud Run L4 GPUs support CUDA 12.x)
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
# ffmpeg and libgl1 are required by OpenCV and DeepLabCut
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy python dependencies
COPY requirements.txt .

# Install dependencies
# 1. Upgrade pip
# 2. Install numpy/scipy as binary-only wheels first (avoids source-build failures)
# 3. Install the rest (deeplabcut will reuse the already-installed scipy)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --only-binary=numpy,scipy "numpy>=1.23.0,<2.0.0" "scipy>=1.11.0,<1.14.0" && \
    pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy backend application code
COPY . .

# Create uploads and outputs directories
RUN mkdir -p uploads outputs

# Expose the port that Cloud Run expects
EXPOSE 8080

# Start the FastAPI app using Uvicorn
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
