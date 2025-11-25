# Start from NVIDIA's CUDA image with cuDNN and Ubuntu 22.04
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Set environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cuda/bin:$PATH

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-distutils \
    python3.11-venv \
    git \
    build-essential \
    libpng-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Create and activate a virtual environment for Python 3.11
RUN python3.11 -m venv /opt/venv

# Ensure the virtual environment is used for all Python commands
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip in the virtual environment
RUN pip install --upgrade pip

# Accept UID/GID from build args with defaults
ARG USER_ID=1000
ARG GROUP_ID=1000

# Create group and user matching host UID/GID
RUN groupadd -g ${GROUP_ID} appgroup && \
    useradd -m -u ${USER_ID} -g appgroup appuser

# Set the working directory inside the container
WORKDIR /workspace

# Install Python dependencies (requirements.txt should be in your repo)
COPY requirements.txt /workspace/
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN apt-get update && apt-get install -y libglib2.0-0
RUN pip install -r /workspace/requirements.txt

# Copy your repository code into the container
#COPY . /workspace/

# Switch to that user
USER appuser

# Command to run your app
CMD ["bash"]
