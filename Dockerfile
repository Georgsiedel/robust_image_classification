FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Set environment variables natively
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# Redirect Matplotlib cache to avoid home directory permission issues
ENV MPLCONFIGDIR=/tmp/matplotlib

# Prevent interactive installation prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install all system requirements cleanly in one layer
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-distutils \
    python3.10-venv \
    git \
    build-essential \
    libpng-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create and isolate virtual environment
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip

# Create unprivileged execution target matching host identity
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} appgroup && \
    useradd -m -u ${USER_ID} -g appgroup appuser

WORKDIR /workspace

#Force pip on torch for cuda 12.4
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install requirements before copying target logic to leverage cache layers
COPY requirements.txt /workspace/
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# Force permissions in various directories
RUN chmod 755 /home && \
    chmod 1777 /tmp && \
    chown -R appuser:appgroup /home/appuser /workspace /opt/venv && \
    chmod -R 755 /home/appuser /bin /usr/bin /usr/local/bin /lib /lib64 /opt/venv

# Switch context to the mapped unprivileged identity safely
USER appuser

CMD ["bash"]