# Use NVIDIA CUDA base image with Python
FROM tensorflow/tensorflow:2.15.0-gpu

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install system dependencies (FIX for libxcb error)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install required packages (use headless OpenCV)
RUN pip install \
    opencv-python-headless \
    matplotlib \
    seaborn \
    scikit-learn \
    tqdm

# Expose port (optional)
EXPOSE 5000

# Run script
CMD ["python", "cnn.py"]