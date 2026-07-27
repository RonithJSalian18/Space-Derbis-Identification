# Use official TensorFlow GPU container image
FROM tensorflow/tensorflow:2.15.0-gpu

# Set working directory
WORKDIR /app

# Install system dependencies (e.g. libgl1 for OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy source code and files
COPY . /app

# Set environment variables
ENV PYTHONPATH=/app

# Default command runs the unified training pipeline for CNN
CMD ["python", "train.py", "--model", "cnn"]