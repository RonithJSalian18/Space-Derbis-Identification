# Use lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (required for OpenCV)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    tensorflow \
    opencv-python \
    scikit-learn \
    seaborn \
    matplotlib

# Expose port (optional, useful if you later add API)
EXPOSE 8000

# Default command
CMD ["python", "CNN/main.py"]