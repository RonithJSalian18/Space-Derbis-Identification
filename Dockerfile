# Use NVIDIA CUDA base image with Python
FROM tensorflow/tensorflow:2.15.0-gpu

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Upgrade pip
RUN pip install --upgrade pip

# Install required packages
RUN pip install \
    opencv-python \
    matplotlib \
    seaborn \
    scikit-learn \
    tqdm

# Expose port (optional if you later add Flask)
EXPOSE 5000

# Run script
CMD ["python", "cnn.py"]
