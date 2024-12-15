# Base image from NVIDIA L4T for Jetson with TensorFlow and OpenCV
FROM nvcr.io/nvidia/l4t-tensorflow:r32.7.1-tf2.5-py3

# Set the working directory
WORKDIR /emotion_detection

# Copy the entire project into the container
COPY . /emotion_detection

# Install additional system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install additional Python dependencies (if needed)
RUN pip install --no-cache-dir numpy

# Expose a port for debugging or APIs (optional)
EXPOSE 8080

# Set the default command to run your live detection script
CMD ["python3", "live.py"]
