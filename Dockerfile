# Use the official Python base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /Emotion-Detection

# Install system dependencies for OpenCV (and other required libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-dev \
    libopencv-contrib-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir \
    numpy \
    opencv-python-headless \
    opencv-contrib-python \
    matplotlib \
    tensorflow==2.12.0  # Version supported for ARM

# Expose the display (for webcam usage, if needed)
ENV DISPLAY=:0

# Copy all the necessary files from the host to the container
COPY . /Emotion-Detection

# Command to run the live emotion detection script
CMD ["python", "live.py"]
