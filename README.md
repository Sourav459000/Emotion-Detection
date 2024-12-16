# Real-Time Emotion Detection System

### Overview
The **Real-Time Emotion Detection System** leverages cutting-edge deep learning techniques to recognize and classify human emotions in real-time using facial expressions. Designed for edge deployment, the system is optimized for low-latency processing on the NVIDIA Jetson Orin Nano, making it ideal for applications such as human-computer interaction, behavioral analysis, and smart environments.

---

### Features
- **Real-Time Emotion Detection**: Processes video input from a webcam with minimal latency.
- **Edge Deployment**: Runs efficiently on the NVIDIA Jetson Orin Nano.
- **Pre-Trained CNN Model**: Utilizes a convolutional neural network trained on FER2013 and CK+ datasets.
- **Dockerized Environment**: Ensures portability and reproducibility.
- **Scalable**: Easily integrates into other systems or use cases.

---

### System Requirements
- **Hardware**: 
  - NVIDIA Jetson Orin Nano
  - USB Webcam
- **Software**:
  - Ubuntu 20.04 or higher
  - Docker (latest version)
  - Python 3.10 or higher

---

### Installation

#### 1. Clone the Repository
```bash
git clone https://github.com/your-repo/emotion-detection.git
cd emotion-detection
```

#### 2. Build the Docker Image
```bash
docker build -t emotion-detection .
```

#### 3. Run the Docker Container
```bash
xhost +local:docker
docker run -it --rm --device /dev/video0:/dev/video0 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix emotion-detection
```

---

### File Structure
```
emotion-detection/
│
├──train.py                # Training the model
├── live.py                # Main script for real-time emotion detection
├── emotion_detection_model.h5                 # Trained CNN model and weights
├── Dockerfile             # Docker configuration file
├── requirements.txt       # Python dependencies
├── README.md              # Project overview and setup guide
└── utils/                 # Helper functions and preprocessing scripts
```

---

### How It Works
1. The system captures live video frames from the connected webcam.
2. Each frame undergoes preprocessing (resizing, normalization, grayscale conversion).
3. The pre-trained CNN model classifies emotions such as happiness, sadness, anger, etc.
4. Results are displayed in real-time with emotion labels overlaid on the video feed.

---

### Datasets Used
- **FER2013**: Facial Expression Recognition dataset with thousands of labeled images.
- **CK+**: The Extended Cohn-Kanade dataset for emotion detection.

---

### Performance Metrics
- **Accuracy**: ~90% on FER2013
- **Latency**: ~40ms/frame on Jetson Orin Nano
- **Supported Emotions**: Happiness, Sadness, Anger, Surprise, Neutral

---

### Future Enhancements
- Expand emotion classes with additional datasets.
- Optimize the model further for even lower latency.
- Integrate audio-based emotion recognition for multimodal analysis.

---

### Contributing
We welcome contributions! Feel free to submit issues or pull requests to improve the project.

---

### Acknowledgments
- **NVIDIA**: For providing the Jetson Orin Nano and Deep Learning resources.
- **FER2013**: For the datasets used to train the model.
