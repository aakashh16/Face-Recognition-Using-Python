# Face-Recognition-Using-Python

# Overview
This project implements real-time face detection and recognition using OpenCV and Python. It uses Haar cascade classifiers to detect faces from a live webcam feed. The project can be extended to include deep learning-based facial recognition using models like FaceNet, dlib, or OpenCV's LBPH (Local Binary Patterns Histograms) recognizer.

Key Features:
- Live face detection using OpenCV
- Uses Haar cascades for face detection
- Can be extended to recognize known faces
- Real-time processing for multiple faces

# Objectives
The goal of this project is to:
- Develop a real-time face detection system using OpenCV and Python.
- Implement Haar cascade classifiers for facial feature detection.
- Optimize the model for efficiency and accuracy in real-time applications.
- Extend the project to include face recognition using deep learning.
- Provide a modular and scalable codebase for further enhancements.

# Dataset
Haar Cascade Classifier XML File:
- Uses OpenCV’s pre-trained haarcascade_frontalface_default.xml model.
- This dataset consists of pre-trained classifiers for detecting human faces.
- Link to XML file: [Haarcascade GitHub](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml)

# Code
```python
import cv2

face_cap = cv2.CascadeClassifier("C:/Users/aakas/Downloads/haarcascade_frontalface_default.xml")
video_cap = cv2.VideoCapture(0)

while True:
    ret, video_data = video_cap.read()
    if not ret:
        break  # Exit loop if the video frame is not captured properly

    color = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)
    faces = face_cap.detectMultiScale(
        color,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (a, b, w, h) in faces:
        cv2.rectangle(video_data, (a, b), (a + w, b + h), (0, 255, 0), 2)  # Fixed

    cv2.imshow("Video Live", video_data)

    if cv2.waitKey(10) == ord("x"):
        break

video_cap.release()
cv2.destroyAllWindows()  # Ensure all windows are closed
```

# Findings
- The Haar cascade classifier successfully detects faces in real-time with good accuracy.
- Performance is fast and efficient but sometimes detects false positives in certain lighting conditions.
- The model works well for frontal face detection but struggles with side profiles.
- Performance can be improved using deep learning models like FaceNet or OpenCV's DNN module.


# Conclusion
This project demonstrates a real-time face detection system using OpenCV and Haar cascades. While Haar cascades provide a quick and lightweight solution for face detection, they lack robustness in detecting faces at different angles and lighting conditions. The next step is to implement a deep learning-based recognition system to enhance accuracy and reliability.


# Future Work
- Face Recognition using Deep Learning: Implement models like FaceNet, Dlib, or OpenCV LBPH for face recognition.
- Improve Accuracy with Pretrained CNNs: Use models such as VGG-Face, ResNet, or EfficientNet.
- Train a Custom Dataset: Collect and train a dataset for face authentication or attendance systems.
- Web & Mobile Integration: Deploy the model as a web app using Flask or FastAPI and integrate with React Native for mobile.
- Enhance Multi-Face Detection: Optimize the detection pipeline to handle multiple faces in a single frame.



















