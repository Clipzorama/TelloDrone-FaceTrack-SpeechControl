# Tello Drone: Face Tracking & Speech Command Control 🚁

Welcome to the Tello Drone Face Tracking & Speech Recognition Project! This repository contains code and resources to enable your Tello drone to detect faces, track objects, and respond to voice commands. A collaboration with Landon, this project was developed at International Hellenic University, Thessaloniki, Greece.

## 🛠 Features

**Face Tracking & Detection:** Real-time detection and tracking using OpenCV's Haar cascades and the MobileNet SSD model.
**Voice Command Integration:** Control your drone through natural language commands powered by the Vosk speech recognition library.
**Object Detection:** Recognize and classify objects in real-time.
**Keyboard Controls:** Manual drone control using Pygame for keypress handling.

## 📸 Preview


![View1](./design/TelloExperience.jpg)
![View2](./design/TelloExperience2.jpg)
![View3](./design/TelloExperience3.jpg)

## 🚀 To Start Off

Ensure you have the following installed:

- Python 3.7+
- OpenCV (cv2 --> Video Capture)
- Pygame (Key manipulation)
- Djitellopy (Tello Correlation)
- Vosk (Speech Recognition)

## Installation 🌟

```bash
git clone https://github.com/yourusername/TelloDrone-FaceTrack-SpeechControl.git
cd TelloDrone-FaceTrack-SpeechControl
pip install -r requirements.txt

```

## 🧩 Sneak Peek of the Code

### Face Detection and Tracking

```python
def findFace(img):
    faceCascade = cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.2, 8)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return img, faces
```

### Speech Command Example

```python
if "take off" in recognized_speech:
    chrision_drone.takeoff()
elif "land" in recognized_speech:
    chrision_drone.land()
```


## Challenges Overcome
- **Synchronization of Drone Streams:** Real-time image capture and processing.
- **Voice Command Accuracy:** Optimized recognition with Vosk models.
- **Adding the keys to Drone:** Implementing the keys so the drone can be controlled from my laptop.

## 📝 Acknowledgments

Special thanks to **Landon** for collaborating on this project and to the International Hellenic University for providing the opportunity to develop it in Thessaloniki, Greece.
