#  Smart Traffic Intersection Control (AI-Powered Emergency Routing)

> An automated, real-time traffic management system designed to detect approaching emergency vehicles, clear intersections dynamically, and establish priority "Green Corridors" to save critical response time.

---

---

##  About the Project
Traditional traffic light systems operate on static timers, often causing ambulances and emergency vehicles to get trapped behind gridlocked traffic. 

This project solves that problem by building an end-to-end smart intersection prototype. Using a live camera feed, an **Artificial Intelligence (YOLOv8)** core detects an ambulance, communicates with a **Python/Flask backend**, and instantly triggers an **Arduino microcontroller** to override normal traffic lights. It creates a secure **Dual-Green Corridor** ahead of the ambulance while locking cross-traffic on red.

---

##  System Architecture
The project is split into four core layers working in real-time sync:
1. **Frontend Dashboard (HTML/JS):** A web-based interface allowing dispatchers to select target routes and monitor real-time system status.
2. **AI & Backend Engine (Python / Flask / OpenCV):** Processes video streams, executes machine learning inference, tracks vehicles via BotSORT, and handles safety timeout timers.
3. **Serial Bridge (USB / 9600 Baud):** Transmits lightweight 1-byte command strings (`'1'`, `'2'`, `'3'`, `'4'`, `'N'`) from the laptop to the hardware.
4. **Hardware Layer (Arduino & LEDs):** A physical 4-way intersection model controlled by an Arduino microcontroller executing fail-safe logic.

---

##  Machine Learning & Dataset Creation

### 1. Custom Dataset & Training
Instead of using a generic pre-trained model (which detects 80+ everyday objects and gets confused by buses or white vans), we built a specialized model:
* **Annotation Platform:** We utilized **Roboflow** to curate and annotate images.
* **Dataset Size:** Built using **337 specialized images** of ambulances from various angles, lighting conditions, and distances.
* **Model Architecture:** Trained using **Ultralytics YOLOv8**. 
* **Outcome:** The custom model is highly optimized for recognizing ambulance structures and flashing emergency lightbars, ensuring near-zero false positives.

### 2. Advanced Tracking with BotSORT
To prevent the AI bounding box from flickering or dropping when an ambulance passes behind a lamppost or tree, we integrated **BotSORT** (`tracker="botsort.yaml"`). This acts as digital persistence, maintaining a locked ID track on the vehicle throughout its passage through the frame.

---

##  Tech Stack
* **AI / Computer Vision:** Python, Ultralytics YOLOv8, OpenCV, BotSORT
* **Backend Web Framework:** Flask, REST APIs
* **Hardware Microcontroller:** Arduino (C++)
* **Communication Protocol:** PySerial (UART over USB)
* **Frontend:** HTML5, CSS3, JavaScript

---

##  Repository Structure
```text
Smart-Traffic-Intersection-Control
│
├── app.py                     # Main Python Flask server & AI video processing loop
├── yolov8_ambulance.pt        # Custom-trained YOLOv8 model weights
├── traffic.mp4                # Simulation video feed for testing
├── requirements.txt           # Python dependency manifest
│
├── hardware/                  
│   └── intersection.ino       # Arduino C++ script for LED control
│
└── templates/
    └── index.html             # Web dashboard UI
