import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from flask import Flask, render_template, Response, jsonify

app = Flask(__name__)
print("Loading AI Model...")
model = YOLO('yolov8n.pt') 

vehicle_blue_history = defaultdict(list)
vehicle_classes = [2, 3, 5, 7]

# Global variable to hold the current status and the new timer
system_status = {"ambulance_detected": False, "emergency_timer": 0}

def generate_frames():
    global system_status 
    cap = cv2.VideoCapture('traffic.mp4')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = cv2.resize(frame, (800, 480))
        results = model.track(frame, persist=True, classes=vehicle_classes, verbose=False)

        for r in results:
            boxes = r.boxes
            if boxes.id is None: continue
                
            for box, track_id in zip(boxes, boxes.id):
                conf = float(box.conf[0]) 
                t_id = int(track_id) 

                if conf > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    vehicle_roi = frame[y1:y2, x1:x2]
                    if vehicle_roi.size == 0 or (x2 - x1) < 50 or (y2 - y1) < 50: continue

                    h, w, _ = vehicle_roi.shape
                    body_roi = vehicle_roi[int(h * 0.3):int(h * 0.9), 0:w]
                    hsv_body = cv2.cvtColor(body_roi, cv2.COLOR_BGR2HSV)
                    mask_white = cv2.inRange(hsv_body, np.array([0, 0, 120]), np.array([180, 60, 255]))
                    white_pixels = cv2.countNonZero(mask_white)
                    body_area = body_roi.shape[0] * body_roi.shape[1]
                    is_white_vehicle = body_area > 0 and (white_pixels / body_area) > 0.20

                    blue_density = 0
                    if is_white_vehicle:
                        top_roof_roi = vehicle_roi[0:int(h * 0.4), 0:w]
                        hsv_roof = cv2.cvtColor(top_roof_roi, cv2.COLOR_BGR2HSV)
                        mask_blue = cv2.inRange(hsv_roof, np.array([90, 120, 120]), np.array([140, 255, 255]))
                        blue_pixels = cv2.countNonZero(mask_blue)
                        roof_area = top_roof_roi.shape[0] * top_roof_roi.shape[1]
                        if roof_area > 0: blue_density = blue_pixels / roof_area

                    vehicle_blue_history[t_id].append(blue_density)
                    if len(vehicle_blue_history[t_id]) > 15: vehicle_blue_history[t_id].pop(0)

                    is_ambulance = False
                    if is_white_vehicle and len(vehicle_blue_history[t_id]) >= 5:
                        if max(vehicle_blue_history[t_id]) > 0.015 and min(vehicle_blue_history[t_id]) < 0.005:
                            is_ambulance = True

                    if is_ambulance:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                        cv2.putText(frame, "EMERGENCY!", (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)
                        
                        # ---> NEW: Set the Latch Timer! (approx 50 frames of guaranteed green light)
                        system_status["emergency_timer"] = 50 
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

        # ---> NEW: Evaluate the Latch Timer
        if system_status.get("emergency_timer", 0) > 0:
            system_status["ambulance_detected"] = True
            system_status["emergency_timer"] -= 1 # Countdown!
        else:
            system_status["ambulance_detected"] = False

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index(): return render_template('index.html')

@app.route('/video_feed')
def video_feed(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status(): return jsonify({"ambulance_detected": system_status["ambulance_detected"]})

if __name__ == "__main__":
    app.run(debug=True, port=5000)