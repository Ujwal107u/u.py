import cv2
import numpy as np
from ultralytics import YOLO
from flask import Flask, render_template, Response, jsonify, request
import serial
import time
import os

# --- 1. HARDWARE INITIALIZATION ---
try:
    arduino = serial.Serial(port='COM3', baudrate=9600, timeout=.1)
    time.sleep(2)
    arduino_connected = True
    print("✅ SUCCESS: Arduino Connected on COM3!")
except Exception as e:
    arduino_connected = False
    print("⚠️ WARNING: Arduino not found. Running in Laptop Simulation Mode.")

app = Flask(__name__)

# --- 2. MODEL INITIALIZATION ---
MODEL_PATH = 'yolov8_ambulance.pt'
if os.path.exists(MODEL_PATH):
    model = YOLO(MODEL_PATH)
    print(f"✅ SUCCESS: Custom Model '{MODEL_PATH}' Loaded!")
else:
    model = YOLO('yolov8n.pt')
    print(f"⚠️ WARNING: '{MODEL_PATH}' missing. Using default model.")

# --- 3. SYSTEM STATE TRACKER ---
system_status = {
    "ambulance_detected": False, 
    "clearance_end_time": 0.0, 
    "target_route": None,
    "route_selected": False
}

# --- 4. CORE VIDEO PROCESSING ---
def generate_frames():
    global system_status
    cap = cv2.VideoCapture('traffic.mp4')
    
    # Standby black screen
    black_frame = np.zeros((480, 800, 3), dtype=np.uint8)

    while True:
        # A. STANDBY SCREEN
        if not system_status["route_selected"]:
            display_frame = black_frame.copy()
            cv2.putText(display_frame, "AWAITING ROUTE SELECTION...", (180, 240), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
            ret_enc, buffer = cv2.imencode('.jpg', display_frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.1)
            continue

        ret, frame = cap.read()
        
        # B. END OF VIDEO & TIMER PAUSE
        if not ret:
            if time.time() < system_status["clearance_end_time"]:
                display_frame = black_frame.copy()
                cv2.putText(display_frame, ".................", (300, 240), 
                            cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
                ret_enc, buffer = cv2.imencode('.jpg', display_frame)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.1)
                continue
            else:
                print("🏁 Sequence Concluded: Reverting to Standby Mode...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                system_status["route_selected"] = False
                system_status["target_route"] = None
                system_status["ambulance_detected"] = False
                continue
        
        # C. AI INFERENCE WITH ADVANCED TRACKING
        frame = cv2.resize(frame, (800, 480))
        
        # Using BotSORT for high-persistence tracking and conf=0.35 to catch awkward angles
        results = model.track(frame, persist=True, conf=0.35, tracker="botsort.yaml", verbose=False)
        
        ambulance_currently_visible = False

        for r in results:
            boxes = r.boxes
            if boxes.id is None: 
                continue
                
            for box, track_id in zip(boxes, boxes.id):
                ambulance_currently_visible = True
                
                # Extract coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                t_id = int(track_id)
                
                # Draw the priority bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                cv2.putText(frame, f"AMBULANCE LOCKED! (ID: {t_id})", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 2)
                
                # Push the 15-second timer forward
                system_status["clearance_end_time"] = time.time() + 15.0

        # D. HARDWARE SIGNALING LOGIC
        previous_state = system_status.get("ambulance_detected", False)
        
        if time.time() < system_status["clearance_end_time"]:
            system_status["ambulance_detected"] = True
        else:
            system_status["ambulance_detected"] = False

        current_state = system_status["ambulance_detected"]
        target = system_status["target_route"]

        # Hardware Triggers
        if current_state == True and previous_state == False:
            print(f"🚨 OVERRIDE: Sending Route '{target}' to Arduino...")
            if arduino_connected: 
                arduino.write(target.encode())
            
        elif current_state == False and previous_state == True:
            print("✅ SYSTEM NORMAL: Sending 'N' to Arduino...")
            if arduino_connected: 
                arduino.write(b'N')

        # E. ONSCREEN UI OVERLAYS
        if current_state:
            time_left = max(0, int(system_status["clearance_end_time"] - time.time()))
            if ambulance_currently_visible:
                cv2.putText(frame, "VEHICLE IN CAMERA VIEW: OVERRIDE ACTIVE", (20, 40),
                            cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 0), 2)
            else:
                cv2.putText(frame, f"CLEARANCE COUNTDOWN: {time_left}s", (20, 40),
                            cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 0, 255), 2)

        ret_enc, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# --- 5. FLASK WEB ROUTES ---
@app.route('/')
def index(): 
    return render_template('index.html')

@app.route('/video_feed')
def video_feed(): 
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status(): 
    return jsonify(system_status)

@app.route('/set_target_route', methods=['POST'])
def set_target_route():
    data = request.json
    if data and 'route' in data:
        system_status["target_route"] = str(data['route'])
        system_status["route_selected"] = True 
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "No route provided"}), 400

if __name__ == "__main__":
    print("🚀 Firing up Command Center on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)