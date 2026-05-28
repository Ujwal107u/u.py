import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

# 1. Load the pre-trained YOLOv8 AI
print("Loading AI Model...")
model = YOLO('yolov8n.pt') 

# 2. Open your downloaded video
video_path = 'traffic.mp4'
cap = cv2.VideoCapture(video_path)

# YOLO COCO Dictionary: 2=car, 3=motorcycle, 5=bus, 7=truck
vehicle_classes = [2, 3, 5, 7]

# NEW: A dictionary to act as the AI's "memory" for flickering lights
# It will store the blue light history for every vehicle ID
vehicle_blue_history = defaultdict(list)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or video not found.")
        break

    # Make frame smaller to SPEED UP the video processing
    frame = cv2.resize(frame, (800, 480))

    # 3. Ask YOLO to TRACK objects (This gives each car a persistent ID number)
    # verbose=False stops it from printing to the terminal, which drastically speeds up the video
    results = model.track(frame, persist=True, classes=vehicle_classes, verbose=False)

    for r in results:
        boxes = r.boxes
        
        # If YOLO hasn't assigned IDs yet (happens on frame 1), skip
        if boxes.id is None:
            continue
            
        # Loop through the bounding boxes AND their unique tracking IDs
        for box, track_id in zip(boxes, boxes.id):
            
            conf = float(box.conf[0]) 
            t_id = int(track_id) # The unique ID for this specific car

            # 4. Only proceed if the AI is highly confident
            if conf > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                vehicle_roi = frame[y1:y2, x1:x2]
                
                # Skip if the box is tiny
                if vehicle_roi.size == 0 or (x2 - x1) < 50 or (y2 - y1) < 50:
                    continue

                h, w, _ = vehicle_roi.shape

                # ==========================================
                # 5A. THE WHITE BODY CHECK
                # ==========================================
                body_roi = vehicle_roi[int(h * 0.3):int(h * 0.9), 0:w]
                hsv_body = cv2.cvtColor(body_roi, cv2.COLOR_BGR2HSV)
                
                # Broad white detection (low saturation, high brightness)
                lower_white = np.array([0, 0, 120])
                upper_white = np.array([180, 60, 255])
                mask_white = cv2.inRange(hsv_body, lower_white, upper_white)
                
                white_pixels = cv2.countNonZero(mask_white)
                body_area = body_roi.shape[0] * body_roi.shape[1]
                
                is_white_vehicle = False
                if body_area > 0 and (white_pixels / body_area) > 0.20: 
                    is_white_vehicle = True

                # ==========================================
                # 5B. THE BLUE SIREN TRACKER
                # ==========================================
                blue_density = 0
                
                if is_white_vehicle:
                    top_roof_roi = vehicle_roi[0:int(h * 0.4), 0:w]
                    hsv_roof = cv2.cvtColor(top_roof_roi, cv2.COLOR_BGR2HSV)

                    # Strict Blue (Glowing)
                    lower_blue = np.array([90, 120, 120]) 
                    upper_blue = np.array([140, 255, 255])
                    mask_blue = cv2.inRange(hsv_roof, lower_blue, upper_blue)

                    blue_pixels = cv2.countNonZero(mask_blue)
                    roof_area = top_roof_roi.shape[0] * top_roof_roi.shape[1]
                    
                    if roof_area > 0:
                        blue_density = blue_pixels / roof_area

                # Save this frame's blue density to the vehicle's memory
                vehicle_blue_history[t_id].append(blue_density)
                
                # Keep only the last 15 frames of memory (approx 0.5 seconds of time)
                if len(vehicle_blue_history[t_id]) > 15:
                    vehicle_blue_history[t_id].pop(0)

                # ==========================================
                # 6. THE FLICKER DECISION ENGINE
                # ==========================================
                is_ambulance = False
                
                # We need at least 5 frames of history to prove it's flickering
                if is_white_vehicle and len(vehicle_blue_history[t_id]) >= 5:
                    # Get the brightest moment and darkest moment from the last 15 frames
                    max_blue = max(vehicle_blue_history[t_id])
                    min_blue = min(vehicle_blue_history[t_id])
                    
                    # FLICKER MATH: 
                    # If it gets very bright (max > 0.015) AND it goes dark (min < 0.005)
                    # That means the light is physically turning on and off!
                    if max_blue > 0.015 and min_blue < 0.005:
                        is_ambulance = True

                # 7. Draw the Results
                if is_ambulance:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                    cv2.putText(frame, "EMERGENCY: AMBULANCE!", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
                    # Show the ID number so you can see the tracker working!
                    cv2.putText(frame, f"ID: {t_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    # Display the live feed
    cv2.imshow('Smart City Vision AI', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()