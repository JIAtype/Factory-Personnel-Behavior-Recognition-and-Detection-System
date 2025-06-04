好的，将这个复杂的Tkinter应用转换为网页形式，并加入头盔检测，是一个涉及多个步骤的重构过程。

我会使用 **Flask** 作为Web框架，因为它轻量且易于与CV处理集成。视频流将通过MJPEG发送到浏览器。

**核心改动思路：**

1.  **Web框架 (Flask):**
    *   创建一个Flask应用。
    *   定义路由 (routes) 来提供HTML页面、视频流、控制命令（开始/停止）、状态更新。
2.  **视频流:**
    *   `detection_loop` 生成的帧将不再放入Tkinter队列，而是编码为JPEG并通过HTTP流式传输。
3.  **UI (HTML/CSS/JavaScript):**
    *   创建一个HTML页面来显示视频流、统计数据、警报和控制按钮。
    *   使用JavaScript与后端Flask API交互（例如，发送开始/停止命令，定期获取统计数据和警报）。
4.  **`BehaviorDetectionSystem` 类重构:**
    *   移除所有Tkinter相关的GUI代码 (`setup_gui`, `update_gui`, `messagebox`, `tk.Label`, `tk.Button`等)。
    *   状态（如`running`, `people_count`, `alerts`）将由类实例维护，并通过Flask API暴露给前端。
    *   警报将存储在一个列表中，前端可以轮询获取。
5.  **头盔检测:**
    *   需要一个额外的YOLO模型（或一个能同时检测人和头盔的模型）。为简单起见，我们将使用一个通用的物体检测模型 (如 `yolov8n.pt`)，并假设它能检测到人和头盔（如果COCO模型不包含头盔，我们会说明这一点，并演示逻辑）。
    *   在 `detection_loop` 中，运行人和姿态检测后，再运行物体检测模型。
    *   关联检测到的人和头盔（例如，通过头盔是否在人的头部区域）。
    *   在 `PersonTracker` 或主类中添加头盔状态。
    *   更新标注和警报逻辑。
6.  **线程管理:**
    *   Flask本身可以在多线程模式下运行。检测循环将在一个后台线程中运行，就像之前一样。

**步骤概览:**

1.  安装 Flask: `pip install Flask ultralytics opencv-python numpy Pillow`
2.  创建 `app.py` (Flask应用主文件)。
3.  创建 `templates` 文件夹并在其中放置 `index.html`。
4.  修改 `BehaviorDetectionSystem` 和 `PersonTracker` 类。
5.  实现头盔检测逻辑。

下面是修改后的代码结构。这将是一个较大的改动，我会提供关键部分和说明。

**1. `app.py` (Flask 应用)**

```python
from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import time
import threading
from datetime import datetime, timedelta
import os
import json
from collections import defaultdict, deque
import queue # Keep queue for alerts if needed, or switch to simple list
import math
from PIL import Image # Will be used by Flask if sending PIL images, but cv2.imencode is better for MJPEG

try:
    from ultralytics import YOLO
except ImportError:
    print("Please install ultralytics: pip install ultralytics")
    exit(1)

# --- PersonTracker Class (largely unchanged, but we'll add helmet status) ---
class PersonTracker:
    def __init__(self, person_id, keypoints):
        self.id = person_id
        self.keypoints_history = deque(maxlen=30)
        self.last_movement_time = time.time()
        self.is_fallen = False
        self.is_stationary = False
        self.fall_alert_sent = False
        self.stationary_alert_sent = False
        self.no_helmet_alert_sent = False # New for helmet
        self.last_alert_time = 0
        self.keypoints_history.append(keypoints)
        self.has_helmet = True # Assume has helmet initially, detection will update this
        self.bbox = None # Store bounding box for helmet association

    def update(self, keypoints, bbox=None): # Add bbox
        self.keypoints_history.append(keypoints)
        if bbox is not None:
            self.bbox = bbox # Update person's bounding box

        if len(self.keypoints_history) >= 2:
            movement = self.calculate_movement()
            if movement > 10:
                self.last_movement_time = time.time()
                if self.is_stationary: self.stationary_alert_sent = False
                if self.is_fallen: self.fall_alert_sent = False
                # if not self.has_helmet: self.no_helmet_alert_sent = False # Reset if person moves
                self.is_stationary = False
                self.is_fallen = False
            else:
                if time.time() - self.last_movement_time > 180: # 3 minutes
                    self.is_stationary = True

        fall_detected = self.detect_fall()
        if fall_detected and not self.is_fallen:
            self.is_fallen = True
            self.fall_alert_sent = False
        elif not fall_detected:
            self.is_fallen = False

    def calculate_movement(self):
        # ... (same as original)
        if len(self.keypoints_history) < 2: return 0
        current = self.keypoints_history[-1]
        previous = self.keypoints_history[-2]
        total_movement = 0
        valid_points = 0
        min_len = min(len(current), len(previous))
        for i in range(min_len):
            if (len(current[i]) > 2 and len(previous[i]) > 2 and
                current[i][2] > 0.5 and previous[i][2] > 0.5):
                dx = current[i][0] - previous[i][0]
                dy = current[i][1] - previous[i][1]
                movement = math.sqrt(dx*dx + dy*dy)
                total_movement += movement
                valid_points += 1
        return total_movement / max(valid_points, 1)

    def detect_fall(self):
        # ... (same as original)
        if not self.keypoints_history: return False
        keypoints = self.keypoints_history[-1]
        nose = keypoints[0] if len(keypoints) > 0 and keypoints[0][2] > 0.5 else None
        left_shoulder = keypoints[5] if len(keypoints) > 5 and keypoints[5][2] > 0.5 else None
        right_shoulder = keypoints[6] if len(keypoints) > 6 and keypoints[6][2] > 0.5 else None
        left_hip = keypoints[11] if len(keypoints) > 11 and keypoints[11][2] > 0.5 else None
        right_hip = keypoints[12] if len(keypoints) > 12 and keypoints[12][2] > 0.5 else None
        if (left_shoulder and right_shoulder and left_hip and right_hip):
            shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
            hip_center_y = (left_hip[1] + right_hip[1]) / 2
            if hip_center_y < shoulder_center_y + 10 : # Added a small tolerance
                 # Check aspect ratio if hips are not significantly lower
                body_width = abs(left_shoulder[0] - right_shoulder[0])
                body_height = abs(shoulder_center_y - hip_center_y)
                if body_height > 0:
                    aspect_ratio = body_width / body_height
                    if aspect_ratio > 1.8: # Adjusted threshold
                        return True
                # If hips are truly above shoulders (more lenient check for fall)
                if hip_center_y < shoulder_center_y - (body_height * 0.2 if body_height > 0 else 5): # Hips significantly above shoulders
                    return True
        return False

    def get_head_region(self):
        """Estimates head region based on keypoints."""
        if not self.keypoints_history:
            return None
        
        keypoints = self.keypoints_history[-1]
        # Keypoints: 0:Nose, 1:L_Eye, 2:R_Eye, 3:L_Ear, 4:R_Ear, 5:L_Shoulder, 6:R_Shoulder
        
        visible_head_kpts = []
        indices = [0, 1, 2, 3, 4] # Nose, Eyes, Ears
        for i in indices:
            if len(keypoints) > i and keypoints[i][2] > 0.3: # Confidence for head keypoints
                visible_head_kpts.append(keypoints[i][:2])
        
        if not visible_head_kpts:
            # Fallback: use area above shoulders if head keypoints are not visible
            if self.bbox:
                x1, y1, x2, y2 = self.bbox[:4]
                # Estimate head as top 25% of the person's bounding box
                head_y2 = y1 + (y2 - y1) * 0.25
                return (x1, y1, x2, head_y2)
            return None

        visible_head_kpts = np.array(visible_head_kpts)
        min_x, min_y = np.min(visible_head_kpts, axis=0)
        max_x, max_y = np.max(visible_head_kpts, axis=0)

        # Expand the box a bit to be more robust
        width = max_x - min_x
        height = max_y - min_y
        
        # Define head box based on keypoints
        # Make it slightly larger and extend upwards
        padding_w = width * 0.3
        padding_h_up = height * 0.8 # More padding upwards
        padding_h_down = height * 0.2

        hx1 = int(min_x - padding_w)
        hy1 = int(min_y - padding_h_up)
        hx2 = int(max_x + padding_w)
        hy2 = int(max_y + padding_h_down)
        
        return (hx1, hy1, hx2, hy2)


# --- BehaviorDetectionSystem Class (Refactored for Flask) ---
class BehaviorDetectionSystem:
    def __init__(self, video_source="test_video.mp4"):
        self.video_source = video_source
        self.pose_model = None # For pose estimation
        self.object_model = None # For helmet/person detection
        self.cap = None
        self.trackers = {}
        self.next_id = 0
        self.running = False
        self.latest_frame = None # Store the latest processed frame for streaming
        self.detection_lock = threading.Lock() # To protect shared resources like latest_frame

        self.alerts = deque(maxlen=50) # Store recent alerts as strings
        self.stats = {
            "people_count": 0,
            "fallen_count": 0,
            "stationary_count": 0,
            "no_helmet_count": 0 # New
        }

        self.is_video_file = self.check_if_video_file(video_source)
        self.alert_cooldown = {} # To prevent spamming alerts for the same person/event

        # Directories for recordings (optional for web, but kept for consistency)
        os.makedirs("alerts_data/recordings", exist_ok=True) 
        os.makedirs("alerts_data/json", exist_ok=True)
        self.recording_buffer = deque(maxlen=300) # 1 min at 5 fps for alert recording
        self.is_recording_alert = False
        self.alert_active_for_recording = False


    def check_if_video_file(self, source):
        try: # if source is int (webcam ID)
            int_source = int(source)
            return False # Treat webcam as a stream
        except ValueError:
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
            return any(source.lower().endswith(ext) for ext in video_extensions)

    def load_models(self):
        try:
            self.pose_model = YOLO('yolov8x-pose.pt') # Or yolov8n-pose.pt for faster inference
            # For helmet detection, you'd ideally use a model trained specifically for helmets.
            # We'll use a general object detector and look for 'person'.
            # If you have a helmet model (e.g., 'helmet_model.pt'), use it here.
            # For demonstration, we'll use yolov8n.pt and simulate helmet logic if 'helmet' class isn't present.
            self.object_model = YOLO('yolov8n.pt') # General object detection
            self.object_model_classes = self.object_model.names
            # Check if 'helmet' class exists in the model, otherwise we'll need to adjust
            self.helmet_class_id = None
            for k, v in self.object_model_classes.items():
                if v.lower() == 'helmet' or v.lower() == 'safety helmet': # common names
                    self.helmet_class_id = k
                    break
            if self.helmet_class_id is None:
                print("[WARNING] 'helmet' class not found in object_model. Helmet detection will be illustrative.")
                # You might map another class for testing, e.g., 'sports ball' if you want to see boxes
                # for cls_id, cls_name in self.object_model_classes.items():
                #    if cls_name == 'sports ball': self.helmet_class_id = cls_id; break

            self.person_class_id = None
            for k, v in self.object_model_classes.items():
                if v.lower() == 'person':
                    self.person_class_id = k
                    break
            if self.person_class_id is None:
                print("[ERROR] 'person' class not found in object_model. This is crucial.")
                return False

            print("[INFO] Models loaded successfully.")
            return True
        except Exception as e:
            self.add_log_message(f"[ERROR] Failed to load models: {str(e)}")
            return False

    def start_detection(self):
        if self.running:
            self.add_log_message("[INFO] Detection already running.")
            return
        
        if not self.pose_model or not self.object_model:
            if not self.load_models():
                return

        self.add_log_message(f"[INFO] Attempting to open video source: {self.video_source}")
        try:
            self.cap = cv2.VideoCapture(int(self.video_source) if self.video_source.isdigit() else self.video_source)
        except Exception as e:
             self.add_log_message(f"[ERROR] OpenCV Error: {e}")
             self.cap = None

        if not self.cap or not self.cap.isOpened():
            self.add_log_message(f"[ERROR] Failed to open video source: {self.video_source}")
            self.running = False # Ensure it's set to false
            return

        self.is_video_file = self.check_if_video_file(self.video_source)
        source_type = "Video File" if self.is_video_file else ("Webcam" if self.video_source.isdigit() else "RTSP Stream")
        self.add_log_message(f"[INFO] Opened {source_type}: {self.video_source}")
        
        self.running = True
        self.trackers.clear()
        self.next_id = 0
        
        self.detection_thread = threading.Thread(target=self.detection_loop)
        self.detection_thread.daemon = True
        self.detection_thread.start()
        self.add_log_message("[INFO] Detection started.")

    def stop_detection(self):
        self.running = False
        if hasattr(self, 'detection_thread') and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=2) # Wait for thread to finish
        
        with self.detection_lock:
            if self.cap:
                self.cap.release()
                self.cap = None
            self.latest_frame = None # Clear the frame
        
        self.trackers.clear()
        # Reset stats but keep alerts log
        self.stats = {k: 0 for k in self.stats}
        self.add_log_message("[INFO] Detection stopped.")


    def iou(self, boxA, boxB):
        # (x1, y1, x2, y2)
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        
        if boxAArea + boxBArea - interArea <= 0: return 0.0
        return interArea / float(boxAArea + boxBArea - interArea)

    def detection_loop(self):
        while self.running:
            if not self.cap or not self.cap.isOpened():
                self.add_log_message("[WARNING] Video capture is not open. Trying to reconnect...")
                time.sleep(5)
                self.cap = cv2.VideoCapture(int(self.video_source) if self.video_source.isdigit() else self.video_source)
                if not self.cap or not self.cap.isOpened():
                    self.add_log_message("[ERROR] Reconnect failed. Stopping detection.")
                    self.running = False
                    break
                else:
                    self.add_log_message("[INFO] Reconnected to video source.")

            ret, frame = self.cap.read()
            if not ret:
                if self.is_video_file:
                    self.add_log_message("[INFO] Video file ended. Looping.")
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else: # Stream error
                    self.add_log_message("[WARNING] Frame not received from stream. Attempting to maintain connection.")
                    time.sleep(0.1) # Brief pause before trying to read again
                    continue # Try to read next frame

            frame_copy_for_buffer = frame.copy()
            self.recording_buffer.append((time.time(), frame_copy_for_buffer))

            # 1. Pose Detection
            pose_results = self.pose_model(frame, verbose=False)
            
            current_person_detections_for_tracking = []
            person_keypoints_map = {} # Store keypoints for persons detected by pose model

            for r in pose_results:
                if r.keypoints is not None and r.boxes is not None:
                    keypoints_data = r.keypoints.data.cpu().numpy()
                    boxes_data = r.boxes.data.cpu().numpy()
                    
                    for i, (kpts, box_coords) in enumerate(zip(keypoints_data, boxes_data)):
                        # box_coords are [x1, y1, x2, y2, conf, cls]
                        # We are interested in persons from pose model, cls should be 0 (person)
                        # but pose model might not always provide class, rely on keypoints presence
                        if box_coords[4] > 0.5 : # Confidence for person detection from pose model
                            current_person_detections_for_tracking.append({
                                'keypoints': kpts,
                                'box': box_coords, # This is person box from pose
                                'id': None
                            })
                            # For associating with object model persons later if needed
                            # but primarily we use this for tracking and fall/stationary
                            # We'll map this to tracker IDs soon.

            # Update trackers based on pose detections
            self.update_trackers(current_person_detections_for_tracking)

            # 2. Object Detection (for Helmets and confirming Persons)
            object_results = self.object_model(frame, verbose=False)
            detected_helmets_boxes = [] # Store [x1,y1,x2,y2] for helmets
            
            # Reset helmet status for all trackers before checking current frame
            for tracker in self.trackers.values():
                tracker.has_helmet = False # Assume no helmet until confirmed

            if self.helmet_class_id is not None: # Only if helmet class is defined
                for r_obj in object_results:
                    for box in r_obj.boxes:
                        if box.cls == self.helmet_class_id and box.conf > 0.4: # Confidence for helmet
                            detected_helmets_boxes.append(box.xyxy[0].cpu().numpy().astype(int))
            
            # 3. Associate Helmets with Tracked Persons
            active_tracker_ids_this_frame = []
            for tracker_id, tracker in self.trackers.items():
                active_tracker_ids_this_frame.append(tracker_id)
                if not tracker.keypoints_history: continue

                head_region = tracker.get_head_region()
                if head_region:
                    tracker.has_helmet = False # Assume no helmet for this person
                    for helmet_box in detected_helmets_boxes:
                        if self.iou(head_region, helmet_box) > 0.1: # IoU threshold for helmet on head
                            tracker.has_helmet = True
                            break # Found a helmet for this person

            # Remove trackers for persons not detected in the current frame for a while
            # (This simple tracker doesn't have explicit "disappeared" logic, relies on update_trackers)

            # 4. Draw annotations and check alerts
            annotated_frame = self.draw_annotations(frame.copy()) # Draw on a copy

            with self.detection_lock:
                self.latest_frame = annotated_frame.copy()
            
            if self.is_video_file:
                time.sleep(1 / 30) # Approx 30 FPS for video files
            else:
                time.sleep(0.01) # Minimal delay for streams to process as fast as possible


    def update_trackers(self, detections):
        # (Largely same as original, ensure 'box' is used for distance if keypoints are sparse)
        used_detections = set()
        
        for tracker_id, tracker in list(self.trackers.items()):
            best_match = None
            best_distance = float('inf')
            
            for i, detection in enumerate(detections):
                if i in used_detections: continue
                
                distance = self.calculate_detection_distance(tracker, detection)
                if distance < best_distance and distance < 150: # Distance threshold (pixels)
                    best_distance = distance
                    best_match = i
            
            if best_match is not None:
                detection = detections[best_match]
                tracker.update(detection['keypoints'], detection['box']) # Pass box too
                detection['id'] = tracker_id
                used_detections.add(best_match)
            else:
                # Consider a "disappeared" counter before removing immediately
                # For now, simple removal if no match
                if time.time() - tracker.last_movement_time > 10: # If no movement and no match for 10s
                     del self.trackers[tracker_id]

        for i, detection in enumerate(detections):
            if i not in used_detections:
                new_tracker = PersonTracker(self.next_id, detection['keypoints'])
                new_tracker.bbox = detection['box'] # Store initial box
                self.trackers[self.next_id] = new_tracker
                detection['id'] = self.next_id
                self.next_id = (self.next_id + 1) % 10000 # Cycle IDs to prevent very large numbers


    def calculate_detection_distance(self, tracker, detection):
        # (same as original)
        if not tracker.keypoints_history: return float('inf')
        last_keypoints = tracker.keypoints_history[-1]
        current_keypoints = detection['keypoints']
        try:
            last_valid = last_keypoints[last_keypoints[:, 2] > 0.5]
            current_valid = current_keypoints[current_keypoints[:, 2] > 0.5]
            if len(last_valid) == 0 or len(current_valid) == 0:
                 # Fallback to bounding box center if keypoints are not good
                if tracker.bbox is not None and 'box' in detection:
                    b1_center_x = (tracker.bbox[0] + tracker.bbox[2]) / 2
                    b1_center_y = (tracker.bbox[1] + tracker.bbox[3]) / 2
                    b2_center_x = (detection['box'][0] + detection['box'][2]) / 2
                    b2_center_y = (detection['box'][1] + detection['box'][3]) / 2
                    return np.linalg.norm(np.array([b1_center_x, b1_center_y]) - np.array([b2_center_x, b2_center_y]))
                return float('inf')
            
            last_center = np.mean(last_valid[:, :2], axis=0)
            current_center = np.mean(current_valid[:, :2], axis=0)
            return np.linalg.norm(last_center - current_center)
        except Exception: return float('inf')

    def draw_annotations(self, frame):
        # (Modified to include helmet status and update web stats)
        annotated_frame = frame # Work on the passed frame
        
        current_fallen_count = 0
        current_stationary_count = 0
        current_no_helmet_count = 0
        current_people_count = len(self.trackers)

        alert_triggered_this_frame = False

        for tracker_id, tracker in self.trackers.items():
            if not tracker.keypoints_history: continue
            
            keypoints = tracker.keypoints_history[-1]
            # self.draw_skeleton(annotated_frame, keypoints) # Optional: can be performance heavy

            # Use tracker's bbox if available, otherwise estimate from keypoints
            box_to_draw = tracker.bbox 
            if box_to_draw is None:
                 valid_keypoints = keypoints[keypoints[:, 2] > 0.5]
                 if len(valid_keypoints) > 0:
                    x_coords = valid_keypoints[:, 0]; y_coords = valid_keypoints[:, 1]
                    x1, y1 = int(x_coords.min()), int(y_coords.min())
                    x2, y2 = int(x_coords.max()), int(y_coords.max())
                    box_to_draw = [x1,y1,x2,y2]

            if box_to_draw is not None:
                x1, y1, x2, y2 = map(int, box_to_draw[:4])
                color = (0, 255, 0) # Green for normal
                status_texts = [f"P{tracker_id}"]

                if tracker.is_fallen:
                    color = (0, 0, 255)  # Red for fallen
                    status_texts.append("FALLEN")
                    current_fallen_count += 1
                    if not tracker.fall_alert_sent and self.can_send_alert(tracker_id, "fall"):
                        self.trigger_alert(f"FALLEN - Person {tracker_id}", tracker_id, "fall")
                        tracker.fall_alert_sent = True
                        alert_triggered_this_frame = True
                elif tracker.is_stationary:
                    color = (0, 165, 255)  # Orange for stationary
                    status_texts.append("STATIONARY")
                    current_stationary_count += 1
                    if not tracker.stationary_alert_sent and self.can_send_alert(tracker_id, "stationary"):
                        self.trigger_alert(f"STATIONARY - Person {tracker_id}", tracker_id, "stationary")
                        tracker.stationary_alert_sent = True
                        alert_triggered_this_frame = True
                
                # Helmet check
                if self.helmet_class_id is not None: # Only if helmet detection is active
                    if not tracker.has_helmet:
                        # If already red/orange, keep that color, else specific for no helmet
                        if color == (0, 255, 0): color = (255, 0, 255) # Purple for no helmet
                        status_texts.append("NO HELMET")
                        current_no_helmet_count +=1
                        if not tracker.no_helmet_alert_sent and self.can_send_alert(tracker_id, "no_helmet"):
                            self.trigger_alert(f"NO HELMET - Person {tracker_id}", tracker_id, "no_helmet")
                            tracker.no_helmet_alert_sent = True
                            alert_triggered_this_frame = True
                    else: # Has helmet
                        status_texts.append("HELMET OK")
                        tracker.no_helmet_alert_sent = False # Reset if helmet is detected again

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                # Draw head region for debugging
                # head_r = tracker.get_head_region()
                # if head_r: cv2.rectangle(annotated_frame, (head_r[0], head_r[1]), (head_r[2], head_r[3]), (255,255,0), 1)

                y_offset = y1 - 7
                for text_line in status_texts:
                    cv2.putText(annotated_frame, text_line, (x1, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                    y_offset -= 15 # Move up for next line of text
        
        # Update global stats
        self.stats["people_count"] = current_people_count
        self.stats["fallen_count"] = current_fallen_count
        self.stats["stationary_count"] = current_stationary_count
        self.stats["no_helmet_count"] = current_no_helmet_count

        # Manage alert recording state
        if alert_triggered_this_frame and not self.alert_active_for_recording:
            self.alert_active_for_recording = True # An alert condition is active
            if not self.is_recording_alert: # Start recording if not already
                self.is_recording_alert = True
                # Use first alert message for filename, or a generic one
                first_alert_msg = next((a.split("] ",1)[1] for a in reversed(self.alerts) if " - Person" in a), "alert_event")
                
                # Start recording in a new thread to avoid blocking detection
                threading.Thread(target=self.save_alert_recording, args=(first_alert_msg,)).start()

        elif not alert_triggered_this_frame and self.alert_active_for_recording:
            # If no alerts in this frame, but an alert was active, potentially stop recording
            # (Recording logic saves for a fixed duration anyway)
            self.alert_active_for_recording = False # No active alert conditions in this frame

        # Display overall stats on frame
        cv2.putText(annotated_frame, f"People: {current_people_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"Fallen: {current_fallen_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255) if current_fallen_count else (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"Stationary: {current_stationary_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255) if current_stationary_count else (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"No Helmet: {current_no_helmet_count}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255) if current_no_helmet_count else (255,255,255), 2, cv2.LINE_AA)
        
        return annotated_frame

    def draw_skeleton(self, frame, keypoints):
        # (Same as original)
        skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
        try:
            for connection in skeleton:
                kpt_a, kpt_b = connection
                if (kpt_a-1 < len(keypoints) and kpt_b-1 < len(keypoints) and kpt_a-1 >= 0 and kpt_b-1 >= 0):
                    x1, y1, c1 = keypoints[kpt_a-1]; x2, y2, c2 = keypoints[kpt_b-1]
                    if c1 > 0.5 and c2 > 0.5: cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
            for kpt in keypoints:
                if len(kpt) >= 3:
                    x, y, conf = kpt[:3]
                    if conf > 0.5: cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)
        except Exception as e: print(f"Error drawing skeleton: {e}")

    def can_send_alert(self, person_id, alert_type, cooldown_seconds=60):
        """Prevents spamming alerts for the same person and type."""
        now = time.time()
        key = (person_id, alert_type)
        last_alert_time_for_key = self.alert_cooldown.get(key, 0)
        if now - last_alert_time_for_key > cooldown_seconds:
            self.alert_cooldown[key] = now
            return True
        return False

    def trigger_alert(self, message, person_id, alert_type):
        # (Modified to add to a list for Flask and handle cooldown logic)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        self.alerts.appendleft(log_message) # Add to front of deque
        print(f"ALERT: {log_message}") # Also print to console

        # Recording logic is now part of draw_annotations decision making
        # based on whether any alert was triggered in the frame.

    def save_alert_recording(self, alert_message_for_filename="alert_event"):
        """Saves a video clip from the recording buffer."""
        if not self.recording_buffer:
            self.add_log_message("[INFO] Recording buffer empty, cannot save alert video.")
            self.is_recording_alert = False # Reset flag
            return

        timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_alert_msg = "".join(c if c.isalnum() else "_" for c in alert_message_for_filename[:30])
        
        video_filename = f"alerts_data/recordings/REC_{safe_alert_msg}_{timestamp_file}.avi"
        json_filename = f"alerts_data/json/INFO_{safe_alert_msg}_{timestamp_file}.json"

        # Get frames from buffer (e.g., last 30 seconds of buffer, or all if shorter)
        # For simplicity, let's try to get frames around the alert time.
        # This implementation will save a short clip from the current buffer.
        # A more robust way would be to continuously save and snip.
        
        # Save ~10-15 seconds of footage (approx 50-75 frames if buffer is at 5fps)
        # Or up to 1 minute if that's what the buffer holds.
        frames_to_save_raw = list(self.recording_buffer) # Get a snapshot
        
        # Let's aim for a fixed duration recording, e.g., 15 seconds, after an alert starts
        # The `is_recording_alert` flag is set when an alert is triggered.
        # This function is called. It should record for a duration then stop.
        
        self.add_log_message(f"[INFO] Starting to save recording: {video_filename}")
        
        # Determine video properties from the first frame
        if not frames_to_save_raw:
            self.is_recording_alert = False
            return

        _ , first_frame_for_props = frames_to_save_raw[0]
        height, width = first_frame_for_props.shape[:2]
        # Use a low FPS for recordings to save space, or estimate from buffer
        # Buffer stores (timestamp, frame). We can estimate FPS if needed.
        # For simplicity, fixed FPS for recording.
        record_fps = 5.0 
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(video_filename, fourcc, record_fps, (width, height))

        # Record for, say, 15 seconds from the point this function is called
        # This means collecting frames for 15s *after* the alert.
        # The buffer should ideally store *before* the alert.
        # Let's refine: Save what's in buffer (pre-alert) + record for X seconds post-alert.
        
        # Write frames currently in buffer (these are pre-alert or during alert start)
        for _, frame_img in frames_to_save_raw:
            out.write(frame_img)
        
        # Continue recording for a short duration, e.g., 10 more seconds
        # This part is tricky with the current buffer setup.
        # A simpler approach for now: save the entire buffer content when an alert is triggered.
        # The `maxlen` of `recording_buffer`
```python
# (Continuing from the previous response for app.py)

        # ... (Inside BehaviorDetectionSystem class)

    def save_alert_recording(self, alert_message_for_filename="alert_event"):
        """Saves a video clip from the recording buffer."""
        if not self.recording_buffer:
            self.add_log_message("[INFO] Recording buffer empty, cannot save alert video.")
            self.is_recording_alert = False # Reset flag
            return

        timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize alert message for filename (take first few words)
        safe_alert_msg_parts = alert_message_for_filename.split(" - Person")[0].replace(" ", "_")
        safe_alert_msg = "".join(c if c.isalnum() else "_" for c in safe_alert_msg_parts[:30])
        
        video_filename = f"alerts_data/recordings/REC_{safe_alert_msg}_{timestamp_file}.avi"
        json_filename = f"alerts_data/json/INFO_{safe_alert_msg}_{timestamp_file}.json"

        # Get a snapshot of the current buffer for saving
        # This will save the content of the buffer at the time of the alert
        frames_to_save_data = list(self.recording_buffer) 
        
        self.add_log_message(f"[INFO] Starting to save recording: {video_filename} using {len(frames_to_save_data)} frames.")
        
        if not frames_to_save_data:
            self.add_log_message("[WARNING] No frames in buffer snapshot to save for alert.")
            self.is_recording_alert = False
            return

        _ , first_frame_for_props = frames_to_save_data[0]
        height, width = first_frame_for_props.shape[:2]
        
        # Estimate FPS from buffer timestamps if possible, otherwise default
        record_fps = 5.0 
        if len(frames_to_save_data) > 1:
            time_diff = frames_to_save_data[-1][0] - frames_to_save_data[0][0]
            if time_diff > 0:
                estimated_fps = len(frames_to_save_data) / time_diff
                record_fps = min(max(estimated_fps, 1.0), 10.0) # Cap FPS for recording
        
        self.add_log_message(f"Recording with estimated FPS: {record_fps:.2f}")

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(video_filename, fourcc, record_fps, (width, height))

        for frame_time, frame_img in frames_to_save_data:
            out.write(frame_img)
        
        out.release()
        self.add_log_message(f"[INFO] Saved alert recording: {video_filename}")

        # Save alert info
        alert_info = {
            'timestamp': timestamp_file,
            'triggering_alert_message': alert_message_for_filename,
            'video_file': video_filename,
            'duration_seconds': len(frames_to_save_data) / record_fps if record_fps > 0 else 0,
            'source_video': self.video_source
        }
        try:
            with open(json_filename, 'w') as f:
                json.dump(alert_info, f, indent=2)
            self.add_log_message(f"[INFO] Saved alert metadata: {json_filename}")
        except Exception as e:
            self.add_log_message(f"[ERROR] Failed to save alert JSON: {e}")
        
        self.is_recording_alert = False # Reset recording flag after saving
        # self.alert_active_for_recording is reset in draw_annotations based on current frame alerts

    def add_log_message(self, message):
        """Adds a message to the internal log and prints it."""
        # This method replaces the direct appends to self.alerts for system messages
        # Alert messages from trigger_alert are still added with timestamp
        if not message.startswith("["): # Add timestamp if not already present
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_entry = f"[{timestamp}] [SYSTEM] {message}"
        else:
            log_entry = message

        print(log_entry) # Print to console for debugging
        self.alerts.appendleft(log_entry) # Add to front of deque

    def get_status(self):
        """Returns current statistics and recent alerts for the web UI."""
        return {
            "running": self.running,
            "source": self.video_source,
            "is_file": self.is_video_file,
            "stats": self.stats.copy(),
            "alerts": list(self.alerts) # Convert deque to list for JSON
        }
    
    def update_video_source(self, new_source):
        if self.running:
            self.add_log_message("[WARNING] Cannot update source while detection is running. Please stop first.")
            return False
        
        self.video_source = new_source
        self.is_video_file = self.check_if_video_file(self.video_source)
        source_type = "Video File" if self.is_video_file else ("Webcam" if self.video_source.isdigit() else "RTSP Stream")
        self.add_log_message(f"Video source updated to: {self.video_source} ({source_type})")
        return True

# --- Flask App Setup ---
app = Flask(__name__)
# Initialize with a default source. Can be changed via UI if needed.
# Ensure video2.mp4 exists or use 0 for webcam.
DEFAULT_VIDEO_SOURCE = "video2.mp4" 
if not os.path.exists(DEFAULT_VIDEO_SOURCE) and DEFAULT_VIDEO_SOURCE != "0":
    print(f"[WARNING] Default video '{DEFAULT_VIDEO_SOURCE}' not found. Consider using webcam '0' or ensure file exists.")
    # You might want to fall back to webcam if file not found:
    # DEFAULT_VIDEO_SOURCE = "0" 
system = BehaviorDetectionSystem(video_source=DEFAULT_VIDEO_SOURCE)

def generate_frames():
    """Generator function for video streaming."""
    while True:
        with system.detection_lock:
            frame = system.latest_frame
        
        if frame is None:
            # If no frame yet (e.g., system starting or stopped), send a placeholder
            # Or just wait with a short sleep. Let's send a black frame.
            black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(black_frame, "System Stopped or Initializing...", (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', black_frame)
            if not ret:
                time.sleep(0.1) # Wait a bit before trying again
                continue
            frame_bytes = buffer.tobytes()
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                time.sleep(0.01) # Minimal sleep if encoding fails
                continue
            frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03) # Adjust to control streaming FPS, e.g., ~30fps

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html', current_source=system.video_source)

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start', methods=['POST'])
def start_detection_route():
    if not system.running:
        # Optionally get new source from request if provided
        data = request.get_json()
        new_source = data.get('source', system.video_source) # Use current if not provided
        if new_source != system.video_source:
            system.update_video_source(new_source) # Update if changed

        threading.Thread(target=system.start_detection, daemon=True).start()
        return jsonify({"message": "Detection starting...", "status": "starting"})
    return jsonify({"message": "Detection already running.", "status": "running"})

@app.route('/stop', methods=['POST'])
def stop_detection_route():
    if system.running:
        system.stop_detection() # This is synchronous for now, but actual stop might take a moment
        return jsonify({"message": "Detection stopping...", "status": "stopping"})
    return jsonify({"message": "Detection not running.", "status": "stopped"})

@app.route('/status', methods=['GET'])
def get_status_route():
    return jsonify(system.get_status())

@app.route('/update_source', methods=['POST'])
def update_source_route():
    if system.running:
        return jsonify({"success": False, "message": "Stop detection before changing source."}), 400
    data = request.get_json()
    new_source = data.get('source')
    if new_source:
        if system.update_video_source(new_source):
             return jsonify({"success": True, "message": f"Source updated to {new_source}."})
        else: # Should not happen if not running
             return jsonify({"success": False, "message": "Failed to update source."}), 500
    return jsonify({"success": False, "message": "No source provided."}), 400


if __name__ == "__main__":
    print("Flask app starting... Open http://127.0.0.1:5000 in your browser.")
    # Set use_reloader=False if you face issues with threads and model loading during reload
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False, threaded=True)
```

**2. `templates/index.html`**

Create a folder named `templates` in the same directory as `app.py`. Inside `templates`, create `index.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Worker Safety Detection System</title>
    <style>
        body { font-family: sans-serif; margin: 0; background-color: #f4f4f4; display: flex; flex-direction: column; min-height: 100vh; }
        header { background-color: #333; color: white; padding: 1em; text-align: center; }
        .container { display: flex; flex: 1; flex-wrap: wrap; padding: 10px; }
        .video-container { flex: 3; min-width: 640px; margin: 10px; background-color: black; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        .video-container img { display: block; width: 100%; height: auto; }
        .controls-stats-container { flex: 1; min-width: 300px; margin: 10px; display: flex; flex-direction: column; }
        .panel { background-color: white; padding: 15px; margin-bottom: 15px; border-radius: 5px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        .panel h2 { margin-top: 0; font-size: 1.2em; border-bottom: 1px solid #eee; padding-bottom: 5px; }
        button { padding: 10px 15px; margin: 5px; border: none; border-radius: 3px; cursor: pointer; background-color: #5cb85c; color: white; }
        button:hover { opacity: 0.9; }
        #stopBtn { background-color: #d9534f; }
        #updateSourceBtn { background-color: #f0ad4e; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 10px; }
        .stat-item { background-color: #e9ecef; padding: 10px; border-radius: 3px; text-align: center; }
        .stat-item strong { display: block; font-size: 1.5em; }
        #alertsLog { list-style-type: none; padding: 0; height: 200px; overflow-y: auto; border: 1px solid #ddd; font-size: 0.9em; }
        #alertsLog li { padding: 5px; border-bottom: 1px solid #eee; }
        #alertsLog li:last-child { border-bottom: none; }
        .alert-critical { color: red; font-weight: bold; }
        .system-status { padding: 10px; background-color: #ddd; text-align: center; margin-bottom: 10px; border-radius: 3px; }
        input[type="text"] { padding: 8px; margin-right: 5px; border: 1px solid #ccc; border-radius: 3px; width: calc(100% - 90px); box-sizing: border-box;}
    </style>
</head>
<body>
    <header>
        <h1>Worker Safety Detection System</h1>
    </header>

    <div class="container">
        <div class="video-container">
            <img id="videoFeed" src="{{ url_for('video_feed') }}" alt="Video Feed">
        </div>

        <div class="controls-stats-container">
            <div class="panel">
                <h2>System Control</h2>
                <div class="system-status" id="systemStatus">Status: Unknown</div>
                <div>
                    <input type="text" id="videoSource" value="{{ current_source }}" placeholder="Video Path or RTSP URL or Webcam ID">
                    <button id="updateSourceBtn">Update</button>
                </div>
                <button id="startBtn">Start Detection</button>
                <button id="stopBtn">Stop Detection</button>
            </div>

            <div class="panel">
                <h2>Live Statistics</h2>
                <div class="stats-grid">
                    <div class="stat-item"><span id="peopleCount">0</span><strong>People</strong></div>
                    <div class="stat-item"><span id="fallenCount" style="color: red;">0</span><strong>Fallen</strong></div>
                    <div class="stat-item"><span id="stationaryCount" style="color: orange;">0</span><strong>Stationary</strong></div>
                    <div class="stat-item"><span id="noHelmetCount" style="color: magenta;">0</span><strong>No Helmet</strong></div>
                </div>
            </div>

            <div class="panel">
                <h2>Event Log</h2>
                <ul id="alertsLog">
                    <li>Log will appear here...</li>
                </ul>
            </div>
        </div>
    </div>

    <script>
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const updateSourceBtn = document.getElementById('updateSourceBtn');
        const videoSourceInput = document.getElementById('videoSource');
        const systemStatusDiv = document.getElementById('systemStatus');

        const peopleCountEl = document.getElementById('peopleCount');
        const fallenCountEl = document.getElementById('fallenCount');
        const stationaryCountEl = document.getElementById('stationaryCount');
        const noHelmetCountEl = document.getElementById('noHelmetCount');
        const alertsLogUl = document.getElementById('alertsLog');

        startBtn.addEventListener('click', () => {
            const source = videoSourceInput.value;
            fetch('/start', { 
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ source: source }) 
            })
            .then(response => response.json())
            .then(data => {
                console.log(data.message);
                // systemStatusDiv.textContent = `Status: ${data.status}`;
            });
        });

        stopBtn.addEventListener('click', () => {
            fetch('/stop', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                console.log(data.message);
                // systemStatusDiv.textContent = `Status: ${data.status}`;
            });
        });

        updateSourceBtn.addEventListener('click', () => {
            const newSource = videoSourceInput.value;
            fetch('/update_source', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ source: newSource })
            })
            .then(response => response.json())
            .then(data => {
                alert(data.message); // Simple alert for update status
                if(data.success) {
                    systemStatusDiv.textContent = `Source: ${newSource}`;
                }
            });
        });

        function updateStatus() {
            fetch('/status')
            .then(response => response.json())
            .then(data => {
                // Update system status display
                let statusText = data.running ? 'Running' : 'Stopped';
                statusText += ` | Source: ${data.source}`;
                systemStatusDiv.textContent = `Status: ${statusText}`;
                if (data.running) {
                    systemStatusDiv.style.backgroundColor = '#d4edda'; // Greenish for running
                    systemStatusDiv.style.color = '#155724';
                } else {
                    systemStatusDiv.style.backgroundColor = '#f8d7da'; // Reddish for stopped
                    systemStatusDiv.style.color = '#721c24';
                }


                // Update stats
                peopleCountEl.textContent = data.stats.people_count;
                fallenCountEl.textContent = data.stats.fallen_count;
                stationaryCountEl.textContent = data.stats.stationary_count;
                noHelmetCountEl.textContent = data.stats.no_helmet_count;

                // Update alerts log (only if changed to avoid flicker)
                const currentAlertsHTML = alertsLogUl.innerHTML;
                let newAlertsHTML = '';
                data.alerts.slice(0, 20).forEach(alertMsg => { // Show latest 20
                    let alertClass = '';
                    if (alertMsg.includes("FALLEN") || alertMsg.includes("NO HELMET")) {
                        alertClass = 'class="alert-critical"';
                    }
                    newAlertsHTML += `<li ${alertClass}>${escapeHtml(alertMsg)}</li>`;
                });
                
                // A simple check to see if content likely changed; more robust would be a hash
                if (alertsLogUl.children.length !== data.alerts.slice(0,20).length || 
                    (alertsLogUl.firstChild && !data.alerts.slice(0,20).includes(alertsLogUl.firstChild.textContent))) {
                    alertsLogUl.innerHTML = newAlertsHTML || "<li>No new events.</li>";
                }

            })
            .catch(error => console.error('Error fetching status:', error));
        }
        
        function escapeHtml(unsafe) {
            return unsafe
                 .replace(/&/g, "&amp;")
                 .replace(/</g, "&lt;")
                 .replace(/>/g, "&gt;")
                 .replace(/"/g, "&quot;")
                 .replace(/'/g, "&#039;");
        }

        // Fetch status periodically
        setInterval(updateStatus, 2000); // Update every 2 seconds
        updateStatus(); // Initial call
    </script>
</body>
</html>
```

**Key Changes and Notes:**

1.  **Flask `app.py`:**
    *   `PersonTracker`: Added `has_helmet` attribute and `get_head_region()` method. Modified `update()` to accept `bbox`.
    *   `BehaviorDetectionSystem`:
        *   Loads two models: `pose_model` and `object_model`.
        *   `load_models()`: Includes logic to find `helmet_class_id` and `person_class_id` from the object model's names.
        *   `detection_loop()`:
            *   Performs pose detection first.
            *   Updates trackers based on pose.
            *   Performs object detection (for helmets).
            *   Associates helmets with tracked persons using `get_head_region()` and IoU.
            *   Sets `tracker.has_helmet`.
        *   `draw_annotations()`:
            *   Draws status for "NO HELMET".
            *   Updates `no_helmet_count` and triggers "NO HELMET" alerts.
            *   Uses `can_send_alert` for all alert types to prevent spam.
            *   Revised alert recording trigger logic.
        *   `save_alert_recording()`: Improved to make filenames more descriptive and estimate FPS for the recording.
        *   `add_log_message()`: Centralized way to add messages to `self.alerts`.
        *   `get_status()`: Returns all necessary data for the frontend.
        *   `update_video_source()`: Allows changing the video source when detection is stopped.
    *   **Flask Routes:**
        *   `/video_feed`: Streams MJPEG.
        *   `/start`, `/stop`: Control detection. Start can now implicitly take the source from the input field.
        *   `/status`: Provides JSON data for UI updates.
        *   `/update_source`: Endpoint to change the video source.
    *   **Error Handling:** Added some basic error messages and reconnection attempts.
    *   **Default Video Source:** Added a check for `video2.mp4` existence.

2.  **`templates/index.html`:**
    *   **Layout:** Video feed on the left, controls/stats/log on the right.
    *   **Video Source Input:** Added an input field and "Update" button.
    *   **Statistics:** Displays people, fallen, stationary, and no-helmet counts.
    *   **Event Log:** Shows recent alerts/log messages. Critical alerts are highlighted.
    *   **JavaScript:**
        *   Handles button clicks to call Flask backend API endpoints.
        *   Periodically calls `/status` to refresh the UI.
        *   Updates the DOM elements with new data.
        *   Basic `escapeHtml` to prevent XSS from alert messages.

**To Run This:**

1.  Save `app.py` and the `templates/index.html` file in the correct structure:
    ```
    your_project_folder/
    ├── app.py
    ├── video2.mp4  (or your test video/RTSP/webcam ID)
    ├── alerts_data/ (will be created automatically)
    │   ├── recordings/
    │   └── json/
    └── templates/
        └── index.html
    ```
2.  Make sure you have a video file named `video2.mp4` in the same directory as `app.py`, or change `DEFAULT_VIDEO_SOURCE` in `app.py` to `0` (for webcam) or a valid RTSP URL.
3.  Install necessary libraries: `pip install Flask ultralytics opencv-python numpy Pillow`
4.  Run from your terminal: `python app.py`
5.  Open your web browser and go to `http://127.0.0.1:5000`.

**Important Considerations for Helmet Detection:**

*   **Model:** The effectiveness of helmet detection heavily depends on the `object_model`. `yolov8n.pt` (COCO) detects 'person' but not 'helmet' specifically.
    *   **Best Solution:** Train a custom YOLO model on a dataset of people with and without helmets.
    *   **Alternative:** Find a pre-trained model that already detects safety helmets.
    *   **Simulating:** If `helmet_class_id` is `None`, the "NO HELMET" logic will trigger for everyone. You could temporarily map another class from COCO (e.g., 'sports ball') to `self.helmet_class_id` in `load_models` just to see the bounding boxes and logic work if you place that object near a person's head.
*   **`get_head_region()`:** This is an estimation. Its accuracy affects helmet association.
*   **IoU Threshold:** The `iou(head_region, helmet_box) > 0.1` threshold might need tuning.

This comprehensive solution provides a solid foundation for a web-based worker safety system. You can further enhance it with features like user authentication, more robust error handling, database storage for alerts, and more sophisticated video analytics.
