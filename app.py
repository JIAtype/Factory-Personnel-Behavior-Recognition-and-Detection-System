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

# --- PersonTracker Class ---
class PersonTracker:
    KEYPOINTS_HISTORY_MAXLEN = 30
    MOVEMENT_THRESHOLD_PIXELS = 10  # Pixels for significant movement
    STATIONARY_DURATION_SECONDS = 180  # 3 minutes to be considered stationary
    FALL_ASPECT_RATIO_THRESHOLD = 1.8
    FALL_HIP_SHOULDER_TOLERANCE_PX = 10
    FALL_HIP_ABOVE_SHOULDER_FACTOR = 0.2 # Hips need to be this factor of body_height above shoulders

    def __init__(self, person_id, keypoints):
        self.id = person_id
        self.keypoints_history = deque(maxlen=self.KEYPOINTS_HISTORY_MAXLEN)
        self.last_movement_time = time.time()
        self.is_fallen = False
        self.is_stationary = False
        self.fall_alert_sent = False
        self.stationary_alert_sent = False
        self.no_helmet_alert_sent = False
        self.last_alert_time = 0 # Generic last alert time for this tracker (not currently used by can_send_alert which is global)
        self.keypoints_history.append(keypoints)
        self.has_helmet = True # Assume has helmet initially, detection will update this
        self.bbox = None # Store person's bounding box [x1, y1, x2, y2] from pose model

    def update(self, keypoints, bbox=None):
        self.keypoints_history.append(keypoints)
        if bbox is not None:
            self.bbox = bbox

        if len(self.keypoints_history) >= 2:
            movement = self.calculate_movement()
            if movement > self.MOVEMENT_THRESHOLD_PIXELS:
                self.last_movement_time = time.time()
                if self.is_stationary: self.stationary_alert_sent = False # Reset alert if moved
                if self.is_fallen: self.fall_alert_sent = False # Reset alert if moved (and not fallen anymore)
                # No_helmet alert is reset when helmet is detected, not purely on movement.
                self.is_stationary = False
                self.is_fallen = False # Fall state should be re-evaluated each frame
            else:
                if time.time() - self.last_movement_time > self.STATIONARY_DURATION_SECONDS:
                    self.is_stationary = True

        fall_detected = self.detect_fall()
        if fall_detected and not self.is_fallen: # New fall event
            self.is_fallen = True
            # self.fall_alert_sent = False # Alert will be triggered by main loop if not sent
        elif not fall_detected and self.is_fallen: # Was fallen, but not anymore
            self.is_fallen = False
            self.fall_alert_sent = False # Allow re-alerting if they fall again later

    def calculate_movement(self):
        if len(self.keypoints_history) < 2: return 0
        current_kpts_arr = self.keypoints_history[-1]
        previous_kpts_arr = self.keypoints_history[-2]
        
        # Ensure keypoints are numpy arrays for easier processing
        current = np.array(current_kpts_arr)
        previous = np.array(previous_kpts_arr)

        total_movement = 0
        valid_points = 0
        
        # Keypoints are expected to be (N, 3) where N is num_keypoints, and columns are x, y, conf
        min_len = min(len(current), len(previous))
        for i in range(min_len):
            if (current.shape[1] > 2 and previous.shape[1] > 2 and # Check if conf is present
                current[i][2] > 0.5 and previous[i][2] > 0.5): # Confidence threshold
                dx = current[i][0] - previous[i][0]
                dy = current[i][1] - previous[i][1]
                movement = math.sqrt(dx*dx + dy*dy)
                total_movement += movement
                valid_points += 1
        return total_movement / max(valid_points, 1)

    def detect_fall(self):
        if not self.keypoints_history: return False
        keypoints = np.array(self.keypoints_history[-1]) # Ensure numpy array

        # Keypoint indices for COCO format
        nose_idx, l_shoulder_idx, r_shoulder_idx, l_hip_idx, r_hip_idx = 0, 5, 6, 11, 12
        
        # Helper to get keypoint if valid
        def get_kpt(idx):
            if len(keypoints) > idx and keypoints[idx][2] > 0.5: # Check confidence
                return keypoints[idx]
            return None

        # nose = get_kpt(nose_idx) # Nose not directly used in this fall logic
        left_shoulder = get_kpt(l_shoulder_idx)
        right_shoulder = get_kpt(r_shoulder_idx)
        left_hip = get_kpt(l_hip_idx)
        right_hip = get_kpt(r_hip_idx)

        if (left_shoulder is not None and right_shoulder is not None and 
            left_hip is not None and right_hip is not None):
            
            shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
            hip_center_y = (left_hip[1] + right_hip[1]) / 2
            
            # Check if person is roughly horizontal (hips not significantly lower than shoulders)
            if hip_center_y < shoulder_center_y + self.FALL_HIP_SHOULDER_TOLERANCE_PX:
                body_width = abs(left_shoulder[0] - right_shoulder[0])
                # Use a small epsilon to avoid division by zero if body_height is tiny
                body_height = abs(shoulder_center_y - hip_center_y) + 1e-6 
                
                aspect_ratio = body_width / body_height
                if aspect_ratio > self.FALL_ASPECT_RATIO_THRESHOLD:
                    return True
            
            # More lenient check: if hips are significantly above shoulders (e.g. headstand like fall)
            # This requires careful tuning, as it might be overly sensitive.
            # Using a factor of estimated body height (shoulder to hip) for this check.
            estimated_body_segment_height = abs(shoulder_center_y - hip_center_y)
            if estimated_body_segment_height > 0 and \
               hip_center_y < shoulder_center_y - (estimated_body_segment_height * self.FALL_HIP_ABOVE_SHOULDER_FACTOR):
                return True
        return False

    def get_head_region(self):
        """Estimates head region based on keypoints or bbox."""
        if not self.keypoints_history: return None
        
        keypoints = np.array(self.keypoints_history[-1])
        
        # Keypoints indices: 0:Nose, 1:L_Eye, 2:R_Eye, 3:L_Ear, 4:R_Ear
        head_indices = [0, 1, 2, 3, 4]
        visible_head_kpts = []
        for i in head_indices:
            if len(keypoints) > i and keypoints[i][2] > 0.3: # Confidence for head keypoints
                visible_head_kpts.append(keypoints[i][:2])
        
        if not visible_head_kpts:
            # Fallback: use area above shoulders if head keypoints are not visible, using person's bbox
            # This is a rough estimate and might not be accurate if person is bent over.
            if self.bbox is not None:
                x1, y1, x2, y2 = map(int, self.bbox[:4])
                # Estimate head as top 20-25% of the person's bounding box height
                head_height_ratio = 0.25 
                head_y1 = y1
                head_y2 = y1 + (y2 - y1) * head_height_ratio
                # Head width can be approximated from bbox width, or a bit narrower
                head_x1 = x1 + (x2 - x1) * 0.1 # Indent a bit
                head_x2 = x2 - (x2 - x1) * 0.1
                return (int(head_x1), int(head_y1), int(head_x2), int(head_y2))
            return None

        visible_head_kpts_np = np.array(visible_head_kpts)
        min_x, min_y = np.min(visible_head_kpts_np, axis=0)
        max_x, max_y = np.max(visible_head_kpts_np, axis=0)

        width = max_x - min_x
        height = max_y - min_y
        
        padding_w = width * 0.35  # Increased padding for width
        padding_h_up = height * 1.0 # More padding upwards for helmet
        padding_h_down = height * 0.25 # Padding downwards

        hx1 = int(min_x - padding_w)
        hy1 = int(min_y - padding_h_up)
        hx2 = int(max_x + padding_w)
        hy2 = int(max_y + padding_h_down)
        
        # Ensure coordinates are positive
        hx1, hy1 = max(0, hx1), max(0, hy1)

        return (hx1, hy1, hx2, hy2)


# --- BehaviorDetectionSystem Class ---
class BehaviorDetectionSystem:
    # --- Configuration Constants ---
    HELMET_IOU_THRESHOLD = 0.15
    ALERT_COOLDOWN_SECONDS = 60 # Cooldown per person per alert type
    
    TRACKER_DISAPPEAR_SECONDS_NO_MATCH = 10 # If no match for this long, tracker is removed
    TRACKER_MATCH_MAX_DISTANCE_PX = 150 # Max distance (pixels) for matching detection to tracker

    POSE_PERSON_CONF_THRESHOLD = 0.5
    OBJECT_HELMET_CONF_THRESHOLD = 0.4
    OBJECT_PERSON_CONF_THRESHOLD = 0.5 # If using object model to also find persons

    RECORDING_BUFFER_MAXLEN = 300 # Frames (e.g., 60s at 5fps, 10s at 30fps)
    RECORDING_VIDEO_FPS = 5.0 # Target FPS for saved alert videos
    # --- End Configuration Constants ---

    def __init__(self, video_source="test_video.mp4"):
        self.video_source = video_source
        self.pose_model = None
        self.object_model = None
        self.cap = None
        self.trackers = {}
        self.next_id = 0
        self.running = False
        self.latest_frame = None
        self.detection_lock = threading.Lock() # Protects latest_frame, stats, alerts

        self.alerts = deque(maxlen=100) # Increased maxlen for more log history
        self.stats = {
            "people_count": 0, "fallen_count": 0,
            "stationary_count": 0, "no_helmet_count": 0,
            "model_status": "Not Loaded", "detection_status": "Stopped"
        }

        self.is_video_file = self.check_if_video_file(video_source)
        self.alert_cooldown_map = {} # {(person_id, alert_type): last_alert_timestamp}

        os.makedirs("alerts_data/recordings", exist_ok=True)
        os.makedirs("alerts_data/json", exist_ok=True)
        self.recording_buffer = deque(maxlen=self.RECORDING_BUFFER_MAXLEN)
        self.is_recording_alert_active = False # True if an alert video is currently being saved
        self.an_alert_condition_exists = False # True if any person currently has an active alert state

    def check_if_video_file(self, source):
        try:
            int(source)
            return False # Webcam ID
        except ValueError:
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
            return any(str(source).lower().endswith(ext) for ext in video_extensions)

    def load_models(self):
        self.add_log_message("[SYSTEM] Attempting to load models...")
        self.stats["model_status"] = "Loading..."
        try:
            self.pose_model = YOLO('yolov8x-pose.pt').to('cuda')
            self.object_model = YOLO('yolov8n.pt').to('cuda')
            
            self.object_model_classes = self.object_model.names
            self.helmet_class_id = next((k for k, v in self.object_model_classes.items() if v.lower() in ['helmet', 'safety helmet']), None)
            self.person_class_id = next((k for k, v in self.object_model_classes.items() if v.lower() == 'person'), None)

            if self.helmet_class_id is None:
                self.add_log_message("[WARNING] 'helmet' class not found in object_model. Helmet detection will be illustrative/disabled.")
            if self.person_class_id is None:
                self.add_log_message("[ERROR] 'person' class not found in object_model. This is crucial for some functionalities.")
                self.stats["model_status"] = "Error: Person class missing"
                return False
            
            self.add_log_message("[INFO] Models loaded successfully.")
            self.stats["model_status"] = "Loaded"
            return True
        except Exception as e:
            self.add_log_message(f"[ERROR] Failed to load models: {str(e)}")
            self.stats["model_status"] = f"Error: {str(e)}"
            self.pose_model = None # Ensure models are None if loading failed
            self.object_model = None
            return False

    def start_detection(self):
        if self.running:
            self.add_log_message("[INFO] Detection already running.")
            return

        if not self.pose_model or not self.object_model:
            if not self.load_models():
                self.stats["detection_status"] = "Stopped (Model Load Failed)"
                return

        self.add_log_message(f"[INFO] Attempting to open video source: {self.video_source}")
        try:
            # Convert to int if it's a digit string (webcam ID), else use as string (path/URL)
            capture_source = int(self.video_source) if str(self.video_source).isdigit() else self.video_source
            self.cap = cv2.VideoCapture(capture_source, cv2.CAP_FFMPEG)
            # self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception as e:
             self.add_log_message(f"[ERROR] OpenCV Error during VideoCapture init: {e}")
             self.cap = None

        if not self.cap or not self.cap.isOpened():
            self.add_log_message(f"[ERROR] Failed to open video source: {self.video_source}")
            self.running = False
            self.stats["detection_status"] = "Stopped (Source Error)"
            return

        self.is_video_file = self.check_if_video_file(self.video_source)
        source_type = "Video File" if self.is_video_file else ("Webcam" if str(self.video_source).isdigit() else "Stream")
        self.add_log_message(f"[INFO] Opened {source_type}: {self.video_source}")
        
        self.running = True
        self.stats["detection_status"] = "Running"
        self.trackers.clear()
        self.next_id = 0
        
        self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.detection_thread.start()
        self.add_log_message("[INFO] Detection thread started.")

    def stop_detection(self):
        self.running = False
        if hasattr(self, 'detection_thread') and self.detection_thread.is_alive():
            self.add_log_message("[INFO] Waiting for detection thread to finish...")
            self.detection_thread.join(timeout=5) # Increased timeout
            if self.detection_thread.is_alive():
                 self.add_log_message("[WARNING] Detection thread did not finish in time.")
        
        with self.detection_lock:
            if self.cap:
                self.cap.release()
                self.cap = None
            self.latest_frame = None
        
        self.trackers.clear()
        # Reset relevant stats, keep model status
        with self.detection_lock:
            self.stats.update({
                "people_count": 0, "fallen_count": 0,
                "stationary_count": 0, "no_helmet_count": 0,
                "detection_status": "Stopped"
            })
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
        
        denominator = float(boxAArea + boxBArea - interArea)
        return interArea / denominator if denominator > 0 else 0.0

    def detection_loop(self):
        while self.running:
            if not self.cap or not self.cap.isOpened():
                self.add_log_message("[WARNING] Video capture is not open. Attempting to reconnect...")
                self.stats["detection_status"] = "Reconnecting..."
                time.sleep(5)
                try:
                    capture_source = int(self.video_source) if str(self.video_source).isdigit() else self.video_source
                    self.cap = cv2.VideoCapture(capture_source)
                except Exception as e:
                    self.add_log_message(f"[ERROR] OpenCV Error during reconnect: {e}")
                    self.cap = None

                if not self.cap or not self.cap.isOpened():
                    self.add_log_message("[ERROR] Reconnect failed. Stopping detection.")
                    self.running = False # This will break the loop
                    self.stats["detection_status"] = "Stopped (Reconnect Failed)"
                    break 
                else:
                    self.add_log_message("[INFO] Reconnected to video source.")
                    self.stats["detection_status"] = "Running"

            ret, frame = self.cap.read()
            if not ret:
                if self.is_video_file:
                    self.add_log_message("[INFO] Video file ended. Looping.")
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else: # Stream error
                    self.add_log_message("[WARNING] Frame not received from stream. Pausing briefly.")
                    time.sleep(0.5) # Longer pause for stream errors
                    continue

            frame_copy_for_buffer = frame.copy()
            self.recording_buffer.append((time.time(), frame_copy_for_buffer))

            # 1. Pose Detection
            pose_results = self.pose_model(frame, verbose=False, conf=self.POSE_PERSON_CONF_THRESHOLD)
            
            current_person_detections_for_tracking = []
            for r in pose_results:
                if r.keypoints is not None and r.boxes is not None:
                    # .data ensures we get tensor data, .cpu().numpy() for numpy array
                    keypoints_data = r.keypoints.data.cpu().numpy() 
                    boxes_data = r.boxes.data.cpu().numpy() # [x1, y1, x2, y2, conf, cls]
                    
                    for i, (kpts, box_coords) in enumerate(zip(keypoints_data, boxes_data)):
                        # Pose model usually detects persons (class 0)
                        # Rely on confidence from boxes_data
                        if box_coords[4] > self.POSE_PERSON_CONF_THRESHOLD: 
                            current_person_detections_for_tracking.append({
                                'keypoints': kpts, # kpts are (N_kpts, 3) with x,y,conf
                                'box': box_coords[:4], # Person bbox [x1,y1,x2,y2]
                                'id': None
                            })
            self.update_trackers(current_person_detections_for_tracking)

            # 2. Object Detection (for Helmets)
            detected_helmets_boxes = []
            if self.helmet_class_id is not None: # Only if helmet class is available and configured
                object_results = self.object_model(frame, verbose=False, classes=[self.helmet_class_id], conf=self.OBJECT_HELMET_CONF_THRESHOLD)
                for r_obj in object_results:
                    for box in r_obj.boxes:
                        # Class check is implicitly handled by `classes` arg in model call, but double check for safety
                        if box.cls == self.helmet_class_id and box.conf > self.OBJECT_HELMET_CONF_THRESHOLD:
                            detected_helmets_boxes.append(box.xyxy[0].cpu().numpy().astype(int))
            
            # 3. Associate Helmets with Tracked Persons
            for tracker in self.trackers.values(): # Iterate over a copy of values if modification occurs
                tracker.has_helmet = False # Reset for current frame, assume no helmet
                if not tracker.keypoints_history or self.helmet_class_id is None: continue

                head_region = tracker.get_head_region()
                if head_region:
                    for helmet_box in detected_helmets_boxes:
                        if self.iou(head_region, helmet_box) > self.HELMET_IOU_THRESHOLD:
                            tracker.has_helmet = True
                            break 
            
            # 4. Draw annotations and check alerts
            annotated_frame = self.draw_annotations_and_handle_alerts(frame.copy())

            with self.detection_lock:
                self.latest_frame = annotated_frame.copy()
            
            # No artificial delay needed here, processing speed will dictate FPS.
            # time.sleep(0.01) 

        # Loop ended, ensure cap is released if it was open
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.add_log_message("[INFO] Detection loop finished.")
        if self.stats["detection_status"] == "Running": # If not stopped by user or error
            self.stats["detection_status"] = "Stopped (Loop Ended)"


    def update_trackers(self, detections):
        # Iterate over a list of items to allow deletion from self.trackers
        current_tracker_ids = list(self.trackers.keys())
        
        matched_detection_indices = set()

        for tracker_id in current_tracker_ids:
            if tracker_id not in self.trackers: continue # Tracker might have been removed in a previous iteration
            tracker = self.trackers[tracker_id]
            best_match_idx = -1
            min_distance = self.TRACKER_MATCH_MAX_DISTANCE_PX
            
            for i, det in enumerate(detections):
                if i in matched_detection_indices: continue
                
                distance = self.calculate_detection_distance(tracker, det)
                if distance < min_distance:
                    min_distance = distance
                    best_match_idx = i
            
            if best_match_idx != -1:
                detection = detections[best_match_idx]
                tracker.update(detection['keypoints'], detection['box'])
                detection['id'] = tracker_id # Assign tracker ID to the matched detection
                matched_detection_indices.add(best_match_idx)
            else:
                # No match found for this tracker
                # Consider disappeared if no detection for TRACKER_DISAPPEAR_SECONDS_NO_MATCH seconds
                # This simple tracker doesn't have a robust "time_last_seen" for unmatched trackers.
                # It relies on last_movement_time which is updated only on match.
                # A more direct approach: add a `last_seen_time` to tracker, update it always.
                # For now, the original logic: if no movement and no match for X seconds.
                if time.time() - tracker.last_movement_time > self.TRACKER_DISAPPEAR_SECONDS_NO_MATCH :
                     del self.trackers[tracker_id]

        # Add new trackers for unmatched detections
        for i, detection in enumerate(detections):
            if i not in matched_detection_indices:
                new_tracker = PersonTracker(self.next_id, detection['keypoints'])
                new_tracker.bbox = detection['box']
                self.trackers[self.next_id] = new_tracker
                detection['id'] = self.next_id # Also mark this detection as assigned
                self.next_id = (self.next_id + 1) % 100000 # Cycle IDs


    def calculate_detection_distance(self, tracker, detection):
        if not tracker.keypoints_history: return float('inf')
        
        last_kps_raw = tracker.keypoints_history[-1]
        current_kps_raw = detection['keypoints']

        try:
            last_keypoints = np.array(last_kps_raw)
            current_keypoints = np.array(current_kps_raw)

            # Filter keypoints by confidence
            last_valid_kps = last_keypoints[last_keypoints[:, 2] > 0.5][:, :2] # x, y
            current_valid_kps = current_keypoints[current_keypoints[:, 2] > 0.5][:, :2] # x, y

            if last_valid_kps.shape[0] == 0 or current_valid_kps.shape[0] == 0:
                # Fallback to bounding box center distance if keypoints are insufficient
                if tracker.bbox is not None and 'box' in detection and detection['box'] is not None:
                    b1_cx = (tracker.bbox[0] + tracker.bbox[2]) / 2
                    b1_cy = (tracker.bbox[1] + tracker.bbox[3]) / 2
                    b2_cx = (detection['box'][0] + detection['box'][2]) / 2
                    b2_cy = (detection['box'][1] + detection['box'][3]) / 2
                    return np.linalg.norm(np.array([b1_cx, b1_cy]) - np.array([b2_cx, b2_cy]))
                return float('inf')
            
            # Calculate distance based on the center of valid keypoints
            last_center = np.mean(last_valid_kps, axis=0)
            current_center = np.mean(current_valid_kps, axis=0)
            return np.linalg.norm(last_center - current_center)

        except (IndexError, ValueError) as e: # More specific exceptions
            self.add_log_message(f"[DEBUG] Error in calculate_detection_distance: {e}")
            return float('inf')
        except Exception as e: # Catch-all for unexpected issues
            self.add_log_message(f"[ERROR] Unexpected error in calculate_detection_distance: {e}")
            return float('inf')

    def draw_annotations_and_handle_alerts(self, frame):
        # Iterate over a list of items in case trackers dict changes (though less likely here)
        # However, tracker attributes are modified.
        
        current_people = 0
        current_fallen = 0
        current_stationary = 0
        current_no_helmet = 0
        
        any_alert_condition_in_frame = False

        # Use list(self.trackers.items()) for safe iteration if trackers could be modified elsewhere,
        # but here it's within the single detection thread. Modifying tracker attributes is fine.
        for tracker_id, tracker in self.trackers.items():
            if not tracker.keypoints_history: continue
            current_people +=1
            
            box_to_draw = tracker.bbox 
            if box_to_draw is None: # Fallback if bbox from pose wasn't stored/valid
                 valid_keypoints = np.array(tracker.keypoints_history[-1])
                 valid_keypoints = valid_keypoints[valid_keypoints[:, 2] > 0.5]
                 if len(valid_keypoints) > 0:
                    x_coords, y_coords = valid_keypoints[:, 0], valid_keypoints[:, 1]
                    box_to_draw = [x_coords.min(), y_coords.min(), x_coords.max(), y_coords.max()]

            color = (0, 255, 0) # Green default
            status_texts = [f"P{tracker_id}"]

            # Fall Detection
            if tracker.is_fallen:
                color = (0, 0, 255)  # Red
                status_texts.append("FALLEN")
                current_fallen += 1
                any_alert_condition_in_frame = True
                if not tracker.fall_alert_sent and self.can_send_alert(tracker_id, "fall"):
                    self.trigger_alert(f"FALLEN - Person {tracker_id}", tracker_id, "fall")
                    tracker.fall_alert_sent = True
            
            # Stationary Detection (only if not fallen)
            elif tracker.is_stationary:
                color = (0, 165, 255)  # Orange
                status_texts.append("STATIONARY")
                current_stationary += 1
                any_alert_condition_in_frame = True
                if not tracker.stationary_alert_sent and self.can_send_alert(tracker_id, "stationary"):
                    self.trigger_alert(f"STATIONARY - Person {tracker_id}", tracker_id, "stationary")
                    tracker.stationary_alert_sent = True
            
            # Helmet Detection (can be combined with other states)
            if self.helmet_class_id is not None:
                if not tracker.has_helmet:
                    if color == (0, 255, 0): color = (255, 0, 255) # Purple if no other alert color
                    status_texts.append("NO HELMET")
                    current_no_helmet +=1
                    any_alert_condition_in_frame = True
                    if not tracker.no_helmet_alert_sent and self.can_send_alert(tracker_id, "no_helmet"):
                        self.trigger_alert(f"NO HELMET - Person {tracker_id}", tracker_id, "no_helmet")
                        tracker.no_helmet_alert_sent = True
                else: # Has helmet
                    # status_texts.append("HELMET OK") # Can be verbose
                    if tracker.no_helmet_alert_sent: # If was previously alerted for no helmet
                        tracker.no_helmet_alert_sent = False # Reset alert status

            if box_to_draw is not None:
                x1, y1, x2, y2 = map(int, box_to_draw[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Optional: Draw head region for debugging
                # head_r = tracker.get_head_region()
                # if head_r: cv2.rectangle(frame, (head_r[0], head_r[1]), (head_r[2], head_r[3]), (255,255,0), 1)

                text_y_pos = y1 - 7
                for i, text_line in enumerate(status_texts):
                    cv2.putText(frame, text_line, (x1, text_y_pos - (i * 15)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        
        # Update global stats under lock
        with self.detection_lock:
            self.stats["people_count"] = current_people
            self.stats["fallen_count"] = current_fallen
            self.stats["stationary_count"] = current_stationary
            self.stats["no_helmet_count"] = current_no_helmet

        # Manage alert recording state
        self.an_alert_condition_exists = any_alert_condition_in_frame
        if self.an_alert_condition_exists and not self.is_recording_alert_active:
            self.is_recording_alert_active = True # Flag that a recording process is now active
            # Find the first alert message that triggered this recording session
            first_alert_msg_for_filename = "alert_event"
            # Check self.alerts (which is prepended)
            if self.alerts:
                 # Heuristic: find a recent PERSON-specific alert message
                for alert_text in list(self.alerts)[:5]: # Check a few recent alerts
                    if " - Person " in alert_text:
                        first_alert_msg_for_filename = alert_text.split("] ",1)[1]
                        break
            
            threading.Thread(target=self.save_alert_recording, args=(first_alert_msg_for_filename,)).start()

        # Display overall stats on frame
        cv2.putText(frame, f"People: {current_people}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220,220,220), 2)
        cv2.putText(frame, f"Fallen: {current_fallen}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255) if current_fallen else (220,220,220), 2)
        cv2.putText(frame, f"Stationary: {current_stationary}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255) if current_stationary else (220,220,220), 2)
        cv2.putText(frame, f"No Helmet: {current_no_helmet}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255) if current_no_helmet else (220,220,220), 2)
        
        return frame

    def can_send_alert(self, person_id, alert_type):
        now = time.time()
        key = (person_id, alert_type)
        last_alert_time_for_key = self.alert_cooldown_map.get(key, 0)
        if now - last_alert_time_for_key > self.ALERT_COOLDOWN_SECONDS:
            self.alert_cooldown_map[key] = now
            return True
        return False

    def trigger_alert(self, message, person_id, alert_type):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] ALERT: {message}"
        
        with self.detection_lock: # Protect alerts deque
            self.alerts.appendleft(log_message)
        print(log_message) # Console log

    def save_alert_recording(self, alert_message_for_filename="alert_event"):
        self.add_log_message(f"[INFO] Alert recording triggered by: {alert_message_for_filename}")
        
        # Snapshot the buffer at the time of alert triggering
        # Note: buffer continues to fill, so this is a snapshot.
        # A more advanced system might mark a point in a continuous recording stream.
        frames_to_save_data = list(self.recording_buffer)
        
        if not frames_to_save_data:
            self.add_log_message("[WARNING] Recording buffer empty when save_alert_recording called.")
            self.is_recording_alert_active = False # Reset flag
            return

        timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_alert_parts = alert_message_for_filename.split(" - Person")[0].replace(" ", "_").replace("-","_")
        safe_alert_msg = "".join(c for c in safe_alert_parts if c.isalnum() or c == '_')[:30]
        
        video_filename = f"alerts_data/recordings/REC_{safe_alert_msg}_{timestamp_file}.avi"
        json_filename = f"alerts_data/json/INFO_{safe_alert_msg}_{timestamp_file}.json"
        
        self.add_log_message(f"[INFO] Saving {len(frames_to_save_data)} frames to {video_filename}")
        
        _ , first_frame_props = frames_to_save_data[0]
        height, width = first_frame_props.shape[:2]
        
        # Use configured FPS for saving video
        out_fps = self.RECORDING_VIDEO_FPS
        # Alternative: Estimate FPS from buffer timestamps if desired
        # if len(frames_to_save_data) > 1:
        #     time_diff = frames_to_save_data[-1][0] - frames_to_save_data[0][0]
        #     if time_diff > 0: out_fps = min(max(len(frames_to_save_data) / time_diff, 1.0), 15.0)
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(video_filename, fourcc, out_fps, (width, height))

        for _, frame_img in frames_to_save_data:
            video_writer.write(frame_img)
        
        video_writer.release()
        self.add_log_message(f"[INFO] Saved alert recording: {video_filename}")

        alert_info = {
            'timestamp': timestamp_file,
            'triggering_alert_message': alert_message_for_filename,
            'video_file': os.path.basename(video_filename), # Store only filename
            'video_path': video_filename, # Full path for server-side access
            'duration_seconds': len(frames_to_save_data) / out_fps if out_fps > 0 else 0,
            'source_video': str(self.video_source),
            'frames_saved': len(frames_to_save_data)
        }
        try:
            with open(json_filename, 'w') as f:
                json.dump(alert_info, f, indent=2)
            self.add_log_message(f"[INFO] Saved alert metadata: {json_filename}")
        except Exception as e:
            self.add_log_message(f"[ERROR] Failed to save alert JSON: {e}")
        
        self.is_recording_alert_active = False # Reset flag, allowing new recordings if new alerts occur

    def add_log_message(self, message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # Ensure it's not an ALERT message which has its own format
        if not message.startswith("[") and "ALERT:" not in message: 
            log_entry = f"[{timestamp}] [SYSTEM] {message}"
        elif not message.startswith("["): # For other non-ALERT messages without timestamp
            log_entry = f"[{timestamp}] {message}"
        else: # Already formatted
            log_entry = message
        
        print(log_entry) # Console for live debugging
        with self.detection_lock: # Protect alerts deque
            self.alerts.appendleft(log_entry)

    def get_status(self):
        with self.detection_lock: # Ensure consistent read of stats and alerts
            stats_copy = self.stats.copy()
            alerts_list = list(self.alerts)
        return {
            "running": self.running,
            "source": str(self.video_source), # Ensure source is string
            "is_file": self.is_video_file,
            "stats": stats_copy,
            "alerts": alerts_list
        }
    
    def update_video_source(self, new_source):
        if self.running:
            self.add_log_message("[WARNING] Cannot update source while detection is running. Please stop first.")
            return False
        
        self.video_source = new_source
        self.is_video_file = self.check_if_video_file(self.video_source)
        source_type = "Video File" if self.is_video_file else ("Webcam" if str(self.video_source).isdigit() else "Stream")
        self.add_log_message(f"Video source updated to: {self.video_source} ({source_type})")
        return True

# --- Flask App Setup ---
app = Flask(__name__)

DEFAULT_VIDEO_SOURCE = "rtsp://admin:Admin4321@172.30.40.125/554/live/stream1"
# DEFAULT_VIDEO_SOURCE = "Video2.mp4" 
# # Default to webcam '0' for broader compatibility
# Example: DEFAULT_VIDEO_SOURCE = "your_video_file.mp4"
# Example: DEFAULT_VIDEO_SOURCE = "rtsp://user:pass@ip_address/stream"

# Check if a specific video file is intended and exists
# if not str(DEFAULT_VIDEO_SOURCE).isdigit() and not os.path.exists(DEFAULT_VIDEO_SOURCE):
# print(f"[WARNING] Default video source '{DEFAULT_VIDEO_SOURCE}' not found or not a webcam ID. Please check.")

system = BehaviorDetectionSystem(video_source=DEFAULT_VIDEO_SOURCE)

def generate_frames():
    """Generator function for video streaming."""
    while True:
        frame_to_send = None
        with system.detection_lock:
            if system.latest_frame is not None:
                frame_to_send = system.latest_frame.copy() # Copy to operate outside lock
        
        if frame_to_send is None:
            # Placeholder frame if system is stopped or frame not ready
            black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            status_text = "System Initializing..."
            if system.stats: # Check if stats initialized
                status_text = system.stats.get("detection_status", "Initializing...")
                if not system.running and system.stats.get("model_status", "").startswith("Error"):
                    status_text = f"Model Error: {system.stats['model_status'].split(': ',1)[-1]}"
                elif not system.running and system.stats.get("detection_status", "").startswith("Stopped (Source Error)"):
                     status_text = "Video Source Error"


            cv2.putText(black_frame, status_text, (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', black_frame)
        else:
            ret, buffer = cv2.imencode('.jpg', frame_to_send)

        if not ret:
            time.sleep(0.05) # Wait a bit if encoding fails
            continue
        
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Let the client pull frames as fast as it can, or sync with detection rate
        # time.sleep(1 / system.TARGET_STREAM_FPS if system.TARGET_STREAM_FPS > 0 else 0.03)
        # For now, minimal sleep to yield control, actual FPS depends on processing
        time.sleep(0.01) 

@app.route('/')
def index():
    return render_template('index.html', current_source=str(system.video_source))

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start', methods=['POST'])
def start_detection_route():
    if system.running:
        return jsonify({"message": "Detection already running.", "status": "already_running"}), 400
    
    data = request.get_json()
    new_source = data.get('source', system.video_source)
    if str(new_source) != str(system.video_source): # Check if source actually changed
        if not system.update_video_source(new_source):
            # update_video_source should not fail if system is not running, but good practice
            return jsonify({"message": "Failed to update source before starting.", "status": "error"}), 500

    # system.start_detection() itself is blocking until source is opened or fails.
    # The detection_loop then runs in a new thread.
    # We can run start_detection in a thread to make this route non-blocking.
    threading.Thread(target=system.start_detection, daemon=True).start()
    # Give a moment for the system.running flag and initial status to be set
    time.sleep(0.5) 
    if system.running:
        return jsonify({"message": "Detection starting...", "status": "starting"})
    else:
        # If system.running is still false, it means start_detection failed (e.g. model load, video source)
        # The specific error should be in system.stats or logs
        error_message = system.stats.get("detection_status", "Failed to start (Unknown error)")
        if "Model Load Failed" in error_message or "Source Error" in error_message:
             # Add more specific message from stats if available
            specific_error = system.stats.get("model_status", "") if "Model" in error_message else error_message
            return jsonify({"message": f"Failed to start: {specific_error}", "status": "error"}), 500
        return jsonify({"message": "Failed to start detection.", "status": "error"}), 500


@app.route('/stop', methods=['POST'])
def stop_detection_route():
    if not system.running:
        return jsonify({"message": "Detection not running.", "status": "already_stopped"}), 400
    
    system.stop_detection() # This will set self.running = False and join the thread.
    return jsonify({"message": "Detection stopping...", "status": "stopping"})

@app.route('/status', methods=['GET'])
def get_status_route():
    return jsonify(system.get_status())

@app.route('/update_source', methods=['POST'])
def update_source_route():
    if system.running:
        return jsonify({"success": False, "message": "Stop detection before changing source."}), 400
    
    data = request.get_json()
    new_source = data.get('source')
    if new_source is None or str(new_source).strip() == "": # Check for empty or None source
        return jsonify({"success": False, "message": "No source provided."}), 400
    
    if system.update_video_source(str(new_source)): # Ensure source is string
         return jsonify({"success": True, "message": f"Source updated to {new_source}."})
    else:
         # This case should ideally not be reached if system.running is false
         return jsonify({"success": False, "message": "Failed to update source (unexpected)."}), 500


if __name__ == "__main__":
    print(f"Flask app starting... Default video source: {DEFAULT_VIDEO_SOURCE}")
    print("Open http://127.0.0.1:5000 (or your server's IP) in your browser.")
    # use_reloader=False is crucial for threaded applications to avoid issues with model loading and thread management
    # threaded=True allows Flask to handle multiple requests concurrently (though Python's GIL limits true parallelism for CPU-bound tasks)
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False, threaded=True)
