# 基础版本，可以简单在终端显示结果。

import cv2
import time
import numpy as np
from ultralytics import YOLO

# --- Configuration ---
# Replace with RTSP stream URL
# RTSP_URL = "rtsp://your_rtsp_stream_url_here"  
# Use a local video file for testing:
RTSP_URL = "video2.mp4"

# Use a YOLOv8 pose model
POSE_MODEL_PATH = 'yolov8x-pose.pt' # Downloads automatically if not present

# Stillness Detection Parameters
STILLNESS_DURATION_THRESHOLD_SEC = 180  # Seconds a person must be still to trigger an alert
MOVEMENT_THRESHOLD_PIXELS = 15       # Max average keypoint movement to be considered "still"

# Fainting Detection Parameters
FAINT_DETECTION_WINDOW_SEC = 1.5     # Time window (seconds) to detect a rapid fall
MIN_VERTICAL_DROP_RATIO = 0.25       # Minimum drop relative to person's height (e.g., 25% of height)
HORIZONTAL_ASPECT_RATIO_THRESHOLD = 1.2 # If keypoint width/height > this, considered horizontal

# Tracking Parameters
IOU_THRESHOLD = 0.3                  # Minimum IOU to match a detection to an existing track
MAX_ABSENT_FRAMES = 15               # Max frames to keep a track if person is not detected

# Keypoint indices (COCO format, used by YOLOv8-Pose)
# 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
# 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
# 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
# 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
UPPER_BODY_KP_INDICES = [0, 5, 6, 11, 12] # Nose, Shoulders, Hips

# --- Helper Functions ---
def get_iou(boxA, boxB):
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the intersection over union
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# --- Main Application ---
def main():
    print(f"Loading YOLOv8 pose model: {POSE_MODEL_PATH}")
    model = YOLO(POSE_MODEL_PATH)
    # model.to('cuda')
    print("Model loaded.")

    print(f"Attempting to open RTSP stream: {RTSP_URL}")
    cap = cv2.VideoCapture(RTSP_URL)

    if not cap.isOpened():
        print(f"Error: Could not open video stream from {RTSP_URL}")
        print("Please check the URL, network connectivity, and camera availability.")
        return

    print("Successfully opened video stream.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Warning: Could not get FPS from stream, defaulting to 20 FPS for calculations.")
        fps = 20 # Default FPS if not available

    stillness_frames_threshold = int(STILLNESS_DURATION_THRESHOLD_SEC * fps)
    faint_detection_frames = int(FAINT_DETECTION_WINDOW_SEC * fps)
    if faint_detection_frames < 2: faint_detection_frames = 2 # Need at least 2 frames

    tracked_persons = {}  # Stores state for each tracked person
    next_person_id = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame or stream ended.")
            # Optional: Try to reconnect
            # cap.release()
            # cap = cv2.VideoCapture(RTSP_URL)
            # if not cap.isOpened():
            #     print("Reconnect failed.")
            #     break
            # else:
            #     print("Reconnected.")
            #     continue
            break

        frame_count += 1
        prev_time = 0
        current_time = time.time()

#添加fps说明
        fps_text = f"FPS: {1 / (current_time - prev_time):.2f}" if prev_time else "FPS: --"
        prev_time = current_time
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # YOLO expects RGB

        # Perform detection and pose estimation
        results = model(frame_rgb, verbose=False, classes=[0]) # Class 0 is 'person'

        current_detections = [] # Store (bbox, keypoints, conf) for persons in this frame
        if results and results[0].boxes is not None and results[0].keypoints is not None:
            for i in range(len(results[0].boxes)):
                box = results[0].boxes[i]
                if box.cls == 0 and box.conf > 0.4: # Person class with confidence
                    bbox = list(map(int, box.xyxy[0].tolist()))
                    keypoints = results[0].keypoints.xy[i].cpu().numpy() # (N_kpts, 2)
                    # Keypoint confidence (optional, but good for filtering)
                    # keypoints_conf = results[0].keypoints.conf[i].cpu().numpy() # (N_kpts,)
                    current_detections.append({'bbox': bbox, 'keypoints': keypoints, 'id': -1, 'conf': box.conf})

        # --- Simple IOU Tracking ---
        unmatched_detections = list(range(len(current_detections)))
        matched_track_ids = set()

        # Try to match existing tracks
        for person_id, p_data in tracked_persons.items():
            best_match_idx = -1
            max_iou = IOU_THRESHOLD
            for i, det in enumerate(current_detections):
                if i in unmatched_detections: # Only consider unmatched detections
                    iou = get_iou(p_data['bbox'], det['bbox'])
                    if iou > max_iou:
                        max_iou = iou
                        best_match_idx = i
            
            if best_match_idx != -1:
                current_detections[best_match_idx]['id'] = person_id
                p_data.update(current_detections[best_match_idx]) # Update track with new data
                p_data['last_seen_frame'] = frame_count
                p_data['absent_frames'] = 0
                unmatched_detections.remove(best_match_idx)
                matched_track_ids.add(person_id)
            else:
                p_data['absent_frames'] += 1 # Mark as absent if not matched

        # Add new tracks for unmatched detections
        for i in unmatched_detections:
            new_id = next_person_id
            next_person_id += 1
            current_detections[i]['id'] = new_id
            tracked_persons[new_id] = {
                **current_detections[i],
                'last_seen_frame': frame_count,
                'absent_frames': 0,
                'still_start_time': current_time, # Assume still until moved
                'y_coords_history': [],           # For faint detection
                'alert_still': False,
                'alert_faint': False,
                'is_fainted_state': False
            }
            matched_track_ids.add(new_id)

        # Remove old tracks
        ids_to_remove = [pid for pid, pdata in tracked_persons.items() if pdata['absent_frames'] > MAX_ABSENT_FRAMES]
        for pid in ids_to_remove:
            print(f"Removing track for person ID {pid}")
            del tracked_persons[pid]

        # --- Process each tracked person for stillness and fainting ---
        for person_id, p_data in tracked_persons.items():
            if person_id not in matched_track_ids and p_data['absent_frames'] <= MAX_ABSENT_FRAMES : # If person was not detected in current frame but track is kept
                continue # Skip processing for this frame

            bbox = p_data['bbox']
            keypoints = p_data['keypoints']
            person_height_estimate = bbox[3] - bbox[1] # y2 - y1

            # --- Stillness Detection ---
            if 'prev_keypoints' in p_data:
                # Calculate average movement of keypoints
                # Only use visible keypoints (YOLO might return 0,0 for non-visible)
                visible_curr_kpts = keypoints[keypoints[:,0] > 0 & (keypoints[:,1] > 0)]
                visible_prev_kpts = p_data['prev_keypoints'][p_data['prev_keypoints'][:,0] > 0 & (p_data['prev_keypoints'][:,1] > 0)]

                if len(visible_curr_kpts) > 0 and len(visible_prev_kpts) > 0:
                    # Ensure consistent number of keypoints for comparison (e.g., by matching indices)
                    # For simplicity, we'll use a rough approach or assume full visibility
                    # A better way is to align based on keypoint indices
                    min_len = min(len(visible_curr_kpts), len(visible_prev_kpts))
                    # This alignment is naive, proper alignment would use indices
                    movement_distances = np.linalg.norm(visible_curr_kpts[:min_len] - visible_prev_kpts[:min_len], axis=1)
                    avg_movement = np.mean(movement_distances) if len(movement_distances) > 0 else MOVEMENT_THRESHOLD_PIXELS + 1

                    if avg_movement < MOVEMENT_THRESHOLD_PIXELS:
                        if p_data['still_start_time'] is None:
                            p_data['still_start_time'] = current_time
                    else: # Moved
                        p_data['still_start_time'] = None
                        p_data['alert_still'] = False # Reset alert if moved
                else: # Not enough keypoints to determine movement
                     p_data['still_start_time'] = None # Can't confirm stillness
                     p_data['alert_still'] = False

            p_data['prev_keypoints'] = keypoints.copy()

            # --- Fainting Detection ---
            # Get average Y of upper body keypoints
            valid_upper_body_kpts_y = [keypoints[j, 1] for j in UPPER_BODY_KP_INDICES if keypoints[j,0]>0 and keypoints[j,1]>0]
            
            if valid_upper_body_kpts_y:
                current_avg_y = np.mean(valid_upper_body_kpts_y)
                p_data['y_coords_history'].append((current_time, current_avg_y))
                # Keep history within the FAINT_DETECTION_WINDOW_SEC
                p_data['y_coords_history'] = [
                    (t, y_val) for t, y_val in p_data['y_coords_history']
                    if current_time - t <= FAINT_DETECTION_WINDOW_SEC
                ]

                # Check for fall if enough history and person is tall enough
                if len(p_data['y_coords_history']) >= faint_detection_frames / 2 and faint_detection_frames > 0 and person_height_estimate > 50:
                    oldest_time, oldest_y = p_data['y_coords_history'][0]
                    vertical_drop = current_avg_y - oldest_y # Positive if dropped (Y increases downwards)

                    if vertical_drop > (MIN_VERTICAL_DROP_RATIO * person_height_estimate):
                        # Check posture: are they horizontal?
                        # Get bounding box of all *visible* keypoints
                        visible_kpts = keypoints[keypoints[:,0]>0 & (keypoints[:,1]>0)]
                        if len(visible_kpts) > 3: # Need at least a few keypoints
                            kpt_min_x, kpt_min_y = np.min(visible_kpts, axis=0)
                            kpt_max_x, kpt_max_y = np.max(visible_kpts, axis=0)
                            kpt_width = kpt_max_x - kpt_min_x
                            kpt_height = kpt_max_y - kpt_min_y

                            if kpt_height > 0 and kpt_width / kpt_height > HORIZONTAL_ASPECT_RATIO_THRESHOLD:
                                if not p_data['alert_faint']: # Alert only once
                                    print(f"ALERT: Person ID {p_data['id']} FAINTED! (Rapid drop & horizontal)")
                                    p_data['alert_faint'] = True
                                    p_data['is_fainted_state'] = True
                                    p_data['still_start_time'] = current_time # Fainted person is also still
                            else: # Dropped but not horizontal (e.g. sitting down fast)
                                if p_data['is_fainted_state']: # If was fainted, but now not horizontal, reset
                                    p_data['alert_faint'] = False
                                    p_data['is_fainted_state'] = False
                        # else: no posture check possible
                    else: # No significant drop
                        if p_data['is_fainted_state']: # If was fainted, but no longer dropping, reset (might be getting up)
                             p_data['alert_faint'] = False
                             p_data['is_fainted_state'] = False
            # else: not enough upper body keypoints visible

            # --- Alerting and Drawing ---
            label_y_offset = 15
            if p_data['is_fainted_state']: # Check internal state not just alert flag
                cv2.putText(frame, f"ID {p_data['id']}: FAINTED", (bbox[0], bbox[1] - label_y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                label_y_offset += 20
            elif p_data['still_start_time'] and (current_time - p_data['still_start_time']) > STILLNESS_DURATION_THRESHOLD_SEC:
                if not p_data['alert_still']: # Alert only once per stillness period
                    print(f"ALERT: Person ID {p_data['id']} STILL for too long!")
                    p_data['alert_still'] = True
                cv2.putText(frame, f"ID {p_data['id']}: STILL TOO LONG", (bbox[0], bbox[1] - label_y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                label_y_offset += 20
            
            # Draw bounding box for the person
            color = (0, 0, 255) if p_data['is_fainted_state'] else \
                    ((0, 165, 255) if p_data['alert_still'] else (0, 255, 0))
            
            line_thickness = max(2, int(0.002 * frame.shape[1]))
            font_scale = 0.5 + 0.001 * frame.shape[1]
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2, line_thickness)
            cv2.putText(frame, f"ID {p_data['id']}", (bbox[0], max(15, bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, line_thickness)
            # cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            # cv2.putText(frame, f"ID: {p_data['id']}", (bbox[0], bbox[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Draw keypoints
            for k_idx, (px, py) in enumerate(keypoints):
                if px > 0 and py > 0: # Draw only if keypoint is detected (not 0,0)
                    cv2.circle(frame, (int(px), int(py)), 3, (255, 0, 0), -1)
                    # cv2.putText(frame, str(k_idx), (int(px)+5, int(py)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200,200,200), 1)

        display_frame = cv2.resize(frame,(1024,576))
        cv2.imshow("Worker Safety Monitoring", display_frame)

        # out_writer.write(frame)
        # cv2.imshow("Worker Safety Monitoring", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Application finished.")

if __name__ == "__main__":
    main()