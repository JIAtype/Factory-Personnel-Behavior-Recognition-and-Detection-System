# 用python实现工厂人员行为识别检测系统，使用yolov8x-pose.pt模型。通过监控RTSP_URL = \"rtsp://your_rtsp_stream_url_here\"获取到画面，检测是否有人摔倒或者超过3分钟一动不动。
# 系统会同步显示监控画面，在画面标出检测到的人员。如果出现特殊情况就在系统中显示，并且弹出对应警报，并保存出现这种情况时的监控画面（5分钟）。
# 系统要用英文，界面要显示必要信息。"

# 我将为您创建一个基于YOLOv8的工厂人员行为识别检测系统。这个系统能够检测人员摔倒和长时间静止不动的情况。

# 我为您创建了一个完整的工厂人员行为识别检测系统。这个系统具有以下主要功能：

# ## 核心功能：

# 1. **实时视频监控**：通过RTSP流获取监控画面
# 2. **人员检测与跟踪**：使用YOLOv8x-pose模型检测人员骨架关键点
# 3. **行为分析**：
#    - 摔倒检测：基于身体姿态角度和比例分析
#    - 静止检测：超过3分钟无明显移动
# 4. **可视化显示**：在画面上标出检测到的人员和状态
# 5. **警报系统**：弹出警报窗口并记录警报信息
# 6. **录像保存**：检测到异常时自动保存5分钟监控画面

# ## 系统界面包含：

# - **实时视频显示区域**
# - **控制面板**：启动/停止检测
# - **统计信息**：实时显示检测到的人数、摔倒人数、静止人数
# - **警报历史**：显示最近的警报记录
# - **设置面板**：可配置RTSP URL

# ## 使用前准备：

# 1. **安装依赖包**：
# ```bash
# pip install ultralytics opencv-python pillow numpy
# ```

# 2. **下载YOLOv8模型**：
# ```bash
# # 系统会自动下载yolov8x-pose.pt模型
# ```

# 3. **配置RTSP URL**：
#    - 将代码中的 `"rtsp://your_rtsp_stream_url_here"` 替换为实际的RTSP流地址
#    - 或在运行后通过界面设置面板修改

# ## 检测算法：

# - **摔倒检测**：分析身体关键点的相对位置，检测异常姿态
# - **静止检测**：跟踪人员移动轨迹，超过3分钟无移动触发警报
# - **人员跟踪**：基于位置相似度的简单跟踪算法

# ## 输出文件：

# - 警报录像保存在 `alerts/` 目录
# - 每个警报包含5分钟的监控录像和对应的JSON信息文件

# 系统采用多线程设计，确保实时性能和用户界面响应。您可以根据实际需求调整检测阈值和参数。

import cv2
import numpy as np
import time
import threading
from datetime import datetime, timedelta
import os
import json
from collections import defaultdict, deque
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import queue
import math

try:
    from ultralytics import YOLO
except ImportError:
    print("Please install ultralytics: pip install ultralytics")
    exit(1)

class PersonTracker:
    def __init__(self, person_id, keypoints):
        self.id = person_id
        self.keypoints_history = deque(maxlen=30)  # Store last 30 frames
        self.last_movement_time = time.time()
        self.is_fallen = False
        self.is_stationary = False
        self.keypoints_history.append(keypoints)
        
    def update(self, keypoints):
        self.keypoints_history.append(keypoints)
        
        # Check for movement
        if len(self.keypoints_history) >= 2:
            movement = self.calculate_movement()
            if movement > 10:  # Threshold for movement detection
                self.last_movement_time = time.time()
                self.is_stationary = False
            else:
                # Check if stationary for more than 3 minutes
                if time.time() - self.last_movement_time > 180:  # 3 minutes
                    self.is_stationary = True
        
        # Check for fall detection
        self.is_fallen = self.detect_fall()
    
    def calculate_movement(self):
        if len(self.keypoints_history) < 2:
            return 0
        
        current = self.keypoints_history[-1]
        previous = self.keypoints_history[-2]
        
        total_movement = 0
        valid_points = 0
        
        for i in range(len(current)):
            if current[i][2] > 0.5 and previous[i][2] > 0.5:  # Confidence threshold
                dx = current[i][0] - previous[i][0]
                dy = current[i][1] - previous[i][1]
                movement = math.sqrt(dx*dx + dy*dy)
                total_movement += movement
                valid_points += 1
        
        return total_movement / max(valid_points, 1)
    
    def detect_fall(self):
        if not self.keypoints_history:
            return False
        
        keypoints = self.keypoints_history[-1]
        
        # Get key body parts (COCO pose format)
        # 0: nose, 5: left_shoulder, 6: right_shoulder, 11: left_hip, 12: right_hip
        nose = keypoints[0] if keypoints[0][2] > 0.5 else None
        left_shoulder = keypoints[5] if keypoints[5][2] > 0.5 else None
        right_shoulder = keypoints[6] if keypoints[6][2] > 0.5 else None
        left_hip = keypoints[11] if keypoints[11][2] > 0.5 else None
        right_hip = keypoints[12] if keypoints[12][2] > 0.5 else None
        
        # Calculate body orientation
        if left_shoulder and right_shoulder and left_hip and right_hip:
            # Calculate torso angle
            shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
            hip_center_y = (left_hip[1] + right_hip[1]) / 2
            
            # If hips are higher than shoulders, likely fallen
            if hip_center_y < shoulder_center_y:
                return True
            
            # Calculate body width vs height ratio
            body_width = abs(left_shoulder[0] - right_shoulder[0])
            body_height = abs(shoulder_center_y - hip_center_y)
            
            if body_height > 0:
                aspect_ratio = body_width / body_height
                # If aspect ratio is too high, person might be lying down
                if aspect_ratio > 2.0:
                    return True
        
        return False

class BehaviorDetectionSystem:
    def __init__(self, rtsp_url="rtsp://your_rtsp_stream_url_here"):
        self.rtsp_url = rtsp_url
        self.model = None
        self.cap = None
        self.trackers = {}
        self.next_id = 0
        self.running = False
        self.frame_queue = queue.Queue(maxsize=10)
        self.alert_queue = queue.Queue()
        self.recording_buffer = deque(maxlen=1500)  # 5 minutes at 5 fps
        self.is_recording_alert = False
        self.alert_start_time = None
        
        # Create output directories
        os.makedirs("alerts", exist_ok=True)
        os.makedirs("recordings", exist_ok=True)
        
        # Initialize GUI
        self.setup_gui()
        
    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Factory Personnel Behavior Detection System")
        self.root.geometry("1200x800")
        
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Video frame
        video_frame = ttk.LabelFrame(main_frame, text="Live Video Feed")
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.video_label = ttk.Label(video_frame)
        self.video_label.pack(padx=10, pady=10)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Control Panel")
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        
        # Status
        status_frame = ttk.LabelFrame(control_frame, text="System Status")
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.status_label = ttk.Label(status_frame, text="Stopped", foreground="red")
        self.status_label.pack(pady=5)
        
        # Controls
        ttk.Button(control_frame, text="Start Detection", command=self.start_detection).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Stop Detection", command=self.stop_detection).pack(fill=tk.X, pady=2)
        
        # Statistics
        stats_frame = ttk.LabelFrame(control_frame, text="Detection Statistics")
        stats_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.people_count_label = ttk.Label(stats_frame, text="People Detected: 0")
        self.people_count_label.pack(anchor=tk.W)
        
        self.fallen_count_label = ttk.Label(stats_frame, text="Fallen: 0")
        self.fallen_count_label.pack(anchor=tk.W)
        
        self.stationary_count_label = ttk.Label(stats_frame, text="Stationary: 0")
        self.stationary_count_label.pack(anchor=tk.W)
        
        # Alerts
        alerts_frame = ttk.LabelFrame(control_frame, text="Recent Alerts")
        alerts_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        self.alerts_text = tk.Text(alerts_frame, height=15, width=30)
        alerts_scrollbar = ttk.Scrollbar(alerts_frame, orient=tk.VERTICAL, command=self.alerts_text.yview)
        self.alerts_text.configure(yscrollcommand=alerts_scrollbar.set)
        
        self.alerts_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        alerts_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Settings
        settings_frame = ttk.LabelFrame(control_frame, text="Settings")
        settings_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(settings_frame, text="RTSP URL:").pack(anchor=tk.W)
        self.rtsp_entry = ttk.Entry(settings_frame, width=25)
        self.rtsp_entry.insert(0, self.rtsp_url)
        self.rtsp_entry.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(settings_frame, text="Update URL", command=self.update_rtsp_url).pack(fill=tk.X)
        
    def update_rtsp_url(self):
        self.rtsp_url = self.rtsp_entry.get()
        
    def load_model(self):
        try:
            self.model = YOLO('yolov8x-pose.pt')
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            return False
    
    def start_detection(self):
        if self.running:
            return
            
        if not self.load_model():
            return
            
        # Initialize video capture
        self.cap = cv2.VideoCapture(self.rtsp_url)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Failed to connect to video stream")
            return
        
        self.running = True
        self.status_label.config(text="Running", foreground="green")
        
        # Start detection thread
        self.detection_thread = threading.Thread(target=self.detection_loop)
        self.detection_thread.daemon = True
        self.detection_thread.start()
        
        # Start GUI update thread
        self.gui_thread = threading.Thread(target=self.update_gui)
        self.gui_thread.daemon = True
        self.gui_thread.start()
        
        # Start alert handler thread
        self.alert_thread = threading.Thread(target=self.handle_alerts)
        self.alert_thread.daemon = True
        self.alert_thread.start()
    
    def stop_detection(self):
        self.running = False
        self.status_label.config(text="Stopped", foreground="red")
        
        if self.cap:
            self.cap.release()
        
        self.trackers.clear()
    
    def detection_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Add frame to recording buffer
            self.recording_buffer.append((time.time(), frame.copy()))
            
            # Run YOLO detection
            results = self.model(frame)
            
            # Process detections
            current_detections = []
            for r in results:
                if r.keypoints is not None:
                    keypoints = r.keypoints.data.cpu().numpy()
                    boxes = r.boxes.data.cpu().numpy()
                    
                    for i, (kpts, box) in enumerate(zip(keypoints, boxes)):
                        if box[4] > 0.5:  # Confidence threshold
                            current_detections.append({
                                'keypoints': kpts,
                                'box': box,
                                'id': None
                            })
            
            # Update trackers
            self.update_trackers(current_detections)
            
            # Draw detections and check for alerts
            annotated_frame = self.draw_annotations(frame)
            
            # Add frame to queue for GUI
            if not self.frame_queue.full():
                self.frame_queue.put(annotated_frame)
            
            time.sleep(0.1)  # Limit processing rate
    
    def update_trackers(self, detections):
        # Simple tracking based on proximity
        used_detections = set()
        
        # Update existing trackers
        for tracker_id, tracker in list(self.trackers.items()):
            best_match = None
            best_distance = float('inf')
            
            for i, detection in enumerate(detections):
                if i in used_detections:
                    continue
                
                # Calculate distance between tracker and detection
                distance = self.calculate_detection_distance(tracker, detection)
                if distance < best_distance and distance < 100:  # Distance threshold
                    best_distance = distance
                    best_match = i
            
            if best_match is not None:
                detection = detections[best_match]
                tracker.update(detection['keypoints'])
                detection['id'] = tracker_id
                used_detections.add(best_match)
            else:
                # Remove tracker if no match found for several frames
                del self.trackers[tracker_id]
        
        # Create new trackers for unmatched detections
        for i, detection in enumerate(detections):
            if i not in used_detections:
                new_tracker = PersonTracker(self.next_id, detection['keypoints'])
                self.trackers[self.next_id] = new_tracker
                detection['id'] = self.next_id
                self.next_id += 1
    
    def calculate_detection_distance(self, tracker, detection):
        if not tracker.keypoints_history:
            return float('inf')
        
        last_keypoints = tracker.keypoints_history[-1]
        current_keypoints = detection['keypoints']
        
        # Calculate center of mass distance
        last_center = np.mean(last_keypoints[last_keypoints[:, 2] > 0.5][:, :2], axis=0)
        current_center = np.mean(current_keypoints[current_keypoints[:, 2] > 0.5][:, :2], axis=0)
        
        return np.linalg.norm(last_center - current_center)
    
    def draw_annotations(self, frame):
        annotated_frame = frame.copy()
        
        fallen_count = 0
        stationary_count = 0
        
        for tracker_id, tracker in self.trackers.items():
            if not tracker.keypoints_history:
                continue
            
            keypoints = tracker.keypoints_history[-1]
            
            # Draw skeleton
            self.draw_skeleton(annotated_frame, keypoints)
            
            # Draw bounding box and status
            x_coords = keypoints[keypoints[:, 2] > 0.5][:, 0]
            y_coords = keypoints[keypoints[:, 2] > 0.5][:, 1]
            
            if len(x_coords) > 0 and len(y_coords) > 0:
                x1, y1 = int(x_coords.min()), int(y_coords.min())
                x2, y2 = int(x_coords.max()), int(y_coords.max())
                
                # Color based on status
                color = (0, 255, 0)  # Green for normal
                status_text = f"Person {tracker_id}: Normal"
                
                if tracker.is_fallen:
                    color = (0, 0, 255)  # Red for fallen
                    status_text = f"Person {tracker_id}: FALLEN"
                    fallen_count += 1
                    self.trigger_alert(f"FALL DETECTED - Person {tracker_id}", tracker_id)
                elif tracker.is_stationary:
                    color = (0, 165, 255)  # Orange for stationary
                    status_text = f"Person {tracker_id}: STATIONARY"
                    stationary_count += 1
                    self.trigger_alert(f"STATIONARY ALERT - Person {tracker_id}", tracker_id)
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, status_text, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Update statistics
        self.update_statistics(len(self.trackers), fallen_count, stationary_count)
        
        return annotated_frame
    
    def draw_skeleton(self, frame, keypoints):
        # COCO pose skeleton connections
        skeleton = [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
            [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
            [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
            [2, 4], [3, 5], [4, 6], [5, 7]
        ]
        
        for connection in skeleton:
            kpt_a, kpt_b = connection
            if kpt_a-1 < len(keypoints) and kpt_b-1 < len(keypoints):
                x1, y1, c1 = keypoints[kpt_a-1]
                x2, y2, c2 = keypoints[kpt_b-1]
                
                if c1 > 0.5 and c2 > 0.5:
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        # Draw keypoints
        for kpt in keypoints:
            x, y, conf = kpt
            if conf > 0.5:
                cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)
    
    def trigger_alert(self, message, person_id):
        current_time = datetime.now()
        alert_data = {
            'time': current_time,
            'message': message,
            'person_id': person_id
        }
        
        if not self.alert_queue.full():
            self.alert_queue.put(alert_data)
        
        # Start recording if not already recording
        if not self.is_recording_alert:
            self.is_recording_alert = True
            self.alert_start_time = time.time()
            
            # Save recording in background thread
            recording_thread = threading.Thread(target=self.save_alert_recording, args=(message,))
            recording_thread.daemon = True
            recording_thread.start()
    
    def save_alert_recording(self, alert_message):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"alerts/alert_{timestamp}.avi"
        
        # Collect frames for 5 minutes
        frames_to_save = []
        start_time = time.time()
        
        while time.time() - start_time < 300:  # 5 minutes
            if self.recording_buffer:
                frame_time, frame = self.recording_buffer[-1]
                frames_to_save.append(frame)
            time.sleep(1)
        
        # Save video
        if frames_to_save:
            height, width = frames_to_save[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(filename, fourcc, 1.0, (width, height))
            
            for frame in frames_to_save:
                out.write(frame)
            
            out.release()
            
            # Save alert info
            alert_info = {
                'timestamp': timestamp,
                'message': alert_message,
                'video_file': filename,
                'duration_minutes': 5
            }
            
            with open(f"alerts/alert_{timestamp}.json", 'w') as f:
                json.dump(alert_info, f, indent=2, default=str)
        
        self.is_recording_alert = False
    
    def handle_alerts(self):
        while self.running:
            try:
                alert = self.alert_queue.get(timeout=1)
                
                # Show popup alert
                self.root.after(0, lambda: messagebox.showwarning(
                    "ALERT", 
                    f"{alert['message']}\nTime: {alert['time'].strftime('%Y-%m-%d %H:%M:%S')}"
                ))
                
                # Add to alerts display
                alert_text = f"[{alert['time'].strftime('%H:%M:%S')}] {alert['message']}\n"
                self.root.after(0, lambda text=alert_text: self.add_alert_text(text))
                
            except queue.Empty:
                continue
    
    def add_alert_text(self, text):
        self.alerts_text.insert(tk.END, text)
        self.alerts_text.see(tk.END)
        
        # Keep only last 50 lines
        lines = self.alerts_text.get(1.0, tk.END).split('\n')
        if len(lines) > 50:
            self.alerts_text.delete(1.0, f"{len(lines)-50}.0")
    
    def update_statistics(self, people_count, fallen_count, stationary_count):
        self.root.after(0, lambda: [
            self.people_count_label.config(text=f"People Detected: {people_count}"),
            self.fallen_count_label.config(text=f"Fallen: {fallen_count}"),
            self.stationary_count_label.config(text=f"Stationary: {stationary_count}")
        ])
    
    def update_gui(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1)
                
                # Convert frame to PhotoImage
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                
                # Resize frame to fit display
                display_width = 640
                display_height = 480
                frame_pil = frame_pil.resize((display_width, display_height), Image.Resampling.LANCZOS)
                
                photo = ImageTk.PhotoImage(frame_pil)
                
                # Update GUI in main thread
                self.root.after(0, lambda: self.video_label.configure(image=photo))
                self.root.after(0, lambda: setattr(self.video_label, 'image', photo))
                
            except queue.Empty:
                continue
    
    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def on_closing(self):
        self.stop_detection()
        self.root.destroy()

if __name__ == "__main__":
    RTSP_URL = "rtsp://your_rtsp_stream_url_here"
    
    app = BehaviorDetectionSystem(RTSP_URL)
    app.run()