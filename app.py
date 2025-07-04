from flask import Flask, render_template, Response, request, jsonify
import threading
import time
import cv2
import numpy as np
from PIL import Image 
try:
    from ultralytics import YOLO
except ImportError:
    print("Error: 'ultralytics' library not found. Please install it via 'pip install ultralytics'.")
    exit(1)

from opcua import ua, Server

import os
import json
import math
from collections import deque
from datetime import datetime

opcua_server = None
opcua_fall_variable = None
opcua_helmet_variable = None
OPCUA_SERVER_ENDPOINT = "opc.tcp://172.30.32.231:4841/SafetyServer/"
OPCUA_NAMESPACE_URI = "http://mycompany.com/safety_alerts"

def start_opcua_server():
    """
    此函数在一个独立的后台线程中启动并运行OPC UA服务器。
    它负责初始化服务器、创建数据节点，并持续运行以等待客户端连接和数据更新。
    """
    global opcua_server, opcua_fall_variable, opcua_helmet_variable
    
    try:
        opcua_server = Server()
        
        opcua_server.set_endpoint(OPCUA_SERVER_ENDPOINT)
        opcua_server.set_server_name("Safety Alert OPC UA Server")
        
        idx = opcua_server.register_namespace(OPCUA_NAMESPACE_URI)

        alerts_obj = opcua_server.nodes.objects.add_object(idx, "SafetyAlerts")
        
        opcua_fall_variable = alerts_obj.add_variable(idx, "FallStatus", 0)
        opcua_fall_variable.set_writable()

        opcua_helmet_variable = alerts_obj.add_variable(idx, "NoHelmetStatus", 0)
        opcua_helmet_variable.set_writable()

        opcua_server.start()
        print(f"✅ OPC UA server started successfully on: {OPCUA_SERVER_ENDPOINT}")
        
        while True:
            time.sleep(1)
            
    except Exception as e:
        print(f"❌ Failed to start OPC UA server: {e}")
    finally:
        if opcua_server:
            opcua_server.stop()
            print("The OPC UA server is stopped.")

class PersonTracker:
    """
    此类负责追踪单个人的状态和行为。
    每个人被检测到时，都会创建一个此类的实例，并赋予一个唯一的ID。
    它记录了人的关键点历史、运动状态、是否摔倒、是否佩戴安全帽等信息。
    """

    KEYPOINTS_HISTORY_MAXLEN = 30     # 保留最近30帧的关键点历史，用于分析和防止内存溢出。
    MOVEMENT_THRESHOLD_PIXELS = 10    # 关键点平均移动超过此像素值，才被认为是“有效移动”。
    STATIONARY_DURATION_SECONDS = 180 # 超过此时间（3分钟）没有有效移动，则判定为“静止”。

    FALL_ASPECT_RATIO_THRESHOLD = 1.8 # 边界框的 宽度/高度 > 1.8，则可能摔倒。
    FALL_HIP_SHOULDER_TOLERANCE_PX = 20 # 允许臀部比肩膀低10个像素，仍视为直立。
    
    STATE_CONFIRM_FRAMES = 3          # 状态需要连续N帧被确认才算生效（例如，连续3帧检测到安全帽才确认佩戴）。
    STATE_DISAPPEAR_FRAMES = 3        # 状态连续N帧未被确认，则认为消失（例如，连续5帧未检测到安全帽才确认未佩戴）。

    MIN_STANDING_HEIGHT_PX = 130  # 最小站立高度
    MIN_HIP_KNEE_DISTANCE_PX = 30  # 臀部膝盖最小距离

    def __init__(self, person_id, keypoints, bbox):
        """
        初始化一个新的个人追踪器。
        - person_id: 分配的唯一ID。
        - keypoints: 第一次检测到的人体关键点数据。
        - bbox: 第一次检测到的人体边界框。
        """
        self.id = person_id
        self.keypoints_history = deque(maxlen=self.KEYPOINTS_HISTORY_MAXLEN) # 使用deque自动管理历史记录长度
        self.bbox = bbox  # 当前人的边界框 [x1, y1, x2, y2]

        self.is_fallen = False
        self.is_stationary = False
        self.has_helmet = True  # 默认为True，直到连续多帧检测不到才变为False
        
        self.fall_alert_sent = False
        self.stationary_alert_sent = False
        self.no_helmet_alert_sent = False

        self.last_update_time = time.time() # 记录此追踪器最后一次在帧中被更新的时间
        self.last_movement_time = time.time()
        self._no_helmet_counter = 0 # 内部计数器：连续未检测到安全帽的帧数
        self._has_helmet_counter = 0 # 内部计数器：连续检测到安全帽的帧数

        self.keypoints_history.append(keypoints)

    def update(self, keypoints, bbox):
        self.last_update_time = time.time()
        self.keypoints_history.append(keypoints)
        self.bbox = bbox

        if len(self.keypoints_history) >= 2:
            movement = self._calculate_movement()
            if movement > self.MOVEMENT_THRESHOLD_PIXELS:
                self.last_movement_time = time.time()
                self.is_stationary = False
                self.stationary_alert_sent = False
            else:
                if time.time() - self.last_movement_time > self.STATIONARY_DURATION_SECONDS:
                    self.is_stationary = True
        
        fall_detected_this_frame = self._detect_fall()
        if fall_detected_this_frame and not self.is_fallen:
            self.is_fallen = True # 状态从未摔倒变为摔倒
        elif not fall_detected_this_frame and self.is_fallen:
            self.is_fallen = False # 状态从摔倒恢复
            self.fall_alert_sent = False # 允许下一次摔倒时再次报警

    def update_helmet_status(self, detected_helmet_this_frame: bool):
        if detected_helmet_this_frame:
            self._has_helmet_counter += 1
            self._no_helmet_counter = 0  # 重置反向状态的计数器
        else:
            self._no_helmet_counter += 1
            self._has_helmet_counter = 0

        if self._has_helmet_counter >= self.STATE_CONFIRM_FRAMES:
            if not self.has_helmet:
                print(f"[INFORMATION] Tracker {self.id}: Status confirmation -> Safety helmet worn")
            self.has_helmet = True
            self.no_helmet_alert_sent = False
            
        elif self._no_helmet_counter >= self.STATE_DISAPPEAR_FRAMES:
            if self.has_helmet:
                print(f"[INFORMATION] Tracker {self.id}: Status confirmation -> Not wearing a helmet")
            self.has_helmet = False

    def _calculate_movement(self) -> float:
        if len(self.keypoints_history) < 2:
            return 0.0
        
        current_kpts = np.array(self.keypoints_history[-1])
        previous_kpts = np.array(self.keypoints_history[-2])
        
        valid_mask = (current_kpts[:, 2] > 0.5) & (previous_kpts[:, 2] > 0.5)
        
        if not np.any(valid_mask):
            return 0.0

        distances = np.linalg.norm(current_kpts[valid_mask, :2] - previous_kpts[valid_mask, :2], axis=1)
        
        return np.mean(distances)

    def _detect_fall(self) -> bool:
        if not self.keypoints_history: 
            return False
        
        kpts = np.array(self.keypoints_history[-1])
        
        keypoints_indices = {
            'left_shoulder': 5,
            'right_shoulder': 6,
            'left_hip': 11,
            'right_hip': 12,
            'head': 0,
            'left_knee': 13,
            'right_knee': 14
        }

        # l_shoulder_idx, r_shoulder_idx, l_hip_idx, r_hip_idx = 5, 6, 11, 12
        # head_idx, l_knee_idx, r_knee_idx = 0, 13, 14
        
        def get_kpt(idx):
            return kpts[idx][:2] if len(kpts) > idx and kpts[idx][2] > 0.5 else None

        # Extract keypoints
        left_shoulder, right_shoulder = get_kpt(keypoints_indices['left_shoulder']), get_kpt(keypoints_indices['right_shoulder'])
        left_hip, right_hip = get_kpt(keypoints_indices['left_hip']), get_kpt(keypoints_indices['right_hip'])
        head = get_kpt(keypoints_indices['head'])
        left_knee, right_knee = get_kpt(keypoints_indices['left_knee']), get_kpt(keypoints_indices['right_knee'])

        if not all(p is not None for p in [left_shoulder, right_shoulder, left_hip, right_hip]):
            return False

        shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
        hip_center_y = (left_hip[1] + right_hip[1]) / 2

        body_width = abs(left_shoulder[0] - right_shoulder[0])
        body_height = abs(shoulder_center_y - hip_center_y) + 1e-6
        aspect_ratio = body_width / body_height

        # if head is not None:
        #     total_body_height = abs(head[1] - hip_center_y)
        #     if total_body_height < self.MIN_STANDING_HEIGHT_PX:
        #         return True

        if left_knee is not None and right_knee is not None:
            knee_center_y = (left_knee[1] + right_knee[1]) / 2
            if abs(knee_center_y - hip_center_y) < self.MIN_HIP_KNEE_DISTANCE_PX:
                return True
            
        if aspect_ratio > self.FALL_ASPECT_RATIO_THRESHOLD:
                return True
                    
        return False

    def get_head_region(self) -> tuple or None:
        if not self.keypoints_history: return None
        kpts = np.array(self.keypoints_history[-1])

        head_indices = [0, 1, 2, 3, 4]
        visible_head_kpts = [kpts[i][:2] for i in head_indices if len(kpts) > i and kpts[i][2] > 0.4]
        
        if len(visible_head_kpts) >= 2: # 至少需要两个点才能估算
            visible_head_kpts_np = np.array(visible_head_kpts)
            min_x, min_y = np.min(visible_head_kpts_np, axis=0)
            max_x, max_y = np.max(visible_head_kpts_np, axis=0)

            face_width = max_x - min_x
            face_height = max_y - min_y

            if 15 < face_width < 250 and 15 < face_height < 250:
                center_x = min_x + face_width / 2
                center_y = min_y + face_height / 2
                base_size = max(face_width, face_height)
                box_width = base_size * 1.2 
                box_height = base_size * 1.5
                hx1 = int(center_x - box_width / 2)
                hy1 = int(center_y - box_height * 0.65) # 65%在上方
                hx2 = int(center_x + box_width / 2)
                hy2 = int(center_y + box_height * 0.35) # 35%在下方
                return max(0, hx1), max(0, hy1), hx2, hy2

        if self.bbox is not None:
            x1, y1, x2, y2 = self.bbox
            person_height = y2 - y1

            head_height = person_height * 0.30 

            person_width = x2 - x1
            head_width = person_width * 0.8

            head_center_x = x1 + person_width / 2
            
            head_x1 = int(head_center_x - head_width / 2)
            head_y1 = int(y1) # 头部从身体的顶端开始
            head_x2 = int(head_center_x + head_width / 2)
            head_y2 = int(y1 + head_height)
            
            return head_x1, head_y1, head_x2, head_y2
        
        return None

class BehaviorDetectionSystem:

    POSE_PERSON_CONF_THRESHOLD = 0.8    # 人体姿态检测的置信度阈值
    HELMET_CONF_THRESHOLD = 0.6       # 安全帽检测的置信度阈值
    HELMET_IOU_THRESHOLD = 0.3        # 头部区域与安全帽框的IoU（交并比）阈值，用于判断是否佩戴

    ALERT_COOLDOWN_SECONDS = 10       # 同一类型警报的冷却时间（秒），防止刷屏

    TRACKER_MAX_DISAPPEAR_SECONDS = 5 # 追踪目标消失超过此时长，则移除追踪器
    TRACKER_MAX_MATCH_DISTANCE_PX = 50 # 匹配新检测与旧追踪器的最大像素距离

    RECORDING_BUFFER_FRAMES = 300     # 录制缓冲区大小（帧）。例如，10 FPS下可缓冲30秒
    RECORDING_VIDEO_FPS = 10.0        # 保存的警报视频的帧率

    MIN_PERSON_HEIGHT_PIXELS = 200 # 新增：人的边界框高度必须大于此像素值，才被视为在有效工作区内

    def __init__(self, video_source):
        self.video_source = video_source
        self.pose_model = None
        self.helmet_model = None
        self.cap = None
        self.running = False

        self.trackers = {} # 存储所有 PersonTracker 实例，格式: {id: tracker_object}
        self.next_id = 0
        self.latest_frame = None # 存储处理后的最新帧，用于Web流
        self.detection_lock = threading.Lock() # 线程锁，保护共享数据（如latest_frame, stats, alerts）
        self.recording_buffer = deque(maxlen=self.RECORDING_BUFFER_FRAMES) # 存储最近的帧用于警报录制

        self.alerts = deque(maxlen=100) # 存储警报和系统日志
        self.stats = {
            "people_count": 0, "fallen_count": 0, "stationary_count": 0, "no_helmet_count": 0,
            "model_status": "Not loaded", "detection_status": "Stopped"
        }
        self.alert_cooldown_map = {} # 记录警报冷却时间戳, 格式: {(person_id, alert_type): timestamp}
        self.is_video_file = self._is_video_file_source()

        os.makedirs("alerts_data/recordings", exist_ok=True)
        os.makedirs("alerts_data/json", exist_ok=True)

    def _is_video_file_source(self):
        try:
            int(self.video_source)
            return False # 是数字，代表摄像头ID
        except ValueError:
            # 检查文件扩展名
            video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
            return any(str(self.video_source).lower().endswith(ext) for ext in video_exts)

    def load_models(self):
        self._log_message("[System] Loading AI model...")
        self.stats["model_status"] = "loading..."
        try:
            self.pose_model = YOLO('yolov8x-pose.pt').to('cuda')
            self.helmet_model = YOLO('best.pt').to('cuda')
            
            self.helmet_class_id = next(
                (k for k, v in self.helmet_model.names.items() if v.lower() == 'helmet'), None
            )
            if self.helmet_class_id is None:
                self._log_message("[WARNING] Class 'helmet' not found in helmet model. Helmet detection will not be available.")
            
            self._log_message("[System] AI model loaded successfully.")
            self.stats["model_status"] = "Loaded"
            return True
        except Exception as e:
            self._log_message(f"[ERROR] Failed to load model: {e}")
            self.stats["model_status"] = "Loading failed"
            self.pose_model = self.helmet_model = None
            return False

    def start_detection(self):
        if self.running:
            self._log_message("[INFORMATION] Test is already running.")
            return

        if not self.pose_model and not self.load_models():
            self.stats["detection_status"] = "Stopped (model loading failed)"
            return

        # self._log_message(f"[INFORMATION] Opening video source: {self.video_source}")
        self._log_message(f"[INFORMATION] Opening video source...")
        try:
            source = int(self.video_source) if str(self.video_source).isdigit() else self.video_source
            self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        except Exception as e:
            self._log_message(f"[Error] Error initializing VideoCapture: {e}")
            self.cap = None

        if not self.cap or not self.cap.isOpened():
            # self._log_message(f"[Error] Unable to open video source: {self.video_source}")
            self._log_message(f"[Error] Unable to open video source")
            self.stats["detection_status"] = "Stopped (video source error)"
            return

        self.running = True
        self.stats["detection_status"] = "Running"
        self.trackers.clear()
        self.next_id = 0
        
        self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.detection_thread.start()
        self._log_message("[INFORMATION] Detection thread started.")

    def stop_detection(self):
        self.running = False
        if hasattr(self, 'detection_thread') and self.detection_thread.is_alive():
            self._log_message("[INFORMATION] Waiting for detection thread to end...")
            self.detection_thread.join(timeout=5)
        
        with self.detection_lock:
            if self.cap:
                self.cap.release()
                self.cap = None
            self.latest_frame = None
        
        self.trackers.clear()
        self.stats.update({
            "people_count": 0, "fallen_count": 0, "stationary_count": 0, "no_helmet_count": 0,
            "detection_status": "Stopped"
        })
        self._log_message("[INFORMATION] Detection stopped.")

    def detection_loop(self):
        TARGET_FPS = 10
        FRAME_INTERVAL = 1.0 / TARGET_FPS
        last_process_time = 0

        while self.running:
            if not self.cap or not self.cap.isOpened():
                self._log_message("[Warning] Video capture disconnected, trying to reconnect after 5 seconds...")
                time.sleep(5)
                self.start_detection() # 尝试重新走一遍启动流程
                break # 退出当前坏掉的循环，让新线程接管

            if self.is_video_file:
                ret, frame = self.cap.read()
                if not ret: # 如果是视频文件，播放到结尾后自动重播
                    self._log_message("[INFORMATION] The video file ends playing and starts looping from the beginning.")
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
            else: # 如果是实时流，进行跳帧处理
                if time.time() - last_process_time < FRAME_INTERVAL:
                    self.cap.grab() # 只抓取帧但不解码，快速跳过
                    continue
                ret, frame = self.cap.read()
                last_process_time = time.time()
                if not ret:
                    self._log_message("[WARNING] Unable to get frame from live stream")
                    continue

            self.recording_buffer.append(frame.copy())

            pose_results = self.pose_model(frame, verbose=False, conf=self.POSE_PERSON_CONF_THRESHOLD)
            helmet_results = self.helmet_model(frame, verbose=False, classes=[self.helmet_class_id], conf=self.HELMET_CONF_THRESHOLD)

            current_detections = []
            for r in pose_results:
                if r.boxes is not None and r.keypoints is not None:
                    for box, kpts in zip(r.boxes.data.cpu().numpy(), r.keypoints.data.cpu().numpy()):
                        if box[4] > self.POSE_PERSON_CONF_THRESHOLD:
                            current_detections.append({'keypoints': kpts, 'box': box[:4]})

            self._update_trackers(current_detections)

            detected_helmet_boxes = [box.xyxy[0].cpu().numpy() for r in helmet_results for box in r.boxes]

            for tracker in self.trackers.values():
                head_region = tracker.get_head_region()
                helmet_found = False
                if head_region:
                    for helmet_box in detected_helmet_boxes:
                        if self._iou(head_region, helmet_box) > self.HELMET_IOU_THRESHOLD:
                            helmet_found = True
                            break
                tracker.update_helmet_status(helmet_found)

            annotated_frame = self._draw_annotations_and_handle_alerts(frame, detected_helmet_boxes)

            with self.detection_lock:
                self.latest_frame = annotated_frame.copy()

        if self.cap:
            self.cap.release()
        self._log_message("[INFORMATION] The detection cycle has ended.")

    def _update_trackers(self, detections):
        """匹配当前帧的检测结果与现有的追踪器，并处理新增和消失的目标。"""
        
        # 标记所有追踪器为“本帧未更新”
        for tracker in self.trackers.values():
            tracker.updated_in_frame = False

        matched_detection_indices = set()

        # 尝试为每个现有追踪器找到最佳匹配
        for tracker_id, tracker in self.trackers.items():
            best_match_idx, min_dist = -1, self.TRACKER_MAX_MATCH_DISTANCE_PX
            
            for i, det in enumerate(detections):
                if i in matched_detection_indices: continue # 跳过已匹配的检测
                
                dist = self._calculate_distance(tracker.bbox, det['box'])
                if dist < min_dist:
                    min_dist, best_match_idx = dist, i
            
            if best_match_idx != -1:
                # 找到匹配，更新追踪器
                detection = detections[best_match_idx]
                tracker.update(detection['keypoints'], detection['box'])
                tracker.updated_in_frame = True
                matched_detection_indices.add(best_match_idx)

        # 移除长时间未更新的追踪器
        disappeared_ids = []
        for tracker_id, tracker in self.trackers.items():
            if not tracker.updated_in_frame:
                if time.time() - tracker.last_update_time > self.TRACKER_MAX_DISAPPEAR_SECONDS:
                    disappeared_ids.append(tracker_id)
                    self._log_message(f"[Tracking] ID-{tracker_id} was removed due to expiration timeout.")
                    continue
            if tracker.bbox is not None:
                box_height = tracker.bbox[3] - tracker.bbox[1] # y2 - y1
                if box_height < self.MIN_PERSON_HEIGHT_PIXELS:
                    disappeared_ids.append(tracker_id)
                    self._log_message(f"[Tracking] ID-{tracker_id} was removed due to being too small ({int(box_height)}px).")

        for an_id in disappeared_ids:
            if an_id in self.trackers: # 双重检查，防止因多条件判断导致重复删除
                del self.trackers[an_id]

        # 为未匹配的检测创建新追踪器
        for i, detection in enumerate(detections):
            if i not in matched_detection_indices:
                new_box = detection['box']
                new_box_height = new_box[3] - new_box[1]
                if new_box_height < self.MIN_PERSON_HEIGHT_PIXELS:
                    # 如果一个新检测到的人一开始就太小，我们直接忽略他，不为他创建追踪器
                    continue
            
                new_tracker = PersonTracker(self.next_id, detection['keypoints'], detection['box'])
                self.trackers[self.next_id] = new_tracker
                self.next_id = (self.next_id + 1) % 100000 # ID循环使用

    def _draw_annotations_and_handle_alerts(self, frame, detected_helmet_boxes):
        """在帧上绘制所有标注（边界框、状态文本等）并检查是否需要触发警报。"""
        # 初始化当前帧的统计数据
        stats_this_frame = {"fallen": 0, "stationary": 0, "no_helmet": 0}
        
        # --- (可选) 绘制所有检测到的安全帽框和头部估算区域，用于调试 ---
        for helmet_box in detected_helmet_boxes:
            x1, y1, x2, y2 = map(int, helmet_box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 182, 0), 2) # 亮蓝色：检测到的安全帽
        
        # --- 遍历所有追踪器，绘制并检查警报 ---
        for tracker_id, tracker in self.trackers.items():
            color = (0, 255, 0) # 默认绿色
            status_texts = [f"ID-{tracker_id}"]
            
            # --- 绘制头部估算区域 (调试用) ---
            head_region = tracker.get_head_region()
            if head_region:
                hx1, hy1, hx2, hy2 = head_region
                cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 255, 255), 1, cv2.LINE_AA) # 黄色虚线：估算的头部区域

            # --- 检查并处理各种警报状态 ---
            if tracker.is_fallen:
                color = (0, 0, 255) # 红色
                status_texts.append("FALLEN")
                stats_this_frame["fallen"] += 1
                if not tracker.fall_alert_sent and self._can_send_alert(tracker_id, "fall"):
                    self._trigger_alert(f"Fall Alarm - ID {tracker_id}", tracker_id, "fall")
                    tracker.fall_alert_sent = True
            
            elif tracker.is_stationary:
                color = (0, 165, 255) # 橙色
                status_texts.append("STATIONARY")
                stats_this_frame["stationary"] += 1
                if not tracker.stationary_alert_sent and self._can_send_alert(tracker_id, "stationary"):
                    self._trigger_alert(f"Long periods of inactivity - ID {tracker_id}", tracker_id, "stationary")
                    tracker.stationary_alert_sent = True
            
            if not tracker.has_helmet:
                color = (255, 0, 255) if color == (0, 255, 0) else color # 紫色 (如果没其他警报)
                status_texts.append("NO HELMET")
                stats_this_frame["no_helmet"] += 1
                if not tracker.no_helmet_alert_sent and self._can_send_alert(tracker_id, "no_helmet"):
                    self._trigger_alert(f"Not wearing a helmet - ID {tracker_id}", tracker_id, "no_helmet")
                    tracker.no_helmet_alert_sent = True
            
            # --- 绘制人体边界框和状态文本 ---
            x1, y1, x2, y2 = map(int, tracker.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            for i, text in enumerate(status_texts):
                y_pos = y1 - 10 - (i * 20)
                cv2.putText(frame, text, (x1, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # --- 更新全局统计数据和OPC UA变量 ---
        with self.detection_lock:
            self.stats["people_count"] = len(self.trackers)
            self.stats["fallen_count"] = stats_this_frame["fallen"]
            self.stats["stationary_count"] = stats_this_frame["stationary"]
            self.stats["no_helmet_count"] = stats_this_frame["no_helmet"]
        self._update_opcua_variables(stats_this_frame["fallen"], stats_this_frame["no_helmet"])
        
        # --- 绘制屏幕左上角的汇总信息 ---
        cv2.putText(frame, f"People: {len(self.trackers)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220,220,220), 2)
        cv2.putText(frame, f"Fallen: {stats_this_frame['fallen']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255) if stats_this_frame['fallen'] else (220,220,220), 2)
        cv2.putText(frame, f"Stationary: {stats_this_frame['stationary']}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255) if stats_this_frame['stationary'] else (220,220,220), 2)
        cv2.putText(frame, f"No Helmet: {stats_this_frame['no_helmet']}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255) if stats_this_frame['no_helmet'] else (220,220,220), 2)

        return frame

    def _update_opcua_variables(self, fallen_count, no_helmet_count):
        """根据当前帧的警报数量，更新OPC UA服务器上的变量值。"""
        if not opcua_server: return
        try:
            # 信号为1表示有警报，为0表示无警报
            fall_signal = 1 if fallen_count > 0 else 0
            helmet_signal = 1 if no_helmet_count > 0 else 0

            if opcua_fall_variable and opcua_fall_variable.get_value() != fall_signal:
                opcua_fall_variable.set_value(fall_signal)
                # print(f"[OPC UA] 更新 FallStatus -> {fall_signal}")

            if opcua_helmet_variable and opcua_helmet_variable.get_value() != helmet_signal:
                opcua_helmet_variable.set_value(helmet_signal)
                # print(f"[OPC UA] 更新 NoHelmetStatus -> {helmet_signal}")
                
        except Exception as e:
            print(f"❌ Error updating OPC UA variables: {e}")

    def _trigger_alert(self, message, person_id, alert_type):
        """触发一个警报：记录日志、更新冷却时间、并启动视频录制。"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] [Warning] {message}"
        self._log_message(log_message)
        
        # 更新冷却时间戳
        self.alert_cooldown_map[(person_id, alert_type)] = time.time()
        
        # 启动一个新线程来保存警报录像，避免阻塞主检测循环
        threading.Thread(target=self._save_alert_recording, args=(message,)).start()

    def _save_alert_recording(self, alert_message):
        """将缓冲区中的帧保存为警报视频文件，并附带一个JSON信息文件。"""
        frames_to_save = list(self.recording_buffer)
        if not frames_to_save:
            self._log_message("[Warning] Buffer was empty when recording was triggered.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 创建一个对文件名安全的消息字符串
        safe_msg = "".join(c for c in alert_message if c.isalnum() or c in ' -').replace(' ', '_')
        
        video_filename = f"alerts_data/recordings/REC_{safe_msg}_{timestamp}.avi"
        json_filename = f"alerts_data/json/INFO_{safe_msg}_{timestamp}.json"
        
        # 获取视频尺寸
        height, width, _ = frames_to_save[0].shape
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(video_filename, fourcc, self.RECORDING_VIDEO_FPS, (width, height))

        for frame in frames_to_save:
            video_writer.write(frame)
        video_writer.release()
        self._log_message(f"[INFORMATION] Alarm video saved: {video_filename}")

        # 创建并保存JSON元数据文件
        alert_info = {
            'timestamp': datetime.now().isoformat(),
            'alert_message': alert_message,
            'video_file': os.path.basename(video_filename),
            'source': str(self.video_source),
            'frames_saved': len(frames_to_save)
        }
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(alert_info, f, indent=4, ensure_ascii=False)

    # --- 辅助/工具函数 ---
    def _log_message(self, message):
        """向控制台和Web前端的日志队列中添加一条消息。"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {message}" if not message.startswith("[") else message
        print(log_entry)
        with self.detection_lock:
            self.alerts.appendleft(log_entry)

    def _iou(self, boxA, boxB):
        """计算两个边界框的交并比 (Intersection over Union)。"""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        denominator = float(boxAArea + boxBArea - interArea)
        return interArea / denominator if denominator > 0 else 0.0
    
    def _calculate_distance(self, boxA, boxB):
        """计算两个边界框中心的欧氏距离。"""
        center_A = ((boxA[0] + boxA[2]) / 2, (boxA[1] + boxA[3]) / 2)
        center_B = ((boxB[0] + boxB[2]) / 2, (boxB[1] + boxB[3]) / 2)
        return math.sqrt((center_A[0] - center_B[0])**2 + (center_A[1] - center_B[1])**2)

    def _can_send_alert(self, person_id, alert_type):
        """检查特定警报是否已过冷却期，可以再次发送。"""
        now = time.time()
        key = (person_id, alert_type)
        last_alert_time = self.alert_cooldown_map.get(key, 0)
        return now - last_alert_time > self.ALERT_COOLDOWN_SECONDS

    # --- Web接口相关函数 ---
    def get_status(self):
        """获取系统当前状态，用于API返回。"""
        with self.detection_lock:
            # 返回数据的副本，避免多线程问题
            return {
                "running": self.running,
                "source": str(self.video_source),
                "stats": self.stats.copy(),
                "alerts": list(self.alerts)
            }
            
    def update_video_source(self, new_source):
        """更新视频源，只能在停止检测时调用。"""
        if self.running:
            self._log_message("[WARNING] Cannot change video source while detection is running. Please stop first.")
            return False
        self.video_source = new_source
        self.is_video_file = self._is_video_file_source()
        self._log_message(f"[INFORMATION] Video source updated to: {self.video_source}")
        return True

app = Flask(__name__)
ADMIN_SECRET_KEY = "610" # 简单的管理员访问密钥

# --- 初始化主系统 ---
DEFAULT_VIDEO_SOURCE = "rtsp://admin:Admin4321@172.30.40.125/554/live/stream1"
# DEFAULT_VIDEO_SOURCE = "test1.mp4"
system = BehaviorDetectionSystem(video_source=DEFAULT_VIDEO_SOURCE)

def generate_frames():
    """一个生成器函数，用于为Web前端提供MJPEG视频流。"""
    while True:
        with system.detection_lock:
            frame = system.latest_frame.copy() if system.latest_frame is not None else None
        
        if frame is None:
            # 如果没有可用的帧（例如，系统刚启动或视频源错误），显示一个黑色背景和状态文本
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            status_text = system.stats.get("detection_status", "Initializing...")
            cv2.putText(frame, status_text, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            time.sleep(0.1) # 稍作等待，避免空循环占用过多CPU

        # 将帧编码为JPEG格式
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        
        # 以multipart/x-mixed-replace格式产出帧数据
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        time.sleep(0.02) # 控制推流帧率，减轻网络和浏览器负担

@app.route('/')
def index():
    """渲染主页面。通过URL参数`key`判断是否为管理员。"""
    is_admin = request.args.get('key') == ADMIN_SECRET_KEY
    return render_template('index.html', current_source=str(system.video_source), is_admin=is_admin)

@app.route('/video_feed')
def video_feed():
    """视频流的API端点。"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start', methods=['POST'])
def start_detection_route():
    """API端点：启动检测。"""
    system.start_detection()
    # 等待一小会儿，让系统状态更新
    time.sleep(1)
    return jsonify(system.get_status())

@app.route('/stop', methods=['POST'])
def stop_detection_route():
    """API端点：停止检测。"""
    system.stop_detection()
    return jsonify(system.get_status())

@app.route('/status', methods=['GET'])
def get_status_route():
    """API端点：获取系统当前状态和日志。"""
    return jsonify(system.get_status())

@app.route('/update_source', methods=['POST'])
def update_source_route():
    """API端点：更新视频源地址。"""
    if system.running:
        return jsonify({"success": False, "message": "Please stop detection before changing the video source."}), 400
    
    new_source = request.json.get('source')
    if not new_source:
        return jsonify({"success": False, "message": "The video source address was not provided."}), 400
    
    if system.update_video_source(new_source):
        return jsonify({"success": True, "message": f"Video source updated to {new_source}。"})
    else:
        # 理论上不应该发生，除非有未知错误
        return jsonify({"success": False, "message": "Failed to update video source."}), 500

# ==============================================================================
#  新增：文件定期清理模块
# ==============================================================================
import shutil

def cleanup_old_files():
    """
    清理超过指定天数的旧警报文件（视频和JSON）。
    此函数会被一个后台线程定期调用。
    """
    # 定义要清理的目录和文件的最大保留期限（14天）
    FILE_MAX_AGE_DAYS = 14
    FOLDERS_TO_CLEAN = [
        "alerts_data/recordings",
        "alerts_data/json"
    ]
    
    now = time.time()
    cutoff = now - (FILE_MAX_AGE_DAYS * 24 * 60 * 60) # 计算14天前的时间戳
    
    print(f"\n[Cleanup] Running scheduled cleanup task. Deleting files older than {FILE_MAX_AGE_DAYS} days...")
    
    deleted_files_count = 0
    deleted_folders_count = 0

    for folder_path in FOLDERS_TO_CLEAN:
        if not os.path.exists(folder_path):
            print(f"[Cleanup] Directory '{folder_path}' not found, skipping.")
            continue
            
        try:
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                
                # 获取文件的最后修改时间
                file_mod_time = os.path.getmtime(file_path)
                
                if file_mod_time < cutoff:
                    try:
                        os.remove(file_path)
                        print(f"[Cleanup] Deleted old file: {file_path}")
                        deleted_files_count += 1
                    except OSError as e:
                        print(f"[Cleanup] Error deleting file {file_path}: {e}")
        except Exception as e:
            print(f"[Cleanup] An error occurred while processing folder {folder_path}: {e}")

    print(f"[Cleanup] Cleanup finished. Deleted {deleted_files_count} old files.")

def scheduled_cleanup_thread():
    """
    一个后台线程，它会先立即运行一次清理任务，
    然后每隔14天再次运行。
    """
    CLEANUP_INTERVAL_SECONDS = 14 * 24 * 60 * 60  # 14天转换为秒

    while True:
        try:
            # 立即执行一次清理
            cleanup_old_files()
            
            # 等待下一个周期
            print(f"[Cleanup] Next cleanup is scheduled in {int(CLEANUP_INTERVAL_SECONDS / 3600 / 24)} days.")
            time.sleep(CLEANUP_INTERVAL_SECONDS)
        
        except Exception as e:
            print(f"[ERROR] The cleanup thread encountered an error: {e}")
            # 发生错误时，等待1小时再重试，防止因永久性错误导致CPU空转
            time.sleep(3600)
# ==============================================================================

if __name__ == "__main__":
    print("--- Starting the safety behavior detection system ---")

    # 1. 在后台线程中启动OPC UA服务器
    print("[Step 1/3] Starting OPC UA Server...")
    opcua_thread = threading.Thread(target=start_opcua_server, daemon=True)
    opcua_thread.start()
    time.sleep(2) # 等待服务器初始化

    # 2. 在后台启动定期文件清理线程
    print("[Step 2/3] Starting scheduled file cleanup service...")
    cleanup_thread = threading.Thread(target=scheduled_cleanup_thread, daemon=True)
    cleanup_thread.start()

    # 3. 启动Flask Web服务器
    print(f"[STEP 3/3] Starting Web Control Interface...")
    print(f"  - Default video source:{DEFAULT_VIDEO_SOURCE}")
    print(f"  - Ordinary user access: http://172.30.32.231:5000")
    print(f"  - Administrator Access: http://172.30.32.231:5000/?key={ADMIN_SECRET_KEY}")
    # use_reloader=False 对于多线程应用是必须的，以防止代码重载器与我们的线程冲突
    app.run(host='172.30.32.231', port=5000, debug=False, use_reloader=False, threaded=True)

# System Control web
# Running on http://172.30.32.231:5000/?key=610

# Normal User
# Running on http://172.30.32.231:5000
