# Factory-Personnel-Behavior-Recognition-and-Detection-System

这是一个集成了YOLOv8姿态估计和物体检测、Flask Web服务以及OPC UA工业通讯的综合性安防行为检测系统。

主要功能:
1.  **实时视频分析**: 从摄像头、RTSP流或视频文件获取视频。
    -   监控RTSP_URL = "rtsp://your_rtsp_stream_url_here"获取到画面
2.  **行为检测**:
    -   人员追踪: 识别并追踪画面中的每一个人。
    -   摔倒检测: 基于人体关键点和外接矩形的宽高比判断是否有人摔倒。
    -   静止检测: 判断人员是否在原地停留过长时间。
    -   安全帽佩戴检测: 结合人员识别和安全帽检测，判断是否佩戴安全帽。
3.  **警报系统**:
    -   当检测到异常行为时，在界面和后台触发警报。
    -   自动录制警报发生前后的视频片段，并保存相关信息为JSON文件。
4.  **Web界面**:
    -   通过Flask提供一个Web界面，实时显示标注后的视频流。
    -   提供控制接口（启动/停止检测、更换视频源）和状态监控（人员数量、警报日志）。
5.  **工业通讯 (OPC UA)**:
    -   启动一个OPC UA服务器。
    -   将关键警报状态（如是否有人摔倒、是否有人未戴安全帽）作为OPC UA变量发布，
      方便与SCADA、MES等工业系统集成。

新开发者上手指南:
-   **环境配置**: 确保已安装 requirements.txt 中的所有库 (flask, opencv-python, numpy, ultralytics, opcua)。
-   **模型准备**: 确保 'yolov8x-pose.pt' (姿态估计) 和 'hemletYoloV8_100epochs.pt' (安全帽检测) 模型文件在项目根目录下。
-   **参数调整**:
    -   `BehaviorDetectionSystem` 类顶部的 `CONFIGURATION` 区域是核心参数区，可调整检测灵敏度、录制设置等。
    -   `PersonTracker` 类顶部的常量区负责单个追踪目标的行为判断阈值。
    -   `__main__` 部分可以修改默认视频源和Web服务器的IP/端口。
-   **运行**: 直接运行app.py文件即可启动整个系统。

访问[管理界面](http://172.30.32.231:5000/?key=610)

只是[监控界面](http://172.30.32.231:5000)

---

安全帽检测模型，训练数据来源[git项目](https://github.com/jomarkow/Safety-Helmet-Detection)