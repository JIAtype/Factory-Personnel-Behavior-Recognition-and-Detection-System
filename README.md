# Factory-Personnel-Behavior-Recognition-and-Detection-System

用python实现工厂人员行为识别检测系统，使用yolov8x-pose.pt模型。通过监控RTSP_URL = "rtsp://your_rtsp_stream_url_here"获取到画面，检测是否有人摔倒或者超过3分钟一动不动。系统会同步显示监控画面，在画面标出检测到的人员。如果出现特殊情况就在系统中显示，并且弹出对应警报，并保存出现这种情况时的监控画面（5分钟）。系统要用英文，界面要显示必要信息。
