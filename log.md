(llava) D:\aiml>conda create -n pose python 

(llava) D:\aiml>conda activate pose

(pose) D:\aiml>pip install opencv-python

(pose) D:\aiml>pip install ultralytics

(pose) D:\aiml>cd safety_monitoring
(pose) D:\aiml\safety_monitoring>python3 app.py

(pose) D:\aiml\safety_monitoring>pip install opencv-contrib-python  

(pose) D:\aiml\safety_monitoring>python app.py 
Creating new Ultralytics Settings v0.0.6 file  
View Ultralytics Settings with 'yolo settings' or at 'C:\Users\splsip258\AppData\Roaming\Ultralytics\settings.json'
Update Settings with 'yolo settings key=value', i.e. 'yolo settings runs_dir=path/to/dir'. For help see https://docs.ultralytics.com/quickstart/#ultralytics-settings.
Loading YOLOv8 pose model: yolov8x-pose.pt
Model loaded.
Attempting to open RTSP stream: video1.mp4
[mov,mp4,m4a,3gp,3g2,mj2 @ 0000020b264224c0] moov atom not found
Error: Could not open video stream from video1.mp4
Please check the URL, network connectivity, and camera availability.

opencv有问题，重新下载一下
好像是python版本太新了

(pose) D:\aiml\safety_monitoring>conda create -n yolo8 python=3.10
(pose) D:\aiml\safety_monitoring>conda activate yolo8   
(yolo8) D:\aiml\safety_monitoring>pip install ultralytics

(yolo8) D:\aiml\safety_monitoring>python app.py

(yolo8) D:\aiml\safety_monitoring>python app.py
Loading YOLOv8 pose model: yolov8x-pose.pt
Model loaded.
Attempting to open RTSP stream: video2.mp4
Successfully opened video stream.
ALERT: Person ID 0 STILL for too long!
ALERT: Person ID 1 STILL for too long!
ALERT: Person ID 2 STILL for too long!
ALERT: Person ID 2 STILL for too long!
ALERT: Person ID 0 STILL for too long!
ALERT: Person ID 2 STILL for too long!
ALERT: Person ID 1 STILL for too long!
Removing track for person ID 3
ALERT: Person ID 2 STILL for too long!
ALERT: Person ID 1 STILL for too long!
ALERT: Person ID 2 STILL for too long!
ALERT: Person ID 2 STILL for too long!
ALERT: Person ID 1 STILL for too long!
ALERT: Person ID 2 STILL for too long!

(yolo8) D:\aiml\safety_monitoring>nvidia-smi
检查自己有没有GPU
设置使用GPU
model.to('cuda')