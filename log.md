(pose) D:\aiml>cd safety_monitoring
(pose) D:\aiml\safety_monitoring>conda create -n yolo8 python=3.10
(pose) D:\aiml\safety_monitoring>conda activate yolo8   
(yolo8) D:\aiml\safety_monitoring>pip install ultralytics
(yolo8) D:\aiml\safety_monitoring>python app.py
(yolo8) D:\aiml\safety_monitoring>nvidia-smi
检查自己有没有GPU
设置使用GPU
model.to('cuda')
