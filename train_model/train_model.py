if __name__ == '__main__':
    from ultralytics import YOLO
    
    model = YOLO('yolov8n.pt') 
    model.info()
    results = model.train(data="data.yaml", epochs=100)

# # model = YOLO("yolov8n.pt")
# # model.train(data="data.yaml", epochs=10)
# # result = model.val()
# # path = model.export(format="onnx")

# # from ultralytics import YOLO

# # Load a COCO-pretrained YOLOv8n model
# model = YOLO("yolov8n.pt")

# # Display model information (optional)
# model.info()

# # Train the model on the COCO8 example dataset for 100 epochs
# # results = model.train(data="data.yaml", epochs=100, imgsz=640)
# results = model.train(data="data.yaml", epochs=100)

# # Run inference with the YOLOv8n model on the 'bus.jpg' image
# # results = model("path/to/bus.jpg")