from ultralytics import YOLO

model=YOLO("yolov5n.pt")

model.train(data="data.yaml",batch=8,imgsz=640,epochs=100)