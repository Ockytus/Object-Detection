# Train du lieu anh
from ultralytics import YOLO
# Load a model
#model = YOLO('yolov8n.yaml') # build a new model from YAML
model = YOLO(r'C:\Users\User\Desktop\Xu ly anh\NEW\runs\detect\train7\weights\last.pt') # load a pretrained model (recommended for training) goc:yolov8n.pt
if __name__ == '__main__': # can khi chay bang GPU
    model.train(data='dataset.yaml', epochs=10, imgsz=640, batch=10, optimizer='Adam', workers=1)
    metrics = model.val()