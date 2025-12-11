from ultralytics import YOLO

model = YOLO('yolov8x.pt')

def detect_objects(frame):

    result = model(frame)[0]

    return result

