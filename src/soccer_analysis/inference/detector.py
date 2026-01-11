from ultralytics import YOLO

#Trainig model based on football dataset
model = YOLO('models/best.pt')

def detect_objects(frame):

    result = model(frame)[0]

    return result

