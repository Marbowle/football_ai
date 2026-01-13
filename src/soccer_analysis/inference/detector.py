from ultralytics import YOLO


#Trainig model based on football dataset
class Detector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_objects(self, frame):
        result = self.model(frame)[0]

        return result




