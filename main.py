import argparse
import cv2
import supervision as sv
from src.inference import detect_objects
from src.tracker import create_tracker

# Configuration for arguments parser
parser = argparse.ArgumentParser("System analizy piłkarskiej")
parser.add_argument('--source_video_path', type=str, required=True, help='source video path')
args = parser.parse_args()

path = args.source_video_path

cap = cv2.VideoCapture(path)

nr_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

tracker = create_tracker()

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

for i in range(nr_of_frames):
    ret, frame = cap.read()
    if ret:
        results = detect_objects(frame)
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)
        annotated_frame = results.plot()
        cv2.imshow("frame", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()