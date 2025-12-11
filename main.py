import cv2
import argparse
from src.inference import detect_objects

#Configuration for arguemnts parser
parser = argparse.ArgumentParser("System analizy piłkarskiej")
parser.add_argument('--source_video_path', type=str, required=True, help='source video path')
args = parser.parse_args()

path = args.source_video_path

cap = cv2.VideoCapture(path)

nr_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

for i in range(nr_of_frames):
    ret, frame = cap.read()
    if ret:
        results = detect_objects(frame)
        annotated_frame = results.plot()
        cv2.imshow("frame", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()