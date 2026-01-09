import argparse
import cv2
import supervision as sv
from src.inference import detect_objects
from src.tracker import create_tracker
from src.team_assigner import TeamAssigner

# Configuration for arguments parser
parser = argparse.ArgumentParser("System analizy piłkarskiej")
parser.add_argument('--source_video_path', type=str, required=True, help='source video path')
args = parser.parse_args()

path = args.source_video_path

cap = cv2.VideoCapture(path)

nr_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
output_path = "result.mp4"

#Assign vidoe to watch
video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

tracker = create_tracker()

team_assigner = TeamAssigner()

bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

for i in range(nr_of_frames):
    ret, frame = cap.read()
    if ret:
        results = detect_objects(frame)
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)

        # Methods to assign right color in first frame
        if i == 0:
            players_only = detections[detections.class_id == 2]
            team_assigner.assign_team_color(frame, players_only)

        # Create labels for correct assign team
        labels = []

        for bbox, _, _, class_id, tracker_id, _ in detections:
            if class_id != 3 and class_id != 0:
                team_id = team_assigner.get_player_team(frame, bbox, tracker_id)
                labels.append(f"ID: {tracker_id} T: {team_id}")
            elif class_id == 3:
                labels.append(f"ID: {tracker_id} REF")
            else:
                labels.append(f"ID: {tracker_id}, sports ball")

        # 2. Drawing frames and labels
        annotated_frame = bounding_box_annotator.annotate(
            scene=frame.copy(),
            detections=detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels
        )
        video_writer.write(annotated_frame)

cap.release()
