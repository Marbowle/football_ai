import cv2
import numpy as np
import pandas as pd
import supervision as sv
from src.soccer_analysis.inference import Detector
from src.soccer_analysis.trackers import Tracker
from src.soccer_analysis.team_assigner import TeamAssigner

class GameAnalyzer:
    def __init__(self, source_video_path, model_path):
        self.source_video_path = source_video_path
        self.model_path = model_path
        self.detector = Detector(model_path)
        self.tracker = Tracker()
        self.team_assigner = TeamAssigner()

    def extract_ball_positions(self):
        cap = cv2.VideoCapture(self.source_video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ball_detections = []

        for i in range(frame_count):
            ret, frame = cap.read()
            ball_bbox = None
            if ret:
                results = self.detector.detect_objects(frame)
                detections = sv.Detections.from_ultralytics(results)

                for bbox, _, _, class_id, _, _ in detections:
                    if class_id == 0:
                        ball_bbox = bbox
                ball_detections.append(ball_bbox)

        cap.release()

        ball_detections = [x if x is not None else [np.nan, np.nan, np.nan, np.nan] for x in ball_detections]
        # Ball interpolation
        df = pd.DataFrame(ball_detections, columns=['x1', 'x2', 'y1', 'y2'])
        # Interpolation
        df = df.interpolate()
        # Fill missing values
        df = df.bfill()

        ball_detections = df.values.tolist()

        return ball_detections


    def process_video(self, output_video_path):
        video_path = self.source_video_path
        ball_detections = self.extract_ball_positions()
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        bounding_box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        for i in range(frame_count):
            ret, frame = cap.read()
            if ret:
                results = self.detector.detect_objects(frame)
                detections = sv.Detections.from_ultralytics(results)
                detections = self.tracker.update_with_detections(detections)


                # Methods to assign right color in first frame
                if i == 0:
                    players_only = detections[detections.class_id == 2]
                    self.team_assigner.assign_team_color(frame, players_only)

                # Create labels for correct assign team
                labels = []

                for bbox, _, _, class_id, tracker_id, _ in detections:
                    if class_id == 1:
                        labels.append(f"ID: {tracker_id} GK")
                    elif class_id == 3:
                        labels.append(f"ID: {tracker_id} REF")
                    elif class_id == 0:
                        labels.append(f"ID: {tracker_id}, sports ball")
                    else:
                        team_id = self.team_assigner.get_player_team(frame, bbox, tracker_id)
                        labels.append(f"ID: {tracker_id} T: {team_id}")

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

                if i < len(ball_detections):
                    if ball_detections[i] is not None:
                        bbox = ball_detections[i]
                        x1, y1, x2, y2 = bbox

                        # Obliczamy środek
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)

                        # drawing function
                        cv2.circle(annotated_frame, (int(center_x), int(center_y)), 5, (255, 255, 255), -1)

                video_writer.write(annotated_frame)
            else:
                break

        cap.release()
        video_writer.release()




