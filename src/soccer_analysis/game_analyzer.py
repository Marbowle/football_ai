import cv2
import numpy as np
import pandas as pd
import supervision as sv
from src.soccer_analysis.inference import Detector
from src.soccer_analysis.trackers import Tracker
from src.soccer_analysis.team_assigner import TeamAssigner
from src.soccer_analysis.player_ball_assigner import PlayerBallAssigner
from src.soccer_analysis.utils.visualizer import Visualizer

class GameAnalyzer:
    def __init__(self, source_video_path, model_path):
        self.source_video_path = source_video_path
        self.model_path = model_path
        self.detector = Detector(model_path)
        self.tracker = Tracker()
        self.team_assigner = TeamAssigner()
        self.player_ball_assigner = PlayerBallAssigner()
        self.player_team_assignments = {}
        self.player_colors_frame1 = {}
        self.visualizer = Visualizer()

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
        df = pd.DataFrame(ball_detections, columns=['x1', 'y1', 'x2', 'y2'])
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

        label_annotator = sv.LabelAnnotator()

        for i in range(frame_count):
            ret, frame = cap.read()
            if ret:
                results = self.detector.detect_objects(frame)
                detections = sv.Detections.from_ultralytics(results)
                detections = self.tracker.update_with_detections(detections)

                if i == 0:
                    players_only = detections[detections.class_id == 2]
                    self.team_assigner.assign_team_color(frame, players_only)

                    for bbox, _, _, class_id, tracker_id, _ in players_only:
                        player_color = self.team_assigner.get_player_color(frame, bbox)
                        team_id = self.team_assigner.kmeans.predict([player_color])[0]
                        self.player_team_assignments[tracker_id] = team_id
                        self.player_colors_frame1[tracker_id] = player_color

                # Calculate ball control
                ball_bbox = ball_detections[i]

                # List of players
                players_bboxes_dict = {}
                for bbox, _, _, class_id, tracker_id, _ in detections:
                    if class_id == 2:
                        players_bboxes_dict[tracker_id] = bbox

                assigned_player_id = -1
                if ball_bbox is not None:
                    assigned_player_id = self.player_ball_assigner.assign_ball_to_player(players_bboxes_dict, ball_bbox)

                for bbox, _, _, class_id, tracker_id, _ in detections:
                    if class_id == 2 and tracker_id not in self.player_team_assignments:
                        player_color = self.team_assigner.get_player_color(frame, bbox)
                        team_id = self.team_assigner.kmeans.predict([player_color])[0]
                        self.player_team_assignments[tracker_id] = team_id

                annotated_frame = frame.copy()

                annotated_frame = self.visualizer.draw_scene(
                    annotated_frame,
                    detections,
                    self.player_team_assignments,
                    self.team_assigner,
                    assigned_player_id,
                    ball_bbox
                )

                annotated_frame = self.visualizer.draw_labels(
                    annotated_frame,
                    detections,
                    self.player_team_assignments
                )

                video_writer.write(annotated_frame)


        cap.release()
        video_writer.release()
