import os
import cv2
import numpy as np
import pandas as pd
import supervision as sv
from src.soccer_analysis.inference import Detector
from src.soccer_analysis.trackers import Tracker
from src.soccer_analysis.team_assigner import TeamAssigner
from src.soccer_analysis.player_ball_assigner import PlayerBallAssigner
from src.soccer_analysis.utils.visualizer import Visualizer
from src.soccer_analysis.utils.pickle_utils import save_detections, load_detections
from src.soccer_analysis.camera_movement_estimator import CameraMovementEstimator
from src.soccer_analysis.view_transformer import ViewTransformer
from src.soccer_analysis.utils.io_utils import save_tracks_to_csv

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
        self.camera_movement_estimator = CameraMovementEstimator()
        self.view_transformer = ViewTransformer()

    def extract_ball_positions(self, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            return load_detections(stub_path)

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
            else:
                break

        cap.release()


        ball_detections = [x if x is not None else [np.nan, np.nan, np.nan, np.nan] for x in ball_detections]
        df = pd.DataFrame(ball_detections, columns=['x1', 'y1', 'x2', 'y2'])
        df = df.interpolate()
        df = df.bfill()
        ball_detections = df.values.tolist()


        if stub_path is not None:
            save_detections(ball_detections, stub_path)

        return ball_detections

    def get_annotations(self, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            print("Wczytywanie detekcji z pliku...")
            return load_detections(stub_path)

        print("Rozpoczynanie detekcji i śledzenia (to może chwilę potrwać)...")
        cap = cv2.VideoCapture(self.source_video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        tracks = []

        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            results = self.detector.detect_objects(frame)
            detections = sv.Detections.from_ultralytics(results)
            detections = self.tracker.update_with_detections(detections)
            tracks.append(detections)

        cap.release()

        if stub_path is not None:
            save_detections(tracks, stub_path)

        return tracks

    def convert_detections_to_tracks(self, detections_list):
        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        for _ in range(len(detections_list)):
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

        for frame_num, detection_frame in enumerate(detections_list):
            for i in range(len(detection_frame)):
                bbox = detection_frame.xyxy[i]
                class_id = detection_frame.class_id[i]
                track_id = detection_frame.tracker_id[i] if detection_frame.tracker_id is not None else -1

                object_type = None
                if class_id == 2:  # Player
                    object_type = "players"
                elif class_id == 1:  # Goalkeeper
                    object_type = "players"
                elif class_id == 3:  # Referee
                    object_type = "referees"
                elif class_id == 0:  # Ball
                    object_type = "ball"
                    track_id = 1

                if object_type:
                    x1, y1, x2, y2 = bbox
                    if object_type == "ball":
                        position = ((x1 + x2) / 2, (y1 + y2) / 2)
                    else:
                        position = ((x1 + x2) / 2, y2)

                    tracks[object_type][frame_num][track_id] = {
                        "bbox": bbox,
                        "position": position,
                        "team_id": None
                    }

        return tracks

    def assign_teams(self, frames, raw_detections, tracks):

        print("Przydzielanie druzyn")
        for frame_num, frame in enumerate(frames):
            if frame_num >= len(raw_detections): continue

            detections = raw_detections[frame_num]

            players_only = detections[detections.class_id == 2]

            if frame_num == 0:
                self.team_assigner.assign_team_color(frame, players_only)


            for i in range(len(players_only)):
                bbox = players_only.xyxy[i]
                tracker_id = players_only.tracker_id[i]

                team_id = self.team_assigner.get_player_team(frame, bbox, tracker_id)


                self.player_team_assignments[tracker_id] = team_id


                if tracker_id in tracks["players"][frame_num]:
                    tracks["players"][frame_num][tracker_id]["team_id"] = team_id

    def process_video(self, output_video_path):
        video_path = self.source_video_path


        raw_detections = self.get_annotations(read_from_stub=True, stub_path='stub_player_detections.pkl')
        ball_detections = self.extract_ball_positions(read_from_stub=True, stub_path='stub_ball_detections.pkl')

        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            frames.append(frame)
        cap.release()

        detections_tracks = self.convert_detections_to_tracks(raw_detections)


        camera_movement = self.camera_movement_estimator.get_camera_movement(
            frames, raw_detections, read_from_stub=True, stub_path='stub_camera_movement.pkl'
        )


        self.camera_movement_estimator.add_adjust_positions_to_tracks(detections_tracks, camera_movement)
        self.view_transformer.add_transformed_position_to_tracks(detections_tracks)

        self.assign_teams(frames, raw_detections, detections_tracks)

        save_tracks_to_csv(detections_tracks, 'output_game_data.csv')


        height, width = frames[0].shape[:2]
        fps = 24
        video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        for i, frame in enumerate(frames):
            if i < len(raw_detections):
                detections = raw_detections[i]
            else:
                detections = sv.Detections.empty()

            ball_bbox = ball_detections[i]
            assigned_player_id = -1
            if ball_bbox is not None:
                players_bboxes_dict = {}
                for bbox, _, _, class_id, tracker_id, _ in detections:
                    if class_id == 2:
                        players_bboxes_dict[tracker_id] = bbox
                assigned_player_id = self.player_ball_assigner.assign_ball_to_player(players_bboxes_dict, ball_bbox)

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

            if i % 50 == 0:
                print(f"Generowanie wideo: klatka {i}/{len(frames)}")

        video_writer.release()
        print("Gotowe! Wideo i CSV zapisane.")