import cv2
import numpy as np
import pandas as pd
import supervision as sv
from src.soccer_analysis.inference import Detector
from src.soccer_analysis.trackers import Tracker
from src.soccer_analysis.team_assigner import TeamAssigner
from src.soccer_analysis.player_ball_assigner import PlayerBallAssigner

class GameAnalyzer:
    def __init__(self, source_video_path, model_path):
        self.source_video_path = source_video_path
        self.model_path = model_path
        self.detector = Detector(model_path)
        self.tracker = Tracker()
        self.team_assigner = TeamAssigner()
        self.player_ball_assigner = PlayerBallAssigner()
        self.player_team_assignments  = {}

        self.ellipse_annotator = sv.EllipseAnnotator(
            color = sv.Color.RED,
            thickness=2
        )

        self.triangle_annotator = sv.TriangleAnnotator(
            color=sv.Color.YELLOW,
            base= 20,
            height = 20
        )

        self.label_annotator = sv.LabelAnnotator(
            text_color=sv.Color.BLACK,
            text_position=sv.Position.BOTTOM_CENTER
        )

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

                #Calculate ball control
                ball_bbox = ball_detections[i]

                #List of players
                players_bboxes_dict = {}
                for bbox, _, _, class_id, tracker_id, _ in detections:
                    if class_id == 2:
                        players_bboxes_dict[tracker_id] = bbox

                assigned_player_id = -1
                if ball_bbox is not None:
                    assigned_player_id = self.player_ball_assigner.assign_ball_to_player(players_bboxes_dict, ball_bbox)

                # Create labels for correct assign team
                labels = []
                annotated_frame = frame.copy()

                for bbox, _, _, class_id, tracker_id, _ in detections:
                    if class_id == 0:
                        continue
                    elif class_id == 1:
                        labels.append("GK")
                    elif class_id == 3:
                        labels.append("REF")
                    else:
                        if tracker_id in self.player_team_assignments:
                            team_id = self.player_team_assignments[tracker_id]
                        else:
                            team_id = self.team_assigner.get_player_team(frame, bbox, tracker_id)
                            self.player_team_assignments[tracker_id] = team_id

                        labels.append(f"ID: {tracker_id} T: {team_id}")
                        team_color = self.team_assigner.team_colors[team_id]
                        self.ellipse_annotator.color = sv.Color(
                            r=int(team_color[0]),
                            g=int(team_color[1]),
                            b=int(team_color[2])
                        )
                        single_player_detection = sv.Detections(
                            xyxy=np.array([bbox]),
                            class_id=np.array([class_id]),
                            tracker_id=np.array([tracker_id])
                        )

                        annotated_frame = self.ellipse_annotator.annotate(
                            scene=annotated_frame,
                            detections=single_player_detection
                        )
                        if tracker_id == assigned_player_id:
                            team_color = self.team_assigner.team_colors[team_id]
                            self.triangle_annotator.color = sv.Color(
                                r=int(team_color[0]),
                                g=int(team_color[1]),
                                b=int(team_color[2])
                            )


                class_id = np.array([0])
                ball_detections_for_drawing = sv.Detections(xyxy=np.array([ball_bbox]), class_id=class_id)
                self.triangle_annotator.color = sv.Color.YELLOW

                # 2. Drawing frames and labels
                annotated_frame = self.label_annotator.annotate(
                    scene=annotated_frame,
                    detections=detections[detections.class_id != 0],
                    labels=labels
                )
                annotated_frame = self.triangle_annotator.annotate(
                    scene=annotated_frame,
                    detections=ball_detections_for_drawing
                )

                video_writer.write(annotated_frame)
            else:
                break

        cap.release()
        video_writer.release()




