import supervision as sv
import numpy as np


class Visualizer:
    def __init__(self):
        self.ellipse_annotator = sv.EllipseAnnotator(
            color=sv.Color.RED,
            thickness=2
        )
        self.triangle_annotator = sv.TriangleAnnotator(
            color=sv.Color.YELLOW,
            base=20,
            height=20
        )
        self.label_annotator = sv.LabelAnnotator(
            text_color=sv.Color.BLACK,
            text_position=sv.Position.BOTTOM_CENTER
        )

    def draw_scene(self, frame, detections, player_team_assignments, team_assigner, assigned_player_id, ball_bbox):
        """
        Draws visual annotations (ellipses, labels, ball markers) onto the video frame.
        It visualizes player positions using team colors and highlights the ball and the player in possession.

        Returns: The annotated video frame with all graphics applied.
        """
        for bbox, _, _, class_id, tracker_id, _ in detections:
            if class_id != 2: continue

            team_id = player_team_assignments.get(tracker_id)
            if team_id is None: continue

            team_color = team_assigner.team_colors[team_id]
            sv_color = sv.Color(r=int(team_color[2]), g=int(team_color[1]), b=int(team_color[0]))

            self.ellipse_annotator.color = sv_color

            single_detection = sv.Detections(
                xyxy=np.array([bbox]),
                class_id=np.array([class_id]),
                tracker_id=np.array([tracker_id])
            )


            frame = self.ellipse_annotator.annotate(scene=frame, detections=single_detection)


            if tracker_id == assigned_player_id:
                self.triangle_annotator.color = sv_color
                frame = self.triangle_annotator.annotate(scene=frame, detections=single_detection)


        if ball_bbox is not None:
            if assigned_player_id == -1:
                self.triangle_annotator.color = sv.Color.YELLOW


            ball_detection = sv.Detections(xyxy=np.array([ball_bbox]), class_id=np.array([0]))
            frame = self.triangle_annotator.annotate(scene=frame, detections=ball_detection)

        return frame

    def draw_labels(self, frame, detections, player_team_assignments):
        """
        Draws bounding boxes and tracking IDs for detected objects on the frame.
        It visualizes specific object details (like Player ID) to assist in tracking verification.

        Returns: The frame with added textual labels and bounding boxes.
        """
        people_detections = detections[detections.class_id != 0]

        labels = []
        for _, _, _, class_id, tracker_id, _ in people_detections:
            if class_id == 1:
                labels.append("GK")
            elif class_id == 3:
                labels.append("REF")
            elif class_id == 2:
                team_id = player_team_assignments.get(tracker_id, "?")
                labels.append(f"ID: {tracker_id} T: {team_id}")

        return self.label_annotator.annotate(
            scene=frame,
            detections=people_detections,
            labels=labels
        )