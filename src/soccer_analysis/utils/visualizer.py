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
        # 1. Rysowanie graczy (Elipsy i Trójkąty posiadania)
        for bbox, _, _, class_id, tracker_id, _ in detections:
            if class_id != 2: continue  # Tylko gracze

            # Pobierz kolor drużyny
            team_id = player_team_assignments.get(tracker_id)
            if team_id is None: continue  # Zabezpieczenie

            team_color = team_assigner.team_colors[team_id]
            # Konwersja koloru dla Supervision (BGR -> RGB)
            sv_color = sv.Color(r=int(team_color[2]), g=int(team_color[1]), b=int(team_color[0]))

            # Ustaw kolory annotatorów
            self.ellipse_annotator.color = sv_color

            # Stwórz pojedynczą detekcję do narysowania
            single_detection = sv.Detections(
                xyxy=np.array([bbox]),
                class_id=np.array([class_id]),
                tracker_id=np.array([tracker_id])
            )

            # Rysuj elipsę
            frame = self.ellipse_annotator.annotate(scene=frame, detections=single_detection)

            # Jeśli ma piłkę -> rysuj trójkąt w kolorze drużyny
            if tracker_id == assigned_player_id:
                self.triangle_annotator.color = sv_color
                frame = self.triangle_annotator.annotate(scene=frame, detections=single_detection)

        # 2. Rysowanie Piłki (Żółty trójkąt, jeśli wolna)
        if ball_bbox is not None:
            if assigned_player_id == -1:
                self.triangle_annotator.color = sv.Color.YELLOW
            # Jeśli ktoś ma piłkę, kolor trójkąta został ustawiony wyżej, więc tu go nie resetujemy,
            # chyba że chcesz wymusić żółty zawsze nad piłką - wtedy odkomentuj linię wyżej.

            ball_detection = sv.Detections(xyxy=np.array([ball_bbox]), class_id=np.array([0]))
            frame = self.triangle_annotator.annotate(scene=frame, detections=ball_detection)

        return frame

    def draw_labels(self, frame, detections, player_team_assignments):
        labels = []
        for _, _, _, class_id, tracker_id, _ in detections:
            if class_id == 1:
                labels.append("GK")
            elif class_id == 3:
                labels.append("REF")
            elif class_id == 2:
                team_id = player_team_assignments.get(tracker_id, "?")
                labels.append(f"ID: {tracker_id} T: {team_id}")
            else:
                labels.append("")  # Ball


        return self.label_annotator.annotate(
            scene=frame,
            detections=detections[detections.class_id != 0],
            labels=labels
        )