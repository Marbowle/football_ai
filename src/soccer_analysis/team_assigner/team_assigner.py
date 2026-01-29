import numpy as np
from sklearn.cluster import KMeans


class TeamAssigner(object):
    def __init__(self):
        self.team_colors = {}
        self.kmeans = None

# Cut the middle to get right color
    def get_player_color(self, frame, bbox):
        """
        Extracts the dominant color from a specific region (bbox) of the frame using K-Means.
        It first crops the player image based on the bounding box, then performs clustering.

        Returns: The cluster centers (RGB values) representing the dominant colors.
        """

        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        h = image.shape[0]
        w = image.shape[1]
        top = int(h * 0.35)
        bottom = int(h * 0.65)
        left = int(w* 0.4)
        right = int(w * 0.6)

        jersey_crop = image[top:bottom, left:right]
        jersey_crop_rgb = jersey_crop[:, :, ::-1]

        mean_color = np.mean(jersey_crop, axis=(0,1))

        return mean_color


# Assign right color for the one team
    def assign_team_color(self, frame, player_detections):
        """
        Determines the two primary team colors by clustering all detected players' colors.
        It uses K-Means to find the two dominant jersey colors across the provided frame.

        Updates the 'team_colors' attribute with the two resulting cluster centers.
        """

        player_colors = []

        for bbox,_, _, _, _, _ in player_detections:
            player_color = self.get_player_color(frame,bbox)
            player_colors.append(player_color)


        self.kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10, random_state=42)
        self.kmeans.fit(player_colors)

        self.team_colors[0] = self.kmeans.cluster_centers_[0]
        self.team_colors[1] = self.kmeans.cluster_centers_[1]

#Assign player for the correct team
    def get_player_team(self, frame, player_bbox, player_id):
        """
        Predicts the team assignment for a specific player based on their jersey color.
        It uses the pre-trained K-Means model to classify the player's extracted color.

        Returns: The predicted team ID (e.g., 0 or 1).
        """

        if self.kmeans is None:
            return 0

        player_color = self.get_player_color(frame, player_bbox)
        team_id = self.kmeans.predict([player_color])[0]

        return team_id


