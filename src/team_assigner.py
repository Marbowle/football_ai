import numpy as np
from sklearn.cluster import KMeans


class TeamAssigner(object):
    def __init__(self):
        self.team_colors = {}
        self.kmeans = KMeans(n_clusters=2)

# Cut the middle to get right color
    def get_player_color(self, frame, bbox):

        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        h = image.shape[0]
        top= int(0.25 * h)
        bottom= int(0.75 * h)

        jersey_crop = image[top:bottom, :]

        mean_color = np.mean(jersey_crop, axis=(0,1))

        return mean_color
# Assign right color for the one team
    def assign_team_color(self, frame, player_detections):

        player_colors = []

        for bbox,_, _, _, _, _ in player_detections:
            player_colors.append(self.get_player_color(frame,bbox))

        self.kmeans.fit(player_colors)

        self.team_colors[0] = self.kmeans.cluster_centers_[0]
        self.team_colors[1] = self.kmeans.cluster_centers_[1]
#Assign player for the correct team
    def get_player_team(self, frame, player_bbox, player_id):

        player_color = self.get_player_color(frame, player_bbox)
        team_id = self.kmeans.predict([player_color])[0]

        return team_id


