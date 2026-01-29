import sys
sys.path.append('../../')


class PlayerBallAssigner:
    def __init__(self):
        self.max_player_ball_distance = 70

    def get_player_feet_position(self, bbox):
        """
        Calculates the coordinates of the player's feet (bottom-center of the bounding box).
        It identifies the ground contact point to accurately map the player's position on the pitch.

        Returns: The [x, y] coordinates representing the player's feet.
        """
        x1, y1, x2, y2 = bbox

        feet_x = (x1 + x2) / 2
        feet_y = y2

        return (int(feet_x), int(feet_y))

    def assign_ball_to_player(self, players, ball_bbox):
        """
        Determines which player is currently in possession of the ball based on proximity.
        It calculates the distance between the ball and each player to assign control to the closest one.

        Returns: The ID of the player in possession, or a sentinel value (e.g., -1/None) if no one controls the ball.
        """
        ball_positions = self.get_center_of_bbox(ball_bbox)

        min_distance = 99999
        assigned_player = -1

        for player_id, player_bbox in players.items():
            player_position = self.get_player_feet_position(player_bbox)

            distance = ((player_position[0] -ball_positions[0])** 2 + (player_position[1] - ball_positions[1]) ** 2) ** 0.5

            if distance < self.max_player_ball_distance:
                if distance < min_distance:
                    min_distance = distance
                    assigned_player = player_id
        return assigned_player

    def get_center_of_bbox(self, bbox):
        """
       Calculates the geometric center point of the provided bounding box.
       It computes the midpoint of the box's width and height to represent the object's general location.

       Returns: The [x, y] coordinates of the center.
       """
        x1, y1, x2, y2 = bbox
        center_x = int((x1 + x2) / 2)
        center_y = int((y2 + y1) / 2)
        return center_x, center_y