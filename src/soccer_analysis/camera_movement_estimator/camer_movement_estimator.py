import pickle
import cv2
import numpy as np
import os
import sys


class CameraMovementEstimator:
    def __init__(self, frame=None):
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        self.features = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7
        )

    def get_camera_movement(self, frames, annotations, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        camera_movement = [[0, 0]] * len(frames)

        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)


        mask_features = np.zeros_like(old_gray)
        mask_features[:] = 255

        for bbox, _, _, class_id, _, _ in annotations[0]:
            if class_id == 2 or class_id == 3:
                x1, y1, x2, y2 = bbox
                mask_features[int(y1):int(y2), int(x1):int(x2)] = 0

        old_features = cv2.goodFeaturesToTrack(old_gray, mask=mask_features, **self.features)

        for frame_num, frame in enumerate(frames):
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            new_features, status, err = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, old_features, None, **self.lk_params
            )

            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0


            if new_features is not None:
                good_new = new_features[status == 1]
                good_old = old_features[status == 1]


                m, _ = cv2.estimateAffinePartial2D(good_old, good_new)

                if m is not None:
                    camera_movement_x = m[0, 2]
                    camera_movement_y = m[1, 2]

                    for i, (new, old) in enumerate(zip(good_new, good_old)):
                        new_features_point = new.ravel()
                        old_features_point = old.ravel()
                        distance = np.linalg.norm(new_features_point - old_features_point)
                        if distance > max_distance:
                            max_distance = distance

                    if max_distance > 100:
                        camera_movement_x, camera_movement_y = 0, 0

                    camera_movement[frame_num] = [camera_movement_x, camera_movement_y]


                    mask_features = np.zeros_like(frame_gray)
                    mask_features[:] = 255

                    for bbox, _, _, class_id, _, _ in annotations[frame_num]:
                        if class_id == 2 or class_id == 3:
                            x1, y1, x2, y2 = bbox
                            mask_features[int(y1):int(y2), int(x1):int(x2)] = 0

                    old_features = cv2.goodFeaturesToTrack(frame_gray, mask=mask_features, **self.features)

            old_gray = frame_gray.copy()

        # 6. Zapisujemy wynik
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)

        return camera_movement