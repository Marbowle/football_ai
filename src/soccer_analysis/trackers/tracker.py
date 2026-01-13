from supervision import ByteTrack
import supervision as sv

class Tracker:
    def __init__(self, lost_track_buffer=30):
        self.tracker = ByteTrack(lost_track_buffer=lost_track_buffer)

    def update_with_detections(self, detections):
        results = self.tracker.update_with_detections(detections)
        return results
