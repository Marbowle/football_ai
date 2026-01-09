from supervision import ByteTrack

def create_tracker():
    byte_track = ByteTrack(lost_track_buffer=30)
    return byte_track