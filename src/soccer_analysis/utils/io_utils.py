import pandas as pd

def save_tracks_to_csv(tracks, output_path):
    data = []

    for object_type, object_tracks in tracks.items():
        for frame_num, frame_data in enumerate(object_tracks):
            for track_id, track_info in frame_data.items():
                row = {
                    'frame_num': frame_num,
                    'object_type': object_type,
                    'track_id': track_id,
                    'team_id': track_info.get('team_id', None),
                    'map_x': None,
                    'map_y': None
                }

                if 'position_transformed' in track_info:
                    position = track_info['position_transformed']
                    row['map_x'] = position[0]
                    row['map_y'] = position[1]

                data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print("Data saved to {}".format(output_path))