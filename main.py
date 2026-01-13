import argparse
import os
from src.soccer_analysis.game_analyzer import GameAnalyzer

def main():
    # Configuration for arguments parser
    parser = argparse.ArgumentParser("System analizy piłkarskiej")
    parser.add_argument('--source_video_path', type=str, required=True, help='source video path')
    parser.add_argument('--model_path', default='models/best.pt', type=str, help='ścieżka do modelu')

    args = parser.parse_args()

    #Paths setup
    path = args.source_video_path
    model_path = args.model_path

    video_name = os.path.basename(path)
    video_name = os.path.splitext(video_name)[0]

    output_dir = "validation videos"
    os.makedirs(output_dir, exist_ok=True)

    output_filename = f"{video_name}_analiza.mp4"
    video_output_path = os.path.join(output_dir, output_filename)

    game_analyzer = GameAnalyzer(path, model_path)
    game_analyzer.process_video(video_output_path)

if __name__ == '__main__':
    main()



