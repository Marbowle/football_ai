import requests
from roboflow import Roboflow

# Tu wklej swój tajny klucz ze strony Roboflow
rf = Roboflow(api_key="Fo4B6OZ04dEHFti1xDX3")

# To jest adres projektu piłkarskiego, który znalazłaś
project = rf.workspace("roboflow-jvwxb").project("soccer-players-5fuqs")
version = project.version(1)

# To polecenie pobierze model w formacie YOLOv8
dataset = version.download("yolov8")