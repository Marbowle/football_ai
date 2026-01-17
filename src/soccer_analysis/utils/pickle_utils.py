import pickle

def save_detections(detections, filename):
    with open(filename, 'wb') as f:
        pickle.dump(detections, f)

def load_detections(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)