import os

def save_labels(labels, out_path="models/labels.txt"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for l in labels:
            f.write(l + "\n")

def load_labels(path="models/labels.txt"):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return [l.strip() for l in f.readlines() if l.strip()]