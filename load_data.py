import os

DATA_DIR = "data"

def get_classes_and_counts(data_dir=DATA_DIR):
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    counts = {c: len([f for f in os.listdir(os.path.join(data_dir,c)) if os.path.isfile(os.path.join(data_dir,c,f))]) for c in classes}
    return classes, counts

if __name__ == "__main__":
    classes, counts = get_classes_and_counts()
    print("Classes found:", classes)
    print("Counts per class:")
    for k,v in counts.items():
        print(f"  {k}: {v}")