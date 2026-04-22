def load_labels(file_path):
    """
    Load class labels from file
    """
    with open(file_path, 'r') as f:
        labels = f.read().strip().split("\n")
    return labels