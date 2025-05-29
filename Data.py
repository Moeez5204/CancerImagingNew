import kagglehub
import os

def download_dataset():
    path = kagglehub.dataset_download("adityamahimkar/iqothnccd-lung-cancer-dataset")
    print("Path to dataset files:", path)
    return path

def get_class_paths(root_path):
    return {
        "Normal": os.path.join(root_path, "Normal"),
        "Benign": os.path.join(root_path, "Benign"),
        "Malignant": os.path.join(root_path, "Malignant")
    }