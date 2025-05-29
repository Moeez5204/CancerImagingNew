import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from PIL import Image
import kagglehub

def load_data():
    dataset_path = kagglehub.dataset_download("adityamahimkar/iqothnccd-lung-cancer-dataset")
    dataset_folder = os.path.join(dataset_path, "The IQ-OTHNCCD lung cancer dataset", "The IQ-OTHNCCD lung cancer dataset")
    categories = ["Bengin cases", "Malignant cases", "Normal cases"]

    data, labels = [], []
    for category in categories:
        category_path = os.path.join(dataset_folder, category)
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = Image.open(img_path).resize((224, 224))
            data.append(np.array(img))
            labels.append(category)

    X = np.array(data).astype('float32') / 255.0
    Y = np.array(labels)
    return train_test_split(X, Y, test_size=0.2, random_state=42), LabelEncoder()