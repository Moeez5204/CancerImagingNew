import kagglehub
import os
import random
from PIL import Image

def get_random_image():
    # Download dataset
    dataset_path = kagglehub.dataset_download("adityamahimkar/iqothnccd-lung-cancer-dataset")
    dataset_folder = os.path.join(dataset_path, "The IQ-OTHNCCD lung cancer dataset", "The IQ-OTHNCCD lung cancer dataset")
    categories = ["Bengin cases", "Malignant cases", "Normal cases"]

    # Randomly pick category
    chosen_category = random.choice(categories)
    category_path = os.path.join(dataset_folder, chosen_category)
    images = os.listdir(category_path)
    if not images:
        raise Exception(f"No images found in {category_path}!")

    chosen_image = random.choice(images)
    image_path = os.path.join(category_path, chosen_image)
    img = Image.open(image_path)

    return img, chosen_category, chosen_image, len(images)