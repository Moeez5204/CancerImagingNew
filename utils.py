from ImageGiver import get_random_image
from tkinter import Tk, filedialog
from PIL import Image

def get_image_from_dialog():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select lung cancer image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if not file_path:
        print("No file selected")
        return None
    img = Image.open(file_path).convert('RGB')
    return img, file_path