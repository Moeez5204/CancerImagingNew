import numpy as np
import matplotlib.pyplot as plt
from utils import get_image_from_dialog
from ImageGiver import get_random_image

def predict_image(model, LE, img):
    img_resized = img.resize((224,224))
    img_array = np.array(img_resized).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    pred_class = LE.inverse_transform([np.argmax(pred)])
    confidence = np.max(pred)
    return pred_class[0], confidence

def manual_test(model, LE):
    result = get_image_from_dialog()
    if result is None:
        return
    img, path = result
    pred_class, confidence = predict_image(model, LE, img)
    print(f"Manual Test - {path}: Predicted {pred_class} ({confidence:.2f})")
    plt.imshow(img)
    plt.title(f"Predicted: {pred_class} ({confidence:.2f})")
    plt.axis('off')
    plt.show()

def random_test(model, LE):
    img, category, img_name, total = get_random_image()
    pred_class, confidence = predict_image(model, LE, img)
    print(f"Random Test - {img_name} from {category}: Predicted {pred_class} ({confidence:.2f})")
    plt.imshow(img)
    plt.title(f"Predicted: {pred_class} ({confidence:.2f})")
    plt.axis('off')
    plt.show()