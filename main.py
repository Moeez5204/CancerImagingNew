from ImageGiver import get_random_image
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import kagglehub
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt


# Download dataset
dataset_path = kagglehub.dataset_download("adityamahimkar/iqothnccd-lung-cancer-dataset")
dataset_folder = os.path.join(dataset_path, "The IQ-OTHNCCD lung cancer dataset", "The IQ-OTHNCCD lung cancer dataset")
categories = ["Bengin cases", "Malignant cases", "Normal cases"]


# Prepare data arrays
data = []
labels = []


for category in categories:
   category_path = os.path.join(dataset_folder, category)
   for img_name in os.listdir(category_path):
       img_path = os.path.join(category_path, img_name)
       img = Image.open(img_path).resize((224, 224))
       data.append(np.array(img))
       labels.append(category)


# Convert lists to numpy arrays
X = np.array(data)
Y = np.array(labels)


# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# Normalize images
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0


# Label encoding
LE = LabelEncoder()
y_train_encoded = LE.fit_transform(y_train)
y_test_encoded = LE.transform(y_test)


# One-hot encoding
y_train_encoded = to_categorical(y_train_encoded, num_classes=3)
y_test_encoded = to_categorical(y_test_encoded, num_classes=3)


# Define model path
model_path = 'lung_cancer_model.h5'


# Load existing model if available
if os.path.exists(model_path):
   print("Loading existing model...")
   model = models.load_model(model_path)
else:
   # Define the model
   model = models.Sequential([
       layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
       layers.MaxPooling2D(2, 2),
       layers.Conv2D(64, (3, 3), activation='relu'),
       layers.MaxPooling2D(2, 2),
       layers.Conv2D(128, (3, 3), activation='relu'),
       layers.MaxPooling2D(2, 2),
       layers.Flatten(),
       layers.Dense(128, activation='relu'),
       layers.Dense(3, activation='softmax')
   ])


   model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])


   model.summary()


   # Train the model
   model.fit(X_train, y_train_encoded, epochs=10, validation_data=(X_test, y_test_encoded))
   model.save(model_path)
   print(f"Model saved to {model_path}")




# Prediction function
def predict_image(image_path):
   img = Image.open(image_path).resize((224, 224))
   img_array = np.array(img).astype('float32') / 255.0
   img_array = np.expand_dims(img_array, axis=0)


   pred = model.predict(img_array)
   pred_class = LE.inverse_transform([np.argmax(pred)])
   confidence = np.max(pred)


   print(f"Predicted class: {pred_class[0]} with confidence of {confidence:.2f}")


   # Show image
   plt.imshow(img)
   plt.title(f"Predicted: {pred_class[0]} ({confidence:.2f})")
   plt.axis('off')
   plt.show()




# 🚀 Test with a random image
def test_with_random_image():
   img, category, img_name, total = get_random_image()
   print(f"Testing Random Image: {img_name} from category {category}")


   img_resized = img.resize((224, 224))
   img_array = np.array(img_resized).astype('float32') / 255.0
   img_array = np.expand_dims(img_array, axis=0)


   pred = model.predict(img_array)
   pred_class = LE.inverse_transform([np.argmax(pred)])
   confidence = np.max(pred)


   print(f"Predicted class: {pred_class[0]} with confidence of {confidence:.2f}")


   # Show image
   plt.imshow(img)
   plt.title(f"Predicted: {pred_class[0]} ({confidence:.2f})")
   plt.axis('off')
   plt.show()



test_with_random_image()

#test