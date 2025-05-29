from data_loader import load_data
from model import build_model
from prediction import manual_test, random_test
from sklearn.preprocessing import LabelEncoder
import os
from tensorflow.keras.utils import to_categorical

# Load data
(X_train, X_test, y_train, y_test), LE = load_data()
y_train_encoded = LE.fit_transform(y_train)
y_test_encoded = LE.transform(y_test)
y_train_encoded = to_categorical(y_train_encoded, num_classes=3)
y_test_encoded = to_categorical(y_test_encoded, num_classes=3)

model_path = 'lung_cancer_model.h5'

# Load or train model
if os.path.exists(model_path):
    from tensorflow.keras.models import load_model
    model = load_model(model_path)
    print("Model loaded from file.")
else:
    model = build_model()
    model.fit(X_train, y_train_encoded, epochs=10, validation_data=(X_test, y_test_encoded))
    model.save(model_path)
    print("Model trained and saved.")

# Run predictions
manual_test(model, LE)
# random_test(model, LE)