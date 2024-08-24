import os
from transformers import MobileViTImageProcessor, MobileViTModel
from PIL import Image
import torch
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np

# --- 1. Load Pre-trained MobileViT ---
model_name = "apple/mobilevit-small"
mobilevit = MobileViTModel.from_pretrained(model_name)
image_processor = MobileViTImageProcessor.from_pretrained(model_name)

# --- 2. Data Loading and Feature Extraction ---
def extract_features(image_paths):
    features = []
    for image_path in image_paths:
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = image_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = mobilevit(**inputs)
                features.append(outputs.pooler_output.squeeze().numpy())
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
    return features

# --- Replace with your actual data paths ---
image_dir = "/Users/jeff/Documents/Code/RideFlag/liveness/mobilevit/images"
bona_fide_dir = os.path.join(image_dir, "bona_fide")
spoof_dir = os.path.join(image_dir, "spoof")

bona_fide_paths = [os.path.join(bona_fide_dir, f) for f in os.listdir(bona_fide_dir) if os.path.isfile(os.path.join(bona_fide_dir, f))]
spoof_paths = [os.path.join(spoof_dir, f) for f in os.listdir(spoof_dir) if os.path.isfile(os.path.join(spoof_dir, f))]

print("Number of bona fide images:", len(bona_fide_paths))
print("Number of spoof images:", len(spoof_paths))

# Extract features
bona_fide_features = extract_features(bona_fide_paths)
spoof_features = extract_features(spoof_paths)

print("Shape of bona_fide_features:", np.array(bona_fide_features).shape)
print("Shape of spoof_features:", np.array(spoof_features).shape)

# Create labels
bona_fide_labels = [0] * len(bona_fide_features)
spoof_labels = [1] * len(spoof_features)

print("Number of bona fide samples:", len(bona_fide_labels))
print("Number of spoof samples:", len(spoof_labels))

# Combine features and labels CORRECTLY
features = np.concatenate((bona_fide_features, spoof_features), axis=0)
labels = np.concatenate((bona_fide_labels, spoof_labels), axis=0)

# --- 3. Data Splitting ---
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

# --- 4. Train SVM Classifier ---
svm_classifier = SVC(kernel='linear', C=1.0)
svm_classifier.fit(X_train, y_train)

# --- 5. Evaluation ---
y_pred = svm_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
