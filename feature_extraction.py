from preprocess import crop_all_faces_from_MSU_MFSD
import os
from transformers import MobileViTImageProcessor, MobileViTModel
from PIL import Image
import torch
import numpy as np
from scipy import stats

#Load Pre-trained MobileViT
model_name = "apple/mobilevit-small"
mobilevit = MobileViTModel.from_pretrained(model_name)
image_processor = MobileViTImageProcessor.from_pretrained(model_name)

def extract_features_from_paths(image_paths):
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

def extract_features_from_faces(face_dict):
    features = {}
    face_dict = {k: face_dict[k] for k in list(face_dict)[:10]}

    for video_name, faces in face_dict.items():
        video_features = []
        for face in faces:
            try:
                if isinstance(face, str):
                    face = Image.open(face).convert("RGB")
                elif isinstance(face, Image.Image):
                    face = face.convert("RGB")
                else:
                    raise TypeError(f"Unexpected type for face: {type(face)}")

                inputs = image_processor(images=face, return_tensors="pt")
                # inputs = image_processor(images=face.resize((224, 224)), return_tensors="pt")
                with torch.no_grad():
                    outputs = mobilevit(**inputs)
                    video_features.append(outputs.pooler_output.squeeze().numpy())
            except Exception as e:
                print(f"Error processing a face: {str(e)}")
        features[video_name] = video_features
        print(f"{video_name}: {len(video_features)} features extracted")

    return features

# Returns the mode of the distribution among features.  This is because SVM requires an equal number of features for all faces. The mode is to be used to filter out the faces.
def analyze_feature_distribution(features):
    # Extract the number of features for each video
    feature_counts = [len(v) for v in features.values()]

    # Compute the mode of these feature counts
    mode_result = stats.mode(feature_counts)

    # Handle both old and new SciPy versions
    if isinstance(mode_result, tuple):  # Old SciPy version
        mode_count = mode_result[0]
        mode_frequency = mode_result[1]
    else:  # New SciPy version
        mode_count = mode_result.mode
        mode_frequency = mode_result.count

    # Ensure mode_count and mode_frequency are scalar
    mode_count = mode_count.item() if hasattr(mode_count, 'item') else mode_count
    mode_frequency = mode_frequency.item() if hasattr(mode_frequency, 'item') else mode_frequency

    # Calculate the distribution of feature counts
    unique, counts = np.unique(feature_counts, return_counts=True)
    distribution = dict(zip(unique, counts))

    print("Distribution of feature counts:", distribution)
    print("Most common number of features extracted:", mode_count)
    print("Frequency of this count:", mode_frequency)

    return mode_count

def filter_by_mode(mode_count, features_dict):
    filtered_features = {}
    for video_name, features in features_dict.items():
        if (len(features) == mode_count):
            filtered_features[video_name] = features
    return filtered_features

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(script_dir, "images/MSU-MFSD/MSU-MFSD-Publish/scene01/real")
    faces_dict = crop_all_faces_from_MSU_MFSD(image_dir)
    features_dict = extract_features_from_faces(faces_dict)
    mode_count = analyze_feature_distribution(features_dict)
    filtered_dict = filter_by_mode(mode_count, features_dict)

    for video_name, features in features_dict.items():
        print(f"{video_name}: extracted features from {len(features)} faces")

    # Total number of faces processed
    total_faces = sum(len(features) for features in features_dict.values())
    filtered_faces = sum(len(features) for features in filtered_dict.values())
    print(f"Total faces processed: {total_faces}")
    print(f"Total filtered faces processed: {filtered_faces}")
    print(f"Feature mode: {mode_count}")
