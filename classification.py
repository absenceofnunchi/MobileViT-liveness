from feature_extraction import extract_features_from_faces
from preprocess import crop_all_faces_from_MSU_MFSD
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import numpy as np
import os
import resource
import itertools

def prepare_data_for_classification(faces_dict, label):
    target_size = (256, 256)  # Match the expected image_size of the Mobile Vision Transformer

    for features in faces_dict.values():
        for img in features:
            img_resized = img.resize(target_size)
            img_array = np.array(img_resized) / 255.0  # Normalize to [0, 1]
            yield img_array, label

def main():
    # The maximum amount of memory you want to allocate in bytes
    # For example, 8 GB would be 8 * 1024 * 1024 * 1024
    soft, hard = resource.getrlimit(resource.RLIMIT_DATA)
    print(f"Soft limit: {soft / (1024**3):.2f} GB")
    print(f"Hard limit: {hard / (1024**3):.2f} GB")
    resource.setrlimit(resource.RLIMIT_DATA, (hard, hard))

    script_dir = os.path.dirname(os.path.abspath(__file__))
    real_images_dir = os.path.join(script_dir, "images/MSU-MFSD/MSU-MFSD-Publish/scene01/real")
    real_faces_dict = crop_all_faces_from_MSU_MFSD(real_images_dir)

    attack_faces_dir = os.path.join(script_dir, "images/MSU-MFSD/MSU-MFSD-Publish/scene01/attack")
    attack_faces_dict = crop_all_faces_from_MSU_MFSD(attack_faces_dir)

    print(f"real faces: {len(real_faces_dict)}")
    print(f"attack faces: {len(attack_faces_dict)}")

    real_gen = prepare_data_for_classification(real_faces_dict, 1)
    attack_gen = prepare_data_for_classification(attack_faces_dict, 0)

    print("rael_gen, attack_gen")

    # Combine generators
    combined_gen = itertools.chain(real_gen, attack_gen)

    print("combined_gen")

    # Shuffle the combined generator
    combined_list = list(combined_gen)
    np.random.shuffle(combined_list)

    print("combined_list")

    # Split the data
    split_index = int(0.8 * len(combined_list))
    train_data = combined_list[:split_index]
    test_data = combined_list[split_index:]

    print("train test data")

    X_train, y_train = zip(*train_data)
    X_test, y_test = zip(*test_data)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Reshape the data
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    svm_classifier = SVC(kernel='rbf', C=1.0)
    svm_classifier.fit(X_train, y_train)

    y_pred = svm_classifier.predict(X_test)

    print(classification_report(y_test, y_pred, target_names=["Attack", "Real"]))


if __name__ == "__main__":
    main()
