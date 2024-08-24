import cv2
from PIL import Image
import os

def read_face_file(face_file_path):
    face_data = {}
    with open(face_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(', ')
            frame_num = int(parts[0])
            face_rect = list(map(int, parts[1:5]))
            eye_coords = list(map(float, parts[5:]))
            face_data[frame_num] = {'face': face_rect, 'eyes': eye_coords}
    return face_data

# def crop_faces(video_path, face_file_path):
#     try:
#         face_data = read_face_file(face_file_path)
#         cap = cv2.VideoCapture(video_path)
#         faces = []
#
#         if not cap.isOpened():
#             raise IOError("Error opening video file")
#
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#
#             frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
#             if frame_num in face_data:
#                 for face_info in face_data[frame_num]:
#                     left, top, right, bottom = face_info['face']
#                     face_region = frame[top:bottom, left:right]
#                     face_region = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
#                     face_region = Image.fromarray(face_region)
#                     faces.append(face_region)
#
#         cap.release()
#         return faces
#
#     except Exception as e:
#         print(f"An error occurred: {str(e)}")
#         return []

def crop_faces(video_path, face_file_path):
    face_data = read_face_file(face_file_path)
    cap = cv2.VideoCapture(video_path)
    faces = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if frame_num in face_data:
            left, top, right, bottom = face_data[frame_num]['face']
            face_region = frame[top:bottom, left:right]
            face_region = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
            face_region = Image.fromarray(face_region)
            faces.append(face_region)

    cap.release()
    return faces

def crop_all_faces_from_MSU_MFSD(directory):
    all_faces = {}
    for filename in os.listdir(directory):
        if filename.endswith(".mp4"):
            video_file_path = os.path.join(directory, filename)
            face_file_path = os.path.join(directory, filename[:-4] + ".face")

            if os.path.exists(face_file_path):
                faces = crop_faces(video_file_path, face_file_path)
                all_faces[filename] = faces
                print(f"Cropped {len(faces)} from {filename}")
            else:
                print(f"No face .file available for {filename}")
    return all_faces

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(script_dir, "images/MSU-MFSD/MSU-MFSD-Publish/scene01/real")
    faces = crop_all_faces_from_MSU_MFSD(image_dir)
    print(len(faces))
