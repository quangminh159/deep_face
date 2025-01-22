from deepface import DeepFace
import numpy as np
import cv2
from mtcnn import MTCNN
def cosine_similarity(embedding1, embedding2):
    e1 = np.array(embedding1)
    e2 = np.array(embedding2)
    return np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
def get_embedding(image_path, model_name="Facenet512"):
    embedding = DeepFace.represent(img_path=image_path, model_name=model_name, enforce_detection=False)[0]["embedding"]
    return embedding
def detect_and_crop_face(image_path):
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detector = MTCNN()
    results = detector.detect_faces(rgb_image)
    # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # faces = face_cascade.detectMultiScale(rgb_image, scaleFactor=1.1, minNeighbors=5)
    if results:
        x, y, width, height = results[0]['box']
        x, y = max(0, x), max(0, y)  
        face = image[y:y + height, x:x + width]
        return face, (x, y, width, height)
    return None, None
def compare_images(image_path1, image_path2, model_name="Facenet512", threshold=0.5):
    embedding1 = DeepFace.represent(img_path=image_path1, model_name=model_name, enforce_detection=False)[0]["embedding"]
    embedding2 = DeepFace.represent(img_path=image_path2, model_name=model_name, enforce_detection=False)[0]["embedding"]
    similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    is_similar = similarity > threshold
    return similarity, is_similar
