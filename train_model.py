import cv2
import os
import numpy as np

dataset_path = "dataset"
faces =[]
labels = []
label_map = {}
current_label = 0

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

for person_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_name)

    if not os.path.isdir(person_path):
        continue

    label_map[current_label] = person_name

    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)

        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in detected_faces:
            face = gray[y:y+h, x:x+w]
            faces.append(face)
            labels.append(current_label)

    current_label += 1

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))
recognizer.save('trainer_model.yml')

print("Training Complete!")
print("Label Map:", label_map)