import cv2
import os
import numpy as np

def load_images(path, size=(100, 100)):
    data = []
    labels = []
    label_map = {}
    label_id = 0

    for person in os.listdir(path):
        label_map[label_id] = person
        person_path = os.path.join(path, person)

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, size)
            img = img.flatten()

            data.append(img)
            labels.append(label_id)

        label_id += 1

    return np.array(data), np.array(labels), label_map