from torchvision import datasets
from torch.utils.data import DataLoader
from face import face_recognition
import cv2
import numpy as np


def who(unknown_image, db_name):
    # unknown_image = face_recognition.load_image_file(img_name_path)
    
    a = unknown_image.shape[1]
    b = unknown_image.shape[0]
    scale = max(200/a, 200/b)
    a = int(a * scale)
    b = int(b * scale)
    
    unknown_image = cv2.resize(unknown_image, (a,b), cv2.INTER_LINEAR)
    unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
    th = 0.5


    f = np.load(db_name)

    dists = [[np.linalg.norm(e1 - unknown_encoding)] for e1 in f['embed']]
    index = np.argmin(dists)
    
    if min(dists)[0] < th:
        return f['names'][index], min(dists)[0]
    return 'unknown', th
