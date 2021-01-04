from torchvision import datasets
from torch.utils.data import DataLoader
from face import face_recognition
# import face_recognition
import cv2
import numpy as np

face_data_path = 'database'


def collate_fn(x):
  return x[0]


dataset = datasets.ImageFolder(face_data_path)
dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}

loader = DataLoader(dataset, collate_fn=collate_fn)
embeddings = []
names = []

for x, y in loader:
  x = np.array(x)
  encoding = face_recognition.face_encodings(x)[0]
  embeddings.append(encoding)
  names.append(dataset.idx_to_class[y])

np.savez('db.npz', embed=np.array(embeddings), names=np.array(names))