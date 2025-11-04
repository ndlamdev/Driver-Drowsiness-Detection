from os import path, listdir

import cv2
import kagglehub
import numpy as np
from keras.src.utils import to_categorical
from sklearn.preprocessing import LabelEncoder


def load_data(data_dir, img_size = 64):
    images = []
    labels = []
    for label_name in listdir(data_dir):
        label_path = path.join(data_dir, label_name)
        for img_file in listdir(label_path):
            img_path = path.join(label_path, img_file)
            # đọc ảnh bằng OpenCV
            img = cv2.imread(img_path)
            if img is None:
                continue
            # chuyển ảnh xám
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # resize
            gray = cv2.resize(gray, (img_size, img_size))
            images.append(gray)
            labels.append(label_name)
    return np.array(images), np.array(labels)

# Download latest version
root_path = kagglehub.dataset_download("serenaraju/yawn-eye-dataset-new")
root_path = path.join(root_path, 'dataset_new')
print("Path to dataset files:", root_path)

train_dir =  path.join(root_path, 'train')
test_dir =  path.join(root_path, 'test')


x_train, y_train = load_data(train_dir)
x_test, y_test = load_data(test_dir)

img_size = 64

# reshape dữ liệu (CNN cần 4D: [samples, height, width, channels])
x_train = x_train.reshape(-1, img_size, img_size, 1) / 255.0
x_test = x_test.reshape(-1, img_size, img_size, 1) / 255.0

# encode nhãn sang số
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# one-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
