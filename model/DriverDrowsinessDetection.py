from os import path, makedirs

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical


class DriverDrowsinessDetection:
    def __init__(self, path_data_trained=None):
        if path_data_trained is not None:
            self.model = load_model(path_data_trained)
            # Load encoder nếu tồn tại
            label_path = path.join(path.dirname(path_data_trained), "label_encoder.pkl")
            if path.exists(label_path):
                self.le = joblib.load(label_path)
                print(f"✅ Loaded label encoder from {label_path}")
            else:
                print("⚠️ Không tìm thấy file label_encoder.pkl, cần fit lại encoder.")
        else:
            self.model = None
            self.history = None
            self.le = LabelEncoder()

    def init_model_to_train(self, total_class, img_size=64):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 1)),
            MaxPooling2D(2, 2),

            Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 1)),
            MaxPooling2D(2, 2),

            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),

            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(total_class, activation='softmax')  # số lớp = số nhãn
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        model.summary()
        self.model = model

    def train(self, x_train, y_train, epochs=15, batch_size=32):
        if self.model is None:
            print('Vui lòng khởi tạo mô hình trước khi train!')
            return
        y_train = self.le.fit_transform(y_train)
        y_train = to_categorical(y_train)
        self.history = self.model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size
        )

    def evaluate(self, x_test, y_test):
        if self.history is None:
            print('Vui lòng train mô hình trước khi đánh giá')
            return
        if self.model is None:
            print('Vui lòng khởi tạo mô hình trước khi train!')
            return
        y_test = self.le.transform(y_test)
        y_test = to_categorical(y_test)
        # Accuracy
        test_loss, test_acc = self.model.evaluate(x_test, y_test)
        print("Test Accuracy:", test_acc)

        # Biểu đồ loss/accuracy
        plt.plot(self.history.history['accuracy'], label='train acc')
        plt.legend()
        plt.show()

        # Dự đoán và in báo cáo
        y_pred = self.model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1)

        print(classification_report(y_true, y_pred_classes, target_names=self.le.classes_))

    def save_model(self, name_str="cnncat2.keras"):
        if self.model is None:
            print("Không có mô hình nào được khởi tạo!")
            return

        if path.dirname(name_str) != '':
            # Tạo thư mục nếu chưa có
            makedirs(path.dirname(name_str), exist_ok=True)

        # Lưu model
        self.model.save(name_str)

        # Lưu LabelEncoder
        label_path = path.join(path.dirname(name_str), "label_encoder.pkl")
        joblib.dump(self.le, label_path)

        print(f"✅ Saved model: {name_str}")
        print(f"✅ Saved label encoder: {label_path}")

    def predict_classes(self, image):
        if self.model is None:
            print("Model chưa được load!")
            return None

        pred = self.model.predict(image)
        pred_class = np.argmax(pred, axis=1)
        pred_label = self.le.inverse_transform(pred_class)
        return pred_label[0]

    def predict_max_index(self, image):
        if self.model is None:
            print("Model chưa được load!")
            return None

        pred = self.model.predict(image)
        return np.argmax(pred, axis=1)
