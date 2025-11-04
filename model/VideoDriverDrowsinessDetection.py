import cv2
import numpy as np
from pygame import mixer

from model.DriverDrowsinessDetection import DriverDrowsinessDetection


class VideoDriverDrowsinessDetection:
    def __init__(self, model: DriverDrowsinessDetection = None):
        if model is None:
            print("Model không thể bị null")
            return
        self.model = model
        mixer.init()
        self.sound = mixer.Sound('data/alarm.wav')
        self.face = cv2.CascadeClassifier('data/data-haarcascades/haarcascade_frontalface_alt.xml')
        self.leye = cv2.CascadeClassifier('data/data-haarcascades/haarcascade_lefteye_2splits.xml')
        self.reye = cv2.CascadeClassifier('data/data-haarcascades/haarcascade_righteye_2splits.xml')

    def predict_object(self, frame, obj, img_size=64):
        count_open = 0
        count_close = 0
        for (x, y, w, h) in obj:
            eye = frame[y:y + h, x:x + w]
            eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
            eye = cv2.resize(eye, (img_size, img_size))
            eye = eye / 255
            eye = eye.reshape(img_size, img_size, -1)
            eye = np.expand_dims(eye, axis=0)
            result = self.model.predict_classes(eye)
            if result == "Open":
                count_open += 1
            if result == "Close":
                count_close += 1

        if count_open > count_close:
            return "Open"
        else:
            return "Close"

    def start(self, cam=1, img_size=64):
        cap = cv2.VideoCapture(cam)
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        score = 0
        thicc = 2
        while (True):
            ret, frame = cap.read()
            height, width = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
            left_eye = self.leye.detectMultiScale(gray)
            right_eye = self.reye.detectMultiScale(gray)
            cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)
            lpred = self.predict_object(frame, left_eye, img_size)
            rpred = self.predict_object(frame, right_eye, img_size)
            if rpred == "Close" and lpred == "Close":
                score = score + 1
                if score > 30:
                    score = 30
                cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                score = score - 1
                cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            if score < 0:
                score = 0
            cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            if score > 15:
                try:
                    if self.sound.get_num_channels() == 0:
                        self.sound.play(loops=-1)  # -1 = lặp vô hạn, 0 = chỉ phát 1 lần
                except:
                    pass
                if thicc < 16:
                    thicc = thicc + 2
                else:
                    thicc = thicc - 2
                    if thicc < 2:
                        thicc = 2
                cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
            else:
                self.sound.stop()
            cv2.imshow('Phần mềm phát hiện nhấm mắt', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
