import cv2
import tensorflow as tf
import numpy as np
from os import listdir
from os.path import join

model = tf.keras.models.load_model('./model/renet')


def cv2tf_img(img):
    img = np.asarray(img)
    img = np.expand_dims(img, axis=-1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def is_anomal(r, x, threshold):

    return np.square(r - x).mean() >= threshold


def predict(frame, threshold):
    stride = 50
    size = 100

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    h, w = img.shape
    num_row = ((h - size) // stride) + 1
    num_col = ((w - size) // stride) + 1

    for i in range(num_row):
        for j in range(num_col):
            eq_img = cv2.equalizeHist(img[i * stride: size + (i * stride), j * stride: size + (j * stride)])
            inp_img = cv2tf_img(eq_img)
            re_img = model.predict(tf.stack([inp_img]))
            if is_anomal(re_img, inp_img, threshold):
                cv2.rectangle(frame, (j * stride, i * stride), (size + (j * stride), size + (i * stride)), (0, 0, 255),
                              2)

    return frame


if __name__ == '__main__':
    fld = './data/nomal/'
    threshold = 0.058
    for f in listdir(fld):
        frame = cv2.imread(join(fld, f))
        frame = predict(frame, threshold)
        cv2.imshow('f', frame)
        cv2.waitKey(1)

    # cap = cv2.VideoCapture('test.avi')
    #
    # while True:
    #     ret, frame = cap.read()
    #     frame = predict(frame, threshold)
    #     cv2.imshow('frame', frame)
    #
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    #
    # # 釋放攝影機
    # cap.release()
    #
    # # 關閉所有 OpenCV 視窗
    # cv2.destroyAllWindows()
