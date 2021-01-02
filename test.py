import cv2
import time

def preprocess(frame):
    stride = 50
    size = 200

    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 360))

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = img.shape
    for x in range((h - size) // stride):
        for y in range((w - size) // stride):
            img2 = img[(x * stride): size + (x * stride), (y * stride): size + (y * stride)]
            img2 = cv2.equalizeHist(img2)
            cv2.rectangle(frame, ((x * stride), (y * stride)), (size + (x * stride), size + (y * stride)), (0, 255, 0), 2)
            out.write(frame)

    out.release()


frame = cv2.imread('./data/anomal/0.bmp')
preprocess(frame)
cv2.destroyAllWindows()

# # 選擇第二隻攝影機
# cap = cv2.VideoCapture('test.avi')
#
# while(True):
#   ret, frame = cap.read()
#
#   cv2.imshow('frame', frame)
#
#   if cv2.waitKey(1) & 0xFF == ord('q'):
#     break
#
# # 釋放攝影機
# cap.release()
#
# # 關閉所有 OpenCV 視窗
# cv2.destroyAllWindows()
