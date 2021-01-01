import cv2
from os import path

project_root = path.dirname(path.dirname(__file__))
output_folder = path.join(project_root, 'raw_data')
video_file = 'test.avi'

cap = cv2.VideoCapture(video_file)
i = 0

while True:
    ret, frame = cap.read()

    if not ret or (cv2.waitKey(1) & 0xFF == ord('q')):
        break

    cv2.imwrite(path.join(output_folder, f'{i}.bmp'), frame)
    i += 1

cap.release()
cv2.destroyAllWindows()
