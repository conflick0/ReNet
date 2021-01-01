import cv2
from os import listdir
from os.path import join, dirname

project_root = dirname(dirname(__file__))
input_folder = join(project_root, 'data/anomal_data/anomal')
output_folder = join(project_root, 'data2/anomal_data/anomal')
files = listdir(input_folder)

stride = 50
size = 200
i = 0

for f in files:
    if i > 10000: break
    img = cv2.imread(join(input_folder, f), 0)
    h, w = img.shape
    for x in range((h - size) // stride):
        for y in range((w - size) // stride):
            img2 = img[(x * stride): size + (x * stride), (y * stride): size + (y * stride)]
            img2 = cv2.equalizeHist(img2)
            cv2.imwrite(join(output_folder, f'{i}.bmp'), img2)
            print(i)
            i += 1
