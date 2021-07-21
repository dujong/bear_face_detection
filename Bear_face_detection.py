import cv2, dlib
import numpy as np
import matplotlib.pyplot as plt

detector =  dlib.cnn_face_detection_model_v1('models/bearface_network.dat')
predictor = dlib.shape_predictor('models/landmarkDetector.dat')

img_path = 'imgs/01.jpg'
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(16,16))
plt.imshow(img)

dets = detector(img, upsample_num_times=1)
print(dets)

for det in dets:
    x1, y1 = det.rect.left(), det.rect.top()
    x2, y2 = det.rect.right(), det.rect.botton()

    print(x1, y1)
    print(x2, y2)

