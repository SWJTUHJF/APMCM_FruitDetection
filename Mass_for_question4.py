import math

import cv2
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import os
import unidecode

import pip
from numpy.array_api import astype


def find_mature_apple_quality(image_path):

    image_path = (image_path)
    image = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv_image, lower_red, upper_red)


    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    size = []
    centers=[]
    radiuss=[]
    qualitys=[]
    image_height=image.shape[0]
    apple_count = 0
    for contour in contours:
        M=cv2.moments(contour)
        if M["m00"]!=0:
            cX=int(M["m10"]/M["m00"])
            cY = image_height-int(M["m01"] / M["m00"])
            centers.append((cX,cY))
            area = cv2.contourArea(contour)/100
            radius=math.sqrt(area/3.1415926)
            size.append(area)
            radiuss.append(radius)
            quality=0.9*1.33333*radius*radius*radius*3.1415926
            qualitys.append(quality)
    return qualitys




all_centers =[]
all_transformed_centers=[]
all_sizes=[]
paths=[]
all_qualitys=[]
root_dir='Attachment/Attachment 1'
start_number,end_number=1,200
for number in range(start_number,end_number+1):
    path=os.path.join(root_dir,f'{number}.jpg')
    paths.append(path)

for path in paths:
    qualitys= find_mature_apple_quality(path)
    all_qualitys.append(qualitys)


plt.hist(all_qualitys, bins=20, alpha=0.7)
plt.xlim(0,120)
plt.title('Apple mass distribution')
plt.xlabel('mass')
plt.ylabel('number')
plt.show()