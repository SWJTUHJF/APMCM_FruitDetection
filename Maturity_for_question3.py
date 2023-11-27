import cv2
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import os
import unidecode

import pip
from numpy.array_api import astype


def calculate_apple_ripeness_with_shape_exclusion(image_path):

    image_path = (image_path)
    image = cv2.imread(image_path)
    image_rgb=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    lower_red = np.array([100, 0, 0])
    upper_red = np.array([255, 100, 100])
    lower_green = np.array([0, 100, 0])

    upper_green = np.array([100, 255, 100])
    mask_red = cv2.inRange(image_rgb, lower_red, upper_red)
    mask_green = cv2.inRange(image_rgb, lower_green, upper_green)

    contours, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        x,y,w,h = cv2.boundingRect(contour)
        if w>0 and h>0:
            aspect_radio=w/h
            if aspect_radio>0.8 and aspect_radio<1.2 and area>100:
                cv2.drawContours(mask_green,[contour],-1,(255,255,255),-1)
            else:
                cv2.drawContours(mask_green, [contour], -1, (0, 0, 0), -1)

    red_count=np.sum(mask_red>0)
    green_count=np.sum(mask_green>0)

    ripness_score=red_count/(red_count+green_count+0.00000001)

    return ripness_score
paths=[]
ripeness_with_shape_exclusion=[]
root_dir='Attachment/Attachment 1'
start_number,end_number=1,200
for number in range(start_number,end_number+1):
    path=os.path.join(root_dir,f'{number}.jpg')
    paths.append(path)

for path in paths:
    score=calculate_apple_ripeness_with_shape_exclusion(path)
    ripeness_with_shape_exclusion.append(score)


plt.hist(ripeness_with_shape_exclusion, bins=20, color='blue', alpha=0.7)
plt.title('Apple maturity distribution')
plt.xlabel('maturity')
plt.ylabel('number')
plt.show()

