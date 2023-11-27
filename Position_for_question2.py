import cv2
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import os
import unidecode

import pip
from numpy.array_api import astype


def find_mature_apple_centers_and_size(image_path):

    image_path = (image_path)
    image = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv_image, lower_red, upper_red)


    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    size = []
    centers=[]
    image_height=image.shape[0]
    apple_count = 0
    for contour in contours:
        M=cv2.moments(contour)
        if M["m00"]!=0:
            cX=int(M["m10"]/M["m00"])
            cY = image_height-int(M["m01"] / M["m00"])
            centers.append((cX,cY))
            area = cv2.contourArea(contour)/20
            size.append(area)
    return centers,size




all_centers =[]
all_transformed_centers=[]
all_sizes=[]
paths=[]
root_dir='Attachment/Attachment 1'
start_number,end_number=1,200
for number in range(start_number,end_number+1):
    path=os.path.join(root_dir,f'{number}.jpg')
    paths.append(path)

for path in paths:
    centers,size= find_mature_apple_centers_and_size(path)
    all_centers.extend(centers)
    all_sizes.extend(size)

for path in paths:
    image_height=cv2.imread(path)
    transformed_center=[(x,image_height-y) for (x,y) in centers]
    all_transformed_centers.extend(transformed_center)

fig=plt.figure()
ax=fig.add_subplot(111)
x_coord,y_coord = zip(*all_centers)
si=all_sizes

plt.scatter(x_coord,y_coord,s=5)

plt.title('Apple location scatter plot')
plt.xlabel('X')
plt.ylabel('Y')

plt.show()


