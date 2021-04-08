
"""
GOAL OF THIS FILE

This code will ideally be used for the unsupervised training algorithm (in this case, K-means), which generates a bank of filters D
extract 16-by-16 pixel grayscale patches represented as a vector of 256 pixel intensities

I.E. Crop and scale images while keeping RGB channels 
"""

import numpy as np
from cv2 import cv2 #Use 'pip install opencv-python' to get module if you don't have it

#Hardcoding path to images. TODO - Don't hardcode

basepath = 'C:\\Users\\Admin\\321_galaxies\\'
imPath_test = 'images_training_rev1\\images_training_rev1\\'
imPath_training = 'images_training_rev1\\images_training_rev1\\'
pic_Index = '100008.jpg'

#Attempt to scale a single image

origImage = cv2.imread(basepath+imPath_training+pic_Index)

newx,newy = origImage.shape[1]/4,origImage.shape[0]/4 #new size (w,h)

scaled_image = cv2.resize(origImage,(int(newx),int(newy)))


#cv2.imshow("original image",origImage)
#cv2.imshow("resize image",newimage)


#Crop image from 424x424 to 160x160. To do we crop axes from {x=0, y=0, x=424, y=424 } => {x=132, y=132, x=292, 400=292}
crop_img = origImage[132:292, 132:292] 

cv2.imwrite("testCrop.jpg", crop_img)
cv2.imshow("cropped", crop_img)

cv2.waitKey(0)