
"""
GOAL OF THIS FILE

This code will ideally be used for the unsupervised training algorithm (in this case, K-means), which generates a bank of filters D
extract 16-by-16 pixel grayscale patches represented as a vector of 256 pixel intensities

I.E. Crop and scale images while keeping RGB channels 

Implementing method outlined in this paper:

https://www-cs.stanford.edu/~acoates/papers/coatesng_nntot2012.pdf

"""

import numpy as np
from cv2 import cv2 #Use 'pip install opencv-python' to get module if you don't have it

class GetImage(object):
    #Load raw image files
    
    #initialize the path (TODO - make basepath an input parameter, don't hardcode too much)
    def __init__(self, pic_Index='100008.jpg'):

        self.basepath = '/Users/samsonmercier/Desktop/PHYS321_FinalProj/Image_class/'
        self.imPath_test = 'Raw/images_test_rev1/'
        self.imPath_training = 'Raw/images_training_rev1/'
        self.pic_Index = pic_Index

        #create variable that stores the image
        self.image = cv2.imread(self.basepath+self.imPath_training+self.pic_Index)

    #Crop image from 424x424 to 160x160. To do we crop axes from {x=0, y=0, x=424, y=424 } => {x=132, y=132, x=292, 400=292}
    def crop(self, pixels_to_keep = 160):

        #Cast as int in order to use variables in the slicing indices
        center = int(424/2)
        dim = int(pixels_to_keep/2)
        cropmin = center - dim
        cropmax = center + dim
        self.image = self.image[cropmin:cropmax, cropmin:cropmax]
        return self

    #function that takes an image and scales the entire image to its new size (unlike crop, image stays intact). 
    def scale(self, new_size = 16):
        dimensions = (int(new_size), int(new_size))
        self.image = cv2.resize(self.image, dimensions)
        return self
    
    #function that crops out random 4x4 patches of an image.
    def patch(self, p_size=4):
        patch_x = np.random.randint(2, 14)
        patch_y = np.random.randint(2, 14)
        dim = int(p_size/2)
        patchminx = patch_x - dim
        patchminy = patch_y - dim
        patchmaxx = patch_x + dim
        patchmaxy = patch_y + dim
        patch = self.image[patchminx:patchmaxx, patchminy:patchmaxy]
        self.image = patch
        return self.image

"""
#Testing the class by creating a 'galaxyPic' object, cropping it, scaling it, and saving the new image

galaxyPic = GetImage()
galaxyPic.crop()
galaxyPic.scale()

cv2.imwrite("testCrop-and-Scale.jpg", galaxyPic.image)
cv2.waitKey(0)
"""
