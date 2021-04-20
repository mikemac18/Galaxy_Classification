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
import glob

class GetImage(object):
    #Load raw image files
    
    #initialize the path (TODO - make basepath an input parameter, don't hardcode too much)

    #Data should be stored in Users/YOUR_USER_NAME/321_galaxies/images_training_rev1
    def __init__(self, username, num_pics=10):

        self.basepath = '/Users/'+username+'/321_galaxies/'
        self.imPath_test = 'images_test_rev1/'
        self.imPath_training =  'images_training_rev1/'
       
        #create variable that stores the image
        self.images = []
        i=0
        for filename in glob.glob(self.basepath + self.imPath_training+'*.jpg'): #assuming jpg
            if(i > num_pics):
                break
            im=cv2.imread(filename)
            self.images.append(im)
            i +=1 

        self.cropped_images = []
        self.scaled_images = []
        self.patches = []

    #Crop image from 424x424 to 160x160. To do we crop axes from {x=0, y=0, x=424, y=424 } => {x=132, y=132, x=292, 400=292}
    def crop(self, pixels_to_keep = 160):
        for galaxyPic in self.images:
            #Cast as int in order to use variables in the slicing indices
            center = int(424/2)
            dim = int(pixels_to_keep/2)
            cropmin = center - dim
            cropmax = center + dim
            self.cropped_images.append(galaxyPic[cropmin:cropmax, cropmin:cropmax])
        return self

    #function that takes an image and scales the entire image to its new size (unlike crop, image stays intact). 
    def scale(self, new_size = 16):
        for galaxyPic in self.cropped_images:
            dimensions = (int(new_size), int(new_size))
            self.scaled_images.append(cv2.resize(galaxyPic, dimensions))
        return self
    
    #function that crops out random 4x4 patches of an image.
    def patch(self, p_size=4, num_patches_per_image = 5):
        for i in range(0, num_patches_per_image):
            for galaxyPic in self.scaled_images:
                patch_x = np.random.randint(2, 14)
                patch_y = np.random.randint(2, 14)
                dim = int(p_size/2)
                patchminx = patch_x - dim
                patchminy = patch_y - dim
                patchmaxx = patch_x + dim
                patchmaxy = patch_y + dim
                self.patches.append(galaxyPic[patchminx:patchmaxx, patchminy:patchmaxy])
        return self


#Testing the class by creating a 'galaxyPic' object, cropping it, scaling it, and saving the new image
galaxyPics = GetImage('samsonmercier/Desktop')
galaxyPics.crop()
galaxyPics.scale()
galaxyPics.patch()

i=0
for patch in galaxyPics.patches:
    cv2.imwrite('patches/patchno'+str(i)+'.jpg', patch)
    i += 1
    #cv2.waitKey(0)
