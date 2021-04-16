"""
GOAL OF THIS FILE
This code will ideally be used for the unsupervised training algorithm (in this case, K-means), which generates a bank of filters D
extract 16-by-16 pixel grayscale patches represented as a vector of 256 pixel intensities
I.E. Crop and scale images while keeping RGB channels 

FOR THIS FILE WE ARE ATTEMPTING TO NORMALIZE AND WHITEN OUR PATCHES
Implementing method outlined in this paper:
https://www-cs.stanford.edu/~acoates/papers/coatesng_nntot2012.pdf
"""

import numpy as np
from cv2 import cv2 #Use 'pip install opencv-python' to get module if you don't have it
import glob
import os

class GetImage(object):
    #Load raw image files
    
    #initialize the path (TODO - make basepath an input parameter, don't hardcode too much)

    #Data should be stored in Users/YOUR_USER_NAME/321_galaxies/images_training_rev1
    def __init__(self, username, num_pics=100, username='samsonmercier/Desktop'):

        self.basepath = '/Users/'+username+'/321_galaxies/'
        self.imPath_test = 'images_test_rev1/'
        self.imPath_training =  'images_training_rev1/'
        self.solPath = 'trainingsolutions_rev1.csv'
        self.solutions=np.loadtxt(self.basepath + self.solPath, delimiter=',', skiprows=1)[:num_pics, 1:4]
       
        #create variable that stores the image
        self.images = []
        i=0
        for filename in glob.glob(self.basepath + self.imPath_training+'*.jpg'): #assuming jpg
            if(i >= num_pics):
                break
            im=cv2.imread(filename)
            self.images.append(im)
            i +=1 
        self.cropped_images = []
        self.scaled_images = []
        self.patches = []
       
    #Get category labels for each image from the solutions file
    def getLabels(self):
        self.label=[]
        for i in self.solutions:
            self.m = np.where(i == max(i))[0][0]
            if self.m == 0:
                label.append('Category1')
            if self.m == 1:
                label.append('Category2')
            if self.m == 2:
                label.append('Category3')
        self.label=np.array(self.label)
        return self
    
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
    
    #function that crops out the four 8x8 patches that compose each image.
    def patch(self, p_size=8):
        for galaxyPic in self.scaled_images:
            #Gets the center for each of the four patches
            for i in (4, 12):
                for j in (4, 12):
                    patch_x = i
                    patch_y = j
                    dim = int(p_size/2)
                    patchminx = patch_x - dim
                    patchminy = patch_y - dim
                    patchmaxx = patch_x + dim
                    patchmaxy = patch_y + dim
                    self.patches.append(galaxyPic[patchminx:patchmaxx, patchminy:patchmaxy])
        return self
    #function that sums the values for the different filters and
    #flatten the 8x8 matrixes for each patch to a 1x64 vector
    def makevector(self):
        self.newpatches = np.sum(self.patches, axis = 3)
        self.patchvector = []
        for i in self.newpatches:
            self.patchvector.append(np.matrix.flatten(i).tolist())
        self.patchvector=np.array(self.patchvector)
        return self
    
        
        ## Function that normalizes the pixels. Takes vstack of patches as input
    def normalize(self):
        temp1 = self.patchvector - self.patchvector.mean(1, keepdims=True)
        temp2 = np.sqrt(self.patchvector.var(1, keepdims=True) + 10)
        self.patchvector = temp1/temp2
        return self
    ## Function that runs ZCA whitening
    def whiten(self):
        cov = np.cov(self.patchvector, rowvar=0)
        self.mean = self.patchvector.mean(0, keepdims=True)
        d, v = np.linalg.eig(cov)
        self.p = np.dot(v, np.dot(np.diag(np.sqrt(1 / (d + 0.1))), v.T))
        self.patchvector = np.dot(self.patchvector - self.mean, self.p)
        return self
    



#To avoid empty clusters eed to initialize clusters from a Normal distribution 
#and normalize them to unit length?
def Kmeans(L, k = 10, iters = 50, batch_size = 40):
    L2 = np.sum(L**2, 1, keepdims=True)
    #initialize centroids
    centroids = np.random.randn(k, L.shape[1]) * 0.1
    for iteration in range(1, iters+1):
        c2 = np.sum(centroids**2, 1, keepdims=True)
        summation = np.zeros((k, L.shape[1]))
        counts = np.zeros((k, 1))
        loss = 0
        for i in range(0, L.shape[0], batch_size):
            last_index = min(i + batch_size, L.shape[0])
            m = last_index - i

            # shape (k, batch_size) - shape (k, 1)
            tmp = np.dot(centroids, L[i:last_index, :].T) - c2
            # shape (batch_size, )
            indices = np.argmax(tmp, 0)
            # shape (1, batch_size)
            val = np.max(tmp, 0, keepdims=True)

            loss += np.sum((0.5 * L2[i:last_index]) - val.T)

            # Don't use a sparse matrix here
            S = np.zeros((batch_size, k))
            S[range(batch_size), indices] = 1

            # shape (k, n_pixels)
            this_sum = np.dot(S.T, L[i:last_index, :])
            summation += this_sum

            this_counts = np.sum(S, 0, keepdims=True).T
            counts += this_counts 
            
        centroids = summation / counts
        
        bad_indices = np.where(counts == 0)[0]
        centroids[bad_indices, :] = 0
    return centroids



#Testing the class by creating a 'galaxyPic' object, cropping it, scaling it, and extracting patches from each
galaxyPics = GetImage()
galaxyPics.getLabels()
galaxyPics.crop()
galaxyPics.scale()
galaxyPics.patch()
galaxyPics.makevector()
galaxyPics.normalize()
galaxyPics.whiten()

dictionnary = Kmeans(galaxyPics.patchvector)
features = []
for i in galaxyPics.patchvector:
    features.append(np.matmul(dictionnary, i))
features = np.array(features)

truedataset = []
for i, x in enumerate(galaxyPics.label):
    truedataset.append((features[i], x))
truedataset = np.array(truedataset, dtype=object)
