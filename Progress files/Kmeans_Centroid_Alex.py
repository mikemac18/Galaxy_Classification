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
import matplotlib.pyplot as plt

class GetImage(object):
    #Load raw image files
    
    #initialize the path (TODO - make basepath an input parameter, don't hardcode too much)

    #Data should be stored in Users/YOUR_USER_NAME/321_galaxies/images_training_rev1
    def __init__(self, num_pics=10, username = "Admin"):

        self.basepath = 'C:\\Users\\'+username+'\\321_galaxies\\'
        self.imPath_test = 'images_test_rev1\\'
        self.imPath_training =  'images_training_rev1\\'
       
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
                img = galaxyPic[patchminx:patchmaxx, patchminy:patchmaxy]
                self.patches.append(img.flatten())
        self.patches=np.vstack(self.patches)
        return self

        ## Function that normalizes the pixels. Takes vstack of patches as input
    def normalize(self):
        temp1 = self.patches - self.patches.mean(1, keepdims=True)
        temp2 = np.sqrt(self.patches.var(1, keepdims=True) + 10)
        self.patches = temp1/temp2
        return self
    ## Function that runs ZCA whitening
    def whiten(self):
        cov = np.cov(self.patches, rowvar=0)
        self.mean = self.patches.mean(0, keepdims=True)
        d, v = np.linalg.eig(cov)
        self.p = np.dot(v, np.dot(np.diag(np.sqrt(1 / (d + 0.1))), v.T))
        self.patches = np.dot(self.patches - self.mean, self.p)
        return self

def spherical_kmeans(X, k, n_iter, batch_size=50):
    """
    Do a spherical k-means.  Line by line port of Coates' matlab code.
    Returns a (k, n_pixels) centroids matrix
    """

    # shape (n_samples, 1)
    x2 = np.sum(X**2, 1, keepdims=True)

    # randomly initialize centroids
    centroids = np.random.randn(k, X.shape[1]) * 0.1

    for iteration in list(range(1, n_iter + 1)):
        # shape (k, 1)
        c2 = 0.5 * np.sum(centroids ** 2, 1, keepdims=True)

        # shape (k, n_pixels)
        summation = np.zeros((k, X.shape[1]))
        counts = np.zeros((k, 1))
        loss = 0

        for i in list(range(0, X.shape[0], batch_size)):
            last_index = min(i + batch_size, X.shape[0])
            m = last_index - i

            # shape (k, batch_size) - shape (k, 1)
            tmp = np.dot(centroids, X[i:last_index, :].T) - c2
            # shape (batch_size, )
            indices = np.argmax(tmp, 0)
            # shape (1, batch_size)
            val = np.max(tmp, 0, keepdims=True)

            loss += np.sum((0.5 * x2[i:last_index]) - val.T)

            # Don't use a sparse matrix here
            S = np.zeros((batch_size, k))
            S[range(batch_size), indices] = 1

            # shape (k, n_pixels)
            this_sum = np.dot(S.T, X[i:last_index, :])
            summation += this_sum

            this_counts = np.sum(S, 0, keepdims=True).T
            counts += this_counts

        # Sometimes raises RuntimeWarnings because some counts can be 0
        centroids = summation / counts

        bad_indices = np.where(counts == 0)[0]
        centroids[bad_indices, :] = 0

        assert not np.any(np.isnan(centroids))

    return centroids

#Testing the class by creating a 'galaxyPic' object, cropping it, scaling it, and extracting patches from each
galaxyPics = GetImage()
galaxyPics.crop()
galaxyPics.scale()
galaxyPics.patch()

print(galaxyPics.patches.shape)

galaxyPics.normalize()
galaxyPics.whiten()

x = spherical_kmeans(galaxyPics.patches, 40, 20)
print(x.shape)

plt.plot(x)
    #cv2.waitKey(0)
