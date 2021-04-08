# PHYS321_FinalProj

Image clustering using K-means for feature generation and a predictor (TBD) for galaxy classification. 

Approach based on:
https://www-cs.stanford.edu/~acoates/papers/coatesng_nntot2012.pdf

Challenge: Is the object a smooth galaxy, a galaxy with features/disk or a star?

==> 3 clusters (smooth, features/disk, star)

Current Important Parameters:
- Crop size 
- scale size
- image patch dimensions
- Number of patches to extract

Step 1: Get images in python 

Step 2: Process raw images by cropping (and scaling?) them

**#TODO**

Step 3: Extract image patches from the processed images (Outlined in section 2 of the Coates paper)

  Alternate routes we could take depending on how we do step 2:
  - getImage_noScaling -> we sample 16x16 image patches from the 160x160 images (Method used in Coates paper)
  - getImage_withScaling -> We sample 4x4 image patches from the 16x16 images (Less computationally intesive)
  
  
