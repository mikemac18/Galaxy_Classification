# PHYS321_FinalProj

FINAL SUBMISSION - GalaxyClassifier.ipynb

Image clustering using K-means for feature generation and various supervised leaning algorithms for galaxy classification. 

Approach based on:
https://www-cs.stanford.edu/~acoates/papers/coatesng_nntot2012.pdf

Raw data from Kaggle: 
https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge

Files for our progression can be found in the folder 'Progress files' and our final submission code is in file GalaxyClassifier.ipynb. 
The contribution by each student can be found in the contribution.txt file. 

Challenge: Is the object a smooth galaxy, a galaxy with features/disk or a star?

==> 3 clusters (smooth, features/disk, star)

What was Done : 
- Image pre-processing : We take n raw images from our training folder, we crop each image from 424-by-424 pixels to 160-by-160 pixels and we scale them from 160-by-160 to 16-by-16 pixels. We then extracted four 8-by-8 patches from each image and flattended each patch into a 1-by-64 vector. Each patch vector was normalized and whitened. 
- Unsupervised learning : We then performed a Mini batch K-means algorithm on our 4n patch vectors to get K centroids. We use the K centroids to build a k-by-64 bank of filters D which we apply to every patch vector to get a 1-by-k feature vector. 
- Supervised learning : We transform our 4n patch feature vectors into n image feature vectors containing 4 patch arrays each. We then gather the labels from each image from the table of probabilities provided for the Kaggle challenge. From these features and labels we train and test a variety of algorithms.
