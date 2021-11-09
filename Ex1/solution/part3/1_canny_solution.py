import numpy as np
from scipy.ndimage.filters import gaussian_filter,convolve
import matplotlib.pyplot as plt
from scipy import where
import cv2

### Read Image ###
im=cv2.imread('../images/zurlim.png', 0).astype('float')


####### Gaussian Smooth Image #######
blurred_im = gaussian_filter(im, sigma=2,order=0,mode='reflect')


###### Gradients x and y (Sobel filters) ######
im_x = convolve(blurred_im,[[-1,0,1],[-2,0,2],[-1,0,1]]) 
im_y = convolve(blurred_im,[[1,2,1],[0,0,0],[-1,-2,-1]])


###### gradient and direction ########
gradient = np.power(np.power(im_x, 2.0) + np.power(im_y, 2.0), 0.5)
theta = np.arctan2(im_y, im_x)

####### Thresholding Criteria #######
thresh=50;
thresholdEdges = (gradient > thresh)


###### Non-maximum suppression ########

###### Convert to degree ######
theta = 180 + (180/np.pi)*theta #
###### Quantize angles ######
x_0,y_0 = where(((theta<22.5)+(theta>157.5)*(theta<202.5)  +(theta>337.5)) == True)
x_45,y_45 = where(((theta>22.5)*(theta<67.5) +(theta>202.5)*(theta<247.5)) == True)
x_90,y_90 = where(((theta>67.5)*(theta<112.5) +(theta>247.5)*(theta<292.5)) == True)
x_135,y_135 = where(((theta>112.5)*(theta<157.5) +(theta>292.5)*(theta<337.5)) == True)

theta[x_0,y_0] = 0        # E-W
theta[x_45,y_45] = 1      # NE
theta[x_90,y_90] = 2      # N-S
theta[x_135,y_135] = 3    # NW

grad_supp = np.zeros((gradient.shape[0],gradient.shape[1]))
for r in range(im.shape[0]):
    for c in range(im.shape[1]):
        
        #Suppress pixels at the image edge
        if r == 0 or r == im.shape[0]-1 or c == 0 or c == im.shape[1] - 1:
            grad_supp[r, c] = 0
            continue
        
        ###### Thresholding #######
        if gradient[r, c]<thresh:
            grad_supp[r, c] = 0
            continue
        
        ######### NMS ##########
        tq = theta[r, c]
        if tq == 0: # E-W
            if gradient[r, c] >= gradient[r, c-1] and gradient[r, c] >= gradient[r, c+1]:
                grad_supp[r, c] = 1
        if tq == 1: # NE
            if gradient[r, c] >= gradient[r-1, c+1] and gradient[r, c] >= gradient[r+1, c-1]:
                grad_supp[r, c] = 1
        if tq == 2: # N-S (vertical)
            if gradient[r, c] >= gradient[r-1, c] and gradient[r, c] >= gradient[r+1, c]:
                grad_supp[r, c] = 1
        if tq == 3: # NW
            if gradient[r, c] >= gradient[r-1, c-1] and gradient[r, c] >= gradient[r+1, c+1]:
                grad_supp[r, c] = 1
###### Binary Thresholding #######
edges = (grad_supp > 0)


# Plotting of results
# No need to change it
plt.close("all")
plt.ion()
f, ax_arr = plt.subplots(1, 2, figsize=(18, 16))
ax_arr[0].set_title("Input Image")
ax_arr[1].set_title("Canny Edge Detector")
ax_arr[0].imshow(im, cmap='gray')
ax_arr[1].imshow(edges, cmap='gray')
plt.show()
plt.pause(5)

