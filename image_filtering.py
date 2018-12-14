#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Filtering

Reference: http://machinelearninguru.com/computer_vision/basics/convolution/image_convolution_1.html

Note: 
    The outputs are slightly different from the original outputs in the post,
    due to the `image.png` file is not available (using `image.jpg` instead)

Zhenhao Ge, 2018-12-12

"""

from __future__ import print_function
from skimage import io, viewer, color
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure
#import pylab

def convolve2d(image, kernel):
    # This function which takes an image and a kernel 
    # and returns the convolution of them
    # Args:
    #   image: a numpy array of size [image_height, image_width].
    #   kernel: a numpy array of size [kernel_height, kernel_width].
    # Returns:
    #   a numpy array of size [image_height, image_width] (convolution output).
    
    kernel = np.flipud(np.fliplr(kernel))    # Flip the kernel
    output = np.zeros_like(image)            # convolution output
    # Add zero padding to the input image
    image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))   
    image_padded[1:-1, 1:-1] = image
    for x in range(image.shape[1]):     # Loop over every pixel of the image
        for y in range(image.shape[0]):
            # element-wise multiplication of the kernel and the image
            output[y,x]=(kernel*image_padded[y:y+3,x:x+3]).sum()        
    return output

### Load and plot image ###

# set the image file name
img_file = 'image.jpg'

# load the image as grayscale in one step
img = io.imread(img_file, as_gray=True)  
# alternatively, you can load the original image and then convert it to grayscale
# img2 = io.imread(img_file)
# img2 = color.rgb2gray(img2)

print('image matrix size: {}'.format(img.shape))  # print the size of image
print('First 5 columns and rows of the image matrix:\n {}'.format(img[:5,:5]*255)) 
viewer.ImageViewer(img).show()  # plot the image

### Convolve the sharpen kernel with an image ###

# Adjust the contrast of the image by applying Histogram Equalization
# clip_limit: normalized between 0 and 1 (higher values give more contrast) 
image_equalized = exposure.equalize_adapthist(img/np.max(np.abs(img)), clip_limit=0.03)
plt.imshow(image_equalized, cmap=plt.cm.gray)
plt.axis('off')
plt.show()

# Convolve the sharpen kernel and the image
kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
image_sharpen = convolve2d(img, kernel)
print('First 5 columns and rows of the image_sharpen matrix:\n {}'.format(image_sharpen[:5,:5]*255))

# Plot the filtered image
plt.imshow(image_sharpen, cmap=plt.cm.gray)
plt.axis('off')
plt.show()

# Adjust the contrast of the filtered image by applying Histogram Equalization 
image_sharpen_equalized = exposure.equalize_adapthist(image_sharpen/np.max(np.abs(image_sharpen)), clip_limit=0.03)
plt.imshow(image_sharpen_equalized, cmap=plt.cm.gray)
plt.axis('off')
plt.show()

### Convolve the sharpen kernal with an image using Python packages (Scipy) ###

import scipy

# you can use 'valid' instead of 'same', then it will not add zero padding
image_sharpen = scipy.signal.convolve2d(img, kernel, 'same')
#image_sharpen = scipy.signal.convolve2d(img, kernel, 'valid')
print('First 5 columns and rows of the image_sharpen matrix:\n {}'.format(image_sharpen[:5,:5]*255))

### Convolve the sharpen kernal with an image using Python packages (OpenCV) ###

import cv2

image_sharpen = cv2.filter2D(img, -1, kernel)
print('First 5 columns and rows of the image_sharpen matrix:\n {}'.format(image_sharpen[:5,:5]*255))

# Adjust the contrast of the filtered image by applying Histogram Equalization 
image_sharpen_equalized = exposure.equalize_adapthist(image_sharpen/np.max(np.abs(image_sharpen)), clip_limit=0.03)
plt.imshow(image_sharpen_equalized, cmap=plt.cm.gray)
plt.axis('off')
plt.show()

### Convolve an edge detection kernel with an image ###

kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
# we use 'valid' which means we do not add zero padding to our image
edges = scipy.signal.convolve2d(img, kernel, 'valid')
print('First 5 columns and rows of the image_sharpen matrix:\n {}'.format(image_sharpen[:5,:5]*255))

# Adjust the contrast of the filtered image by applying Histogram Equalization
edges_equalized = exposure.equalize_adapthist(edges/np.max(np.abs(edges)), clip_limit=0.03)
plt.imshow(edges_equalized, cmap=plt.cm.gray)    # plot the edges_clipped
plt.axis('off')
plt.show()

### Apply sharpen and edge detection filters back to back ###

# apply sharpen filter to the original image
sharpen_kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
image_sharpen = scipy.signal.convolve2d(img, sharpen_kernel, 'valid')

# apply edge detection filter to the sharpened image
edge_kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
edges = scipy.signal.convolve2d(image_sharpen, edge_kernel, 'valid')

# Adjust the contrast of the filtered image by applying Histogram Equalization
edges_equalized = exposure.equalize_adapthist(edges/np.max(np.abs(edges)), clip_limit=0.03)
plt.imshow(edges_equalized, cmap=plt.cm.gray)    # plot the edges_clipped
plt.axis('off')
plt.show()

### Apply blur filter to denoise an image ###

# apply blur filter to the edge detection filtered image
blur_kernel = np.array([[1,1,1],[1,1,1],[1,1,1]])/9.0;
denoised = scipy.signal.convolve2d(edges, blur_kernel, 'valid')

# Adjust the contrast of the filtered image by applying Histogram Equalization
denoised_equalized = exposure.equalize_adapthist(denoised/np.max(np.abs(denoised)), clip_limit=0.03)
plt.imshow(denoised_equalized, cmap=plt.cm.gray)    # plot the denoised_clipped
plt.axis('off')
plt.show()