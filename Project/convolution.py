# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 02:15:00 2025

@author: Acer
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
def gaussian_value(x,y,sigma):
    return (1/(2*math.pi*(sigma**2)))*math.exp(-(x**2+y**2)/(2*sigma**2))

def gaussian_smoothing_kernel(sigma=1):
    size=(5*sigma)|1
    kernel=np.zeros((size,size),dtype=np.float32)
    k=size//2
    for i in range(size):
        for j in range(size):
            x=i-k
            y=j-k
            kernel[i,j]=gaussian_value(x, y, sigma)
    return kernel

def gaussian_value2(x,y,sigma):
    return (((x**2+y**2-2*sigma**2)/(sigma**4))*math.exp(-((x**2+y**2)/(2*sigma**2))))
    

def gaussian_sharpening_kernel(size,sigma):
    kernel=np.zeros((size,size))
    k=size//2
    for i in range(-k,k+1):
        for j in range(-k,k+1):
            kernel[i+k][j+k]=gaussian_value2(i, j, sigma)
    return kernel    

def convolve2d(img,kernel,center=None):
    if center is None:
        center=(kernel.shape[0]//2,kernel.shape[1]//2)

    h,w=img.shape
    kh,kw=kernel.shape
    top=kh-center[0]-1
    bottom=kh-top-1
    left=kw-center[1]-1
    right=kw-left-1
    
    padded_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT)
    conv_img=np.zeros_like(padded_img)
    
    for i in range(top,h+top):
        for j in range(left,left+w):
            sum=0.0
            for fi in range(-top,bottom+1):
                for fj in range(-left,right+1):
                    sum+=(padded_img[i+fi,j+fj]*kernel[bottom-fi,right-fj])
            conv_img[i,j]=sum
    norm=np.round(cv2.normalize(conv_img,None,0,255,cv2.NORM_MINMAX)).astype(np.uint8)
    output_img=norm[top:top+h,left:left+w]
    
    return output_img

    
lena = cv2.imread(r"C:/Users/Acer/Desktop/imagelab/lab1/Lena.jpg", cv2.IMREAD_GRAYSCALE)

kernel=gaussian_smoothing_kernel()
x=convolve2d(lena, kernel)
y=cv2.filter2D(lena, ddepth=cv2.CV_32F, kernel=kernel)
y=cv2.normalize(y,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
z=x-y
mx=sum(np.abs(z))
#print(mx)
cv2.imshow("img",lena)
cv2.imshow("com",x)
cv2.imshow("comy",y)
cv2.waitKey(0)
cv2.destroyAllWindows()

    