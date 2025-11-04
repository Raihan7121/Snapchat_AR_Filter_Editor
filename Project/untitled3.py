# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 00:28:21 2025

@author: Acer
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from tabulate import tabulate

def img_area(bin_img):
    return np.count_nonzero(bin_img)
def img_perimeter(border_img):
    return np.count_nonzero(border_img)

def calc_descriptors(binary_img):
    kernel=np.ones((3,3),np.uint8)
    eroded = cv2.erode(binary_img,kernel,iterations=1)
    border_img=binary_img-eroded
    
    area = img_area(binary_img)
    perimeter= img_perimeter(border_img)
    
    coords = np.column_stack(np.where(binary_img > 0))
    ymin, xmin = coords.min(axis=0)
    ymax, xmax = coords.max(axis=0)
    max_diameter = max(xmax - xmin, ymax - ymin)
    
    contours,_=cv2.findContours(binary_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    a=b=1
    if len(contours)>0 and len(contours[0])>=5:
        cnt=max(contours,key=cv2.contourArea)
        ellipse=cv2.fitEllipse(cnt)
        (_,_),(ma,MA),_=ellipse
        a=max(ma,MA)
        b=min(ma,MA)
    form_factor =(4 *math.pi*area)/(perimeter**2) 
    compactness = (perimeter**2)/area 
    eccentricity = math.sqrt(1 - (b/a)**2) 
    roundness = (4 * area) / (np.pi * (max_diameter ** 2) )
    return form_factor,roundness,compactness,eccentricity

def kl_divergence(p,q):
    p=np.array(p)
    q= np.array(q)
    p=p/np.sum(p)
    q= q/np.sum(q)
    return np.sum(p*np.log(p/q))

def euclidean_distance(p,q):
    p=np.array(p)
    q=np.array(q)
    return math.sqrt(sum((p-q)**2))
def cosine_similarity(p,q):
    p=np.array(q)
    q=np.array(q)
    return sum(p*q)/(math.sqrt(sum(p**2))*math.sqrt(sum(q**2)))

def sim_matrix(train_images,test_images):
    train_descriptors = []
    test_descriptors = []
    
    for i,img in enumerate(train_images):
        ff,rd,cs,ec=calc_descriptors(img)
        train_descriptors.append([ff,rd,cs,ec])
        
    for i,img in enumerate(test_images):
        ff,rd,cs,ec=calc_descriptors(img)
        test_descriptors.append([ff,rd,cs,ec])
    sim_mat = []
    for i,test_d in enumerate(test_descriptors):
        sim_row = []
        for j,train_d in enumerate(train_descriptors):
            d=kl_divergence(test_d, train_d)
            sim_row.append(d)
        sim_mat.append(sim_row)
    return sim_mat

train_images = [
    cv2.imread(r"C:\Users\Acer\Desktop\imagelab\lab5b\train\c1.jpg",0),
    cv2.imread(r"C:\Users\Acer\Desktop\imagelab\lab5b\train\p1.png",0),
    cv2.imread(r"C:\Users\Acer\Desktop\imagelab\lab5b\train\t1.jpg",0)
    ]

test_images = [
    cv2.imread(r"C:\Users\Acer\Desktop\imagelab\lab5b\test\c2.jpg",0),
    cv2.imread(r"C:\Users\Acer\Desktop\imagelab\lab5b\test\p2.png",0),
    cv2.imread(r"C:\Users\Acer\Desktop\imagelab\lab5b\test\p3.jpg",0),
    cv2.imread(r"C:\Users\Acer\Desktop\imagelab\lab5b\test\t2.jpg",0)
    ]
    
train_images = [cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1] for img in train_images ]
test_images = [cv2.threshold(img,127,255,cv2.THRESH_BINARY)[1] for img in test_images]

result = sim_matrix(train_images, test_images)

row_headers = [f'Test {i+1}' for i in range(4)]
col_headers = [f'GT {i+1}' for i in range(3)]

distances_matrx = np.array(result)
print(tabulate(distances_matrx[0:4,0:3],headers=col_headers,showindex=row_headers,tablefmt='grid'))