# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 18:47:23 2020

@author: prana
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from numpy import linalg as LA
import math
import scipy.stats as stats
count = 0
import glob
from scipy.stats import norm

    
def getBoundingbox1(img):
    img1=img.copy()
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    height,width,layers = img.shape
    #print(height,width)
    mask = np.zeros((height,width), np.uint8)
    contours=cv2.findContours(thresh,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #  print(contours)
    
    smallest_x=11110
    biggest_x=0
    
    
    smallest_y=111110
    biggest_y=0         
    
    for cnt1 in contours[0]:
        for cnt in cnt1:
            #print(cnt)
            if smallest_x>cnt[0][0]:
                smallest_x=cnt[0][0]
                
            if biggest_x<cnt[0][0]:
                biggest_x=cnt[0][0]
                
            if smallest_y>cnt[0][1]:
                smallest_y=cnt[0][1]
                
            if biggest_y<cnt[0][1]:
                biggest_y=cnt[0][1]
                
           
            
            
    #print(smallest_x,biggest_x,smallest_y,biggest_y)
    crop=img1[smallest_y:biggest_y,smallest_x:biggest_x]       
        
    
    

    return crop
    
            
        
    
    
    
    
def getBoundingbox(img):

    img1=img.copy()
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    height,width,layers = img.shape
    mask = np.zeros((height,width), np.uint8)
    
    edges = cv2.Canny(thresh, 100, 200)

    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 10000, param1 = 3, param2 = 3, minRadius = 0, maxRadius = 1000)
    #print(circles)
    for i in circles[0,:]:
        i[2]=i[2]
        # Draw on mask
        cv2.circle(mask,(i[0],i[1]),i[2],(255,255,255),thickness=-1)
    
    # Copy that image using that mask
    masked_data = cv2.bitwise_and(img1, img1, mask=mask)
    
    # Apply Threshold
    _,thresh = cv2.threshold(mask,1,255,cv2.THRESH_BINARY)
    
    # Find Contour
    #contours = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    x=circles[0][0][0]
    y=circles[0][0][1]
    r=circles[0][0][2]
    #print(x,y,r)
    # Crop masked_data
    crop = masked_data[int(y-r):int(y+r),int(x-r):int(x+r)]

    return crop

def gaussian(x, mu, sig):
    return ((1/(sig*math.sqrt(2*math.pi)))*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))))

images=[]
images = [cv2.imread(file) for file in glob.glob(r"Green_new/*.jpg")]
histogram_b=[]
histogram_g=[]
histogram_r=[]
mean_b=[]
mean_g=[]
mean_r=[]
std_b=[]
std_g=[]
std_r=[]

for image in images:
    #print(np.shape(image))
    #cv2.imshow("image",image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()    
    image=getBoundingbox1(image) 
    #cv2.imshow("cropped_image",image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    for i in range(3):           #for all color channels
        
        if i==0:
            h=cv2.calcHist([image],[i],None,[256],[50,255])
            histogram_b.append(h)
        if i==1:
            h=cv2.calcHist([image],[i],None,[256],[50,255])
            histogram_g.append(h)
        if i==2:
            h=cv2.calcHist([image],[i],None,[256],[50,255])
            histogram_r.append(h)
            
    mean,std=cv2.meanStdDev(image)
    mean_b.append(mean[0])
    mean_g.append(mean[1])
    mean_r.append(mean[2])
    std_b.append(std[0])
    std_g.append(std[1])
    std_r.append(std[2])
x=range(0,256)
#print(np.shape(histogram_b))
histogram_b=np.sum(histogram_b,axis=0)/len(histogram_b)
histogram_g=np.sum(histogram_g,axis=0)/len(histogram_g)
histogram_r=np.sum(histogram_r,axis=0)/len(histogram_r)
plt.plot(histogram_b,color='b')
plt.plot(histogram_g,color='g')
plt.plot(histogram_r,color='r')
plt.show()
mean_b_val=np.mean(mean_b)
mean_g_val=np.mean(mean_g)
mean_r_val=np.mean(mean_r)
std_b_val=np.mean(std_b)
std_g_val=np.mean(std_g)
std_r_val=np.mean(std_r)
print(mean_b_val,mean_g_val,mean_r_val)
print(std_b_val,std_g_val,std_r_val)
ans_b=gaussian(x,mean_b_val,std_b_val)
ans_g=gaussian(x,mean_g_val,std_g_val)
ans_r=gaussian(x,mean_r_val,std_r_val)
plt.plot(ans_b,color='b')
plt.plot(ans_g,color='g')
plt.plot(ans_r,color='r')
plt.show()
