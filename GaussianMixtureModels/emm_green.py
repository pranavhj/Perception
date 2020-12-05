# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 00:59:07 2020

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
from scipy.stats import multivariate_normal

def Probabilty(x_co, mean, std):
    return (1 / (std * math.sqrt(2 * math.pi))) * (math.exp(-((x_co - mean) ** 2) / (2 * (std) ** 2)))
    #return 1/np.sqrt(2*3.142)*np.exp(-1/2*np.power(((x_co-mean)/std),2))
    #return multivariate_normal.pdf(x_co, mean, std)
    


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

def gaussian(x, mu, sig):
    return ((1/(sig*math.sqrt(2*math.pi)))*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))))



images=[]
images = [cv2.imread(file) for file in glob.glob(r"Green_new/*.jpg")]
pixels=[]
for image in images:
    image = cv2.GaussianBlur(image,(5,5),0)
    image=getBoundingbox(image)
    img=image[:,:,1]              #getting green comp
    #print(np.shape(img))
    for i in range(len(img)):
        for j in range(len(img[i])):
            pixels.append(img[i][j])

print(len(pixels))
mean1 = 100
mean2 = 230
mean3 = 100

std1 = 20
std2 = 20
std3 = 20


#mean1 = 104.14
#mean2 = 155.33
#mean3 = 94.62
#
#std1 = 75.86
#std2 = 103.79
#std3 = 64.71
n=0
means=[]
stds=[]    
while n<50:
    prob1 = []
    prob2 = []
    prob3 = []
    prob4 = []
    b1 = []
    b2 = []
    b3 = []
    b4 = []
    for im in pixels:
        p1 = Probabilty(im, mean1, std1)
        prob1.append(p1)
        p2 = Probabilty(im, mean2, std2)
        prob2.append(p2)
        p3 = Probabilty(im, mean3, std3)
        prob3.append(p3)
        #print(p1,p2,p3)
        #print(p1)
        b1.append((p1 * (1 / 3)) / (p1 * (1 / 3) + p2 * (1 / 3) + p3 * (1 / 3) ))
        b2.append((p2 * (1 / 3)) / (p1 * (1 / 3) + p2 * (1 / 3) + p3 * (1 / 3) ))
        b3.append((p3 * (1 / 3)) / (p1 * (1 / 3) + p2 * (1 / 3) + p3 * (1 / 3) ))
        
    mean1 = (np.matmul(np.array(b1) , np.transpose(np.array(pixels)))) / np.sum(np.array(b1))
    mean2 = (np.matmul(np.array(b2) , np.transpose(np.array(pixels)))) / np.sum(np.array(b2))
    mean3 = (np.matmul(np.array(b3) , np.transpose(np.array(pixels)))) / np.sum(np.array(b3))
    
    std1 = ((np.matmul(np.array(b1) , np.transpose(np.array(pixels)-mean1)**2))/(np.sum(np.array(b1))))**(1/2)
    std2 = ((np.matmul(np.array(b2) , np.transpose(np.array(pixels)-mean2)**2))/(np.sum(np.array(b2))))**(1/2)
    std3 = ((np.matmul(np.array(b3) , np.transpose(np.array(pixels)-mean3)**2))/(np.sum(np.array(b3))))**(1/2)
    #std1 = (np.sum(np.array(b1) * ((np.array(pixels)) - mean1) ** (2)) / np.sum(np.array(b1))) ** (1 / 2)
    #std2 = (np.sum(np.array(b2) * ((np.array(pixels)) - mean2) ** (2)) / np.sum(np.array(b2))) ** (1 / 2)
    #std3 = (np.sum(np.array(b3) * ((np.array(pixels)) - mean3) ** (2)) / np.sum(np.array(b3))) ** (1 / 2)
    if n>0:
        if abs(mean1-means[len(means)-1][0])<0.2  and   abs(mean2-means[len(means)-1][1])<0.2  and   abs(mean3-means[len(means)-1][2])<0.2   and   abs(std1-stds[len(stds)-1][0])<0.2    and   abs(std2-stds[len(stds)-1][1])<0.2   and   abs(std3-stds[len(stds)-1][2])<0.2:  
            print("Convergence")
            break
    means.append([mean1,mean2,mean3])
    stds.append([std1,std2,std3])
    n = n + 1
    print("mean",mean1, mean2, mean3)
    print("std",std1, std2, std3)
print('final mean- ',mean1,mean2,mean3)
print('final strd- ',std1, std2, std3)



