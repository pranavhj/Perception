# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 18:55:46 2020

@author: prana
"""

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
    


def getBoundingbox(img):
    img1=img.copy()
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
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
images = [cv2.imread(file) for file in glob.glob(r"Yellow_new/*.jpg")]
pixels_red=[]
pixels_green=[]
for image in images:
    image = cv2.GaussianBlur(image,(5,5),0)
    
    image=getBoundingbox(image)
    #cv2.imshow("image",image)
    #cv2.waitKey(0)
    
    
    img=image[:,:,2]              #getting red comp
    #print(np.shape(img))
    for i in range(len(img)):
        for j in range(len(img[i])):
            pixels_green.append(img[i][j])
            
    img=image[:,:,1]              #getting green comp
    #print(np.shape(img))
    for i in range(len(img)):
        for j in range(len(img[i])):
            pixels_red.append(img[i][j])
            
    

print(len(pixels_red))
mean1 = 50
mean2 = 250
mean3 = 250

std1 = 20
std2 = 20
std3 = 20
n=0
means=[]
stds=[]    
while(n!=30):
    b11 = []
    b21 = []
    b31 = []
    b12=[]
    b22=[]
    b32=[]
    for im1 in pixels_red:
        prob_red_1 = Probabilty(im1, mean1, std1)
        prob_red_2 = Probabilty(im1, mean2, std2)
        prob_red_3 = Probabilty(im1, mean3, std3)
        b11.append((prob_red_1*(1/3))/(prob_red_1*(1/3) + prob_red_2*(1/3) + prob_red_3*(1/3)))
        b21.append((prob_red_2 * (1 / 3)) / (prob_red_1 * (1 / 3) + prob_red_2 * (1 / 3) + prob_red_3 * (1 / 3)))
        b31.append((prob_red_3 * (1 / 3)) / (prob_red_1 * (1 / 3) + prob_red_2 * (1 / 3) + prob_red_3 * (1 / 3)))
    for im2 in pixels_green:
        prob_green_1 = Probabilty(im2, mean1, std1)
        prob_green_2 = Probabilty(im2, mean2, std2)
        prob_green_3 = Probabilty(im2, mean3, std3)
        b12.append((prob_green_1*(1/3))/(prob_green_1*(1/3) + prob_green_2*(1/3) + prob_green_3*(1/3)))
        b22.append((prob_green_2 * (1 / 3)) / (prob_green_1 * (1 / 3) + prob_green_2 * (1 / 3) + prob_green_3 * (1 / 3)))
        b32.append((prob_green_3 * (1 / 3)) / (prob_green_1 * (1 / 3) + prob_green_2 * (1 / 3) + prob_green_3 * (1 / 3)))
    
    mean_red_1 = (np.matmul(np.array(b11) , np.transpose(np.array(pixels_red)))) / np.sum(np.array(b11))
    mean_red_2 = (np.matmul(np.array(b21) , np.transpose(np.array(pixels_red)))) / np.sum(np.array(b21))
    mean_red_3 = (np.matmul(np.array(b21) , np.transpose(np.array(pixels_red)))) / np.sum(np.array(b31))
    
    
    mean_green_1 = (np.matmul(np.array(b12) , np.transpose(np.array(pixels_green)))) / np.sum(np.array(b11))
    mean_green_2 = (np.matmul(np.array(b22) , np.transpose(np.array(pixels_green)))) / np.sum(np.array(b21))
    mean_green_3 = (np.matmul(np.array(b32) , np.transpose(np.array(pixels_green)))) / np.sum(np.array(b31))
    
    
    std_red_1 = ((np.matmul(np.array(b11) , np.transpose(np.array(pixels_red)-mean1)**2))/(np.sum(np.array(b11))))**(1/2)
    std_red_2 = ((np.matmul(np.array(b21) , np.transpose(np.array(pixels_red)-mean2)**2))/(np.sum(np.array(b21))))**(1/2)
    std_red_3 = ((np.matmul(np.array(b31) , np.transpose(np.array(pixels_red)-mean3)**2))/(np.sum(np.array(b31))))**(1/2)
    
    
    std_green_1 = ((np.matmul(np.array(b12) , np.transpose(np.array(pixels_green)-mean1)**2))/(np.sum(np.array(b12))))**(1/2)
    std_green_2 = ((np.matmul(np.array(b22) , np.transpose(np.array(pixels_green)-mean2)**2))/(np.sum(np.array(b22))))**(1/2)
    std_green_3 = ((np.matmul(np.array(b32) , np.transpose(np.array(pixels_green)-mean3)**2))/(np.sum(np.array(b32))))**(1/2)
    
    
    
 
    mean1=(mean_red_1+mean_green_1)/2
    mean2 = (mean_red_2 + mean_green_2) / 2
    mean3 = (mean_red_3 + mean_green_3) / 2
    std1 = (std_red_1 + std_green_1) / 2
    std2 = (std_red_2 + std_green_2) / 2
    std3 = (std_red_3 + std_green_3) / 2
    if n>0:
        if abs(mean1-means[len(means)-1][0])<0.2  and   abs(mean2-means[len(means)-1][1])<0.2  and   abs(mean3-means[len(means)-1][2])<0.2   and   abs(std1-stds[len(stds)-1][0])<0.2    and   abs(std2-stds[len(stds)-1][1])<0.2   and   abs(std3-stds[len(stds)-1][2])<0.2:  
            print("Convergence")
            break
    means.append([mean1,mean2,mean3])
    stds.append([std1,std2,std3])
    print("mean",mean1, mean2, mean3)
    print("std",std1, std2, std3)
    n = n + 1



