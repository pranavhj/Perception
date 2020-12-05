#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from numpy import linalg as LA
import math
import scipy.stats as stats


# In[2]:


def gaussian(x, mu, sig):
    return ((1/(sig*math.sqrt(2*math.pi)))*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))))


# In[3]:
def KalmannFilter(centersandradii,n):
    if len(centersandradii)>5:
        prev=centersandradii[len(centersandradii)-1]
        prev_prev=centersandradii[len(centersandradii)-2]
        vel_x=0
        vel_y=0
        for i in range(3):
            vel_x=vel_x+(centersandradii[len(centersandradii)-1-i][0][0]-centersandradii[len(centersandradii)-2-i][0][0])#(prev[0][0]-prev_prev[0][0])
            vel_y=vel_y+(centersandradii[len(centersandradii)-1-i][0][1]-centersandradii[len(centersandradii)-2-i][0][1])#(prev[0][1]-prev_prev[0][1])
        next_x=prev[0][0]+(vel_x/3)
        next_y=prev[0][1]+(vel_y/3)
        return [(int(next_x),int(next_y)),int((prev[1]+prev_prev[1])/2)]
    else:
        return centersandradii[len(centersandradii)-1]

x=list(range(0, 256))
mb1=np.array([170.1123])
sb1=np.array([36.1331436])
mg1=np.array([239.95395])
sg1=np.array([7.3541856])
mr1=np.array([252.3011604])
sr1=np.array([2.373163])

x=list(range(0, 256))
mb1=np.array([90.35])
sb1=np.array([82.49])
mg1=np.array([244.16])           #B-170   G-240    R-252
sg1=np.array([9.22])
mr1=np.array([244.16])
sr1=np.array([9.22])
# In[4]:


ans_b1=gaussian(x, mb1, sb1)
ans_g1=gaussian(x, mg1, sg1)
ans_r1=gaussian(x, mr1, sr1)


# In[5]:


plt.plot(ans_b1, 'b')
print(max(ans_b1))
#plt.show()

plt.plot(ans_g1, 'g')
print(max(ans_g1))
#plt.show()


plt.plot(ans_r1, 'r')
print(max(ans_r1))
plt.show()


# In[6]:

centersandradii=[]
c=cv2.VideoCapture("detectbuoy.avi")
first_detec=0
n=0
while (True):
    n=n+1
    ret,image=c.read()
    if image is None:
        break
    image_g=image[:,:,1]
    image_r=image[:,:,2]
    image_b=image[:,:,0]
    cv2.imshow("green",image_g)
    cv2.imshow("red",image_r)
    cv2.imshow("blue",image_b)
    if ret == True:
        img_out1=np.zeros(image_r.shape, dtype = np.uint8)
         
        for i in range(0,image_r.shape[0]):
            for j in range(0,image_r.shape[1]):
                y=image_r[i][j]
                
                if ans_r1[y]>0.002 and image_g[i][j]<150:
                    #print(ans_r[y], 'r')
                    img_out1[i][j]=255
                    
               
                    
                    
                    
        ret, threshold = cv2.threshold(img_out1, 240, 255, cv2.THRESH_BINARY)
        kernel1 = np.ones((3,2),np.uint8)

    
        dilation1 = cv2.dilate(threshold,kernel1,iterations = 6)
        contours1= cv2.findContours(dilation1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        biggest_contour=[]
        biggest_contour_area=0
        normal_detected_flag=0
        for contour in contours1[0]:
            area=cv2.contourArea(contour)
            (x,y),radius = cv2.minEnclosingCircle(contour)
            if area>biggest_contour_area    and    y>100:
                biggest_contour=contour
                biggest_contour_area=area

        if biggest_contour_area>30:
            (x,y),radius = cv2.minEnclosingCircle(biggest_contour)
            center = (int(x),int(y)-1)
            radius = int(radius) - 1
            if radius > 12:
                if first_detec==0:
                        first_detec=[center,radius]
                centersandradii.append([center,radius])
                cv2.circle(image,center,radius,(0,255,255),2)
                normal_detected_flag=1
                
        if normal_detected_flag!=1   and  first_detec!=0:
            center,radius=KalmannFilter(centersandradii,n)
            if center[1]<480   and   center[0]<640:
                cv2.circle(image,center,radius,(0,255,255),2)
                centersandradii.append([center,radius])
                print("Kalmann detec",center,radius)
                    
        cv2.imshow("Threshold",dilation1)
        cv2.imshow('YoYo1', image)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break      # wait for ESC key to exit

    else:
        break
        
c.release()
cv2.destroyAllWindows()   


# In[ ]:




