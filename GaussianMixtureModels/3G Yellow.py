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
mb2=np.array([88.071])
sb2=np.array([81.74])
mg2=np.array([234.7244762171])
sg2=np.array([7.32973545834207])
mr2=np.array([234.7230400106925])
sr2=np.array([7.3286748652304865])






# In[4]:


ans_b2=gaussian(x, mb2, sb2)
ans_g2=gaussian(x, mg2, sg2)
ans_r2=gaussian(x, mr2, sr2)


# In[5]:


plt.plot(ans_b2, 'b')
print(max(ans_b2))
#plt.show()

plt.plot(ans_g2, 'g')
print(max(ans_g2))
#plt.show()


plt.plot(ans_r2, 'r')
print(max(ans_r2))
plt.show()


# In[ ]:





# In[ ]:





# In[9]:

centersandradii=[]
first_detec=0
n=0
c=cv2.VideoCapture("detectbuoy.avi")

while (True):
    ret,image=c.read()
    if image is None:
        break
    image_g=image[:,:,1]
    image_r=image[:,:,2]
    image_b=image[:,:,0]
    cv2.imshow("green",image_g)
    cv2.imshow("red",image_r)
    cv2.imshow("blue",image_b)
    check=0
    if ret == True:
        img_out2=np.zeros(image_r.shape, dtype = np.uint8)
        check+check+1
        for i in range(0,image_r.shape[0]):
            for j in range(0,image_r.shape[1]):
                y=image_r[i][j]
                z= image_g[i][j]
                #if check < 50:
                    #value=130
                #else:
                    #value = 200
                #if ((ans_r2[y] +ans_r2[z])/2) > 0.05  and ((ans_b2[y] +ans_b2[z])/2) < 0.015 and image_b[i][j]<value:
                    #img_out2[i][j]=255
                if (ans_r2[y]) > 0.030  and (ans_g2[z]) >0.030 and   image_g[i][j]>170   and   image_b[i][j]<140:#:and image_b[i][j]<value:
                    img_out2[i][j]=255
                else:
                    img_out2[i][j]=0
                    
        ret, threshold2 = cv2.threshold(img_out2, 240, 255, cv2.THRESH_BINARY)
        kernel2 = np.ones((3,3),np.uint8)
    
        dilation2 = cv2.dilate(threshold2,kernel2,iterations = 6)
        #dilation=cv2.GaussianBlur(dilation,(5,5),0)
        contours2= cv2.findContours(dilation2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        biggest_contour=[]
        biggest_contour_area=0
        normal_detected_flag=0
        for contour in contours2[0]:
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
                    
        cv2.imshow("Threshold",dilation2)
        cv2.imshow('YoYo1', image)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break      # wait for ESC key to exit

    else:
        break
        
c.release()
cv2.destroyAllWindows()   

# In[ ]:





# In[ ]:




