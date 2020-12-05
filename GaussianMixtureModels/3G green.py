#!/usr/bin/env python
# coding: utf-8

# In[10]:


import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from numpy import linalg as LA
import math
import scipy.stats as stats


# In[11]:


def gaussian(x, mu, sig):
    return ((1/(sig*math.sqrt(2*math.pi)))*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))))


# In[12]:
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
mb3=np.array([85])
sb3=np.array([88])
mg3=np.array([232])         #B-230   G-200    R-244
sg3=np.array([18])
mr3=np.array([85])
sr3=np.array([88])


x=list(range(0, 256))
mb3=np.array([65.17])
sb3=np.array([80.36])
mg3=np.array([218.87])         #B-230   G-200    R-244
sg3=np.array([21.68])
mr3=np.array([65.17])
sr3=np.array([80.36])





# In[13]:


ans_b3=gaussian(x, mb3, sb3)
ans_g3=gaussian(x, mg3, sg3)
ans_r3=gaussian(x, mr3, sr3)
#ans_k=gaussian(x, mk, sk)


# In[14]:


plt.plot(ans_b3, 'b')
print(max(ans_b3))
#plt.show()

plt.plot(ans_g3, 'g')
print(max(ans_g3))
#plt.show()


plt.plot(ans_r3, 'r')
print(max(ans_r3))

plt.show()


# In[16]:


c=cv2.VideoCapture("detectbuoy.avi")
first_detec=0
centersandradii=[]
n=0
while (True):
    n=n+1
    ret,image=c.read()
    image_g=image[:,:,1]
    image_r=image[:,:,2]
    image_b=image[:,:,0]
    cv2.imshow("green",image_g)
    cv2.imshow("red",image_r)
    cv2.imshow("blue",image_b)
    print(np.shape(image_r))
    if True:#ret == True:
        img_out3=np.zeros(image_g.shape, dtype = np.uint8)
        
        for i in range(0,image_g.shape[0]):
            for j in range(0,image_g.shape[1]):
                z=image_g[i][j]
                y=image_r[i][j]
                if ans_g3[z]>0.015  and  image_r[i][j]<150 and image_g[i][j]>220  and  i<480-(480/6):  
                    img_out3[i][j]=255
                else:
                    img_out3[i][j]=0  
        ret, threshold3 = cv2.threshold(img_out3, 240, 255, cv2.THRESH_BINARY)
        kernel3 = np.ones((2,2),np.uint8)
    
        dilation3 = cv2.dilate(threshold3,kernel3,iterations =9)
        contours3= cv2.findContours(dilation3, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        normal_detected_flag=0
        for contour in contours3[0]:
            
            if cv2.contourArea(contour) >  30:
                (x,y),radius = cv2.minEnclosingCircle(contour)
                center = (int(x),int(y))
                radius = int(radius)
                
                if radius > 13 and radius < 15.5:
                    cv2.circle(image,center,radius,(0,255,0),2)
                    #print(center,radius)
                    if first_detec==0:
                        first_detec=[center,radius]
                    centersandradii.append([center,radius])
                    print("normally detected",center,radius)
                    normal_detected_flag=1
        
        if normal_detected_flag!=1   and  first_detec!=0:
            center,radius=KalmannFilter(centersandradii,n)
            if center[1]<480   and   center[0]<640:
                cv2.circle(image,center,radius,(0,255,0),2)
                centersandradii.append([center,radius])
                print("Kalmann detec",center,radius)
                    
                    
                    
        cv2.imshow("Threshold",dilation3)
        cv2.imshow('YoYo1', image)
        k = cv2.waitKey(15) & 0xff
        if k == 27:
            break      # wait for ESC key to exit

    else:
        break
        
c.release()
cv2.destroyAllWindows()   





# In[ ]:





# In[ ]:





# In[ ]:




