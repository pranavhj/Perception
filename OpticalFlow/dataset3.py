#!/usr/bin/env python
# coding: utf-8



import numpy as np
import cv2
import copy
from scipy.interpolate import RectBivariateSpline
import os
import glob

 
#                       template     image           rect        x,y_initializ
def LKTFunction(initialtemp, initialtemp1, rectpoints, pos0=np.zeros(2)):
    
    threshold = 0.001 
    x1, y1, x2, y2 = rectpoints[0], rectpoints[1], rectpoints[2], rectpoints[3]
    initial_y, initial_x = np.gradient(initialtemp1) 
    #print(np.shape(initial_x))
    dp = 1 

    while np.square(dp).sum() > threshold:
        
        posx, posy = pos0[0], pos0[1] 
        x1_warp, y1_warp, x2_warp, y2_warp = x1 + posx, y1 + posy, x2 + posx, y2 + posy 
        w_t=87
        h_t=36
        x = np.arange(0, initialtemp.shape[0], 1)
        y = np.arange(0, initialtemp.shape[1], 1)
        
        a=np.arange(float(x1),float(x2),float((x2-x1)/w_t))
        b=np.arange(float(y1),float(y2),float((y2-y1)/h_t))
        aa,bb=np.meshgrid(a,b)
        #print(np.shape(aa),"  ",np.shape(bb))
        

        a_warp=np.arange(float(x1_warp),float(x2_warp),float((x2_warp-x1_warp)/w_t))
        b_warp=np.arange(float(y1_warp),float(y2_warp),float((y2_warp-y1_warp)/h_t))
        aa_warp,bb_warp=np.meshgrid(a_warp,b_warp)
        
        spline = RectBivariateSpline(x, y, initialtemp)
        T = spline.ev(bb, aa) 

        spline1 = RectBivariateSpline(x, y, initialtemp1)
        warpImg = spline1.ev(bb_warp, aa_warp)

        error = T - warpImg  
        #print(np.shape(error),"error")
        errorImg = np.reshape(error,(h_t*w_t,1))
        #print(np.shape(errorImg))
        spline_gx = RectBivariateSpline(x, y, initial_x) 
        initial_x_warp = spline_gx.ev(bb_warp, aa_warp)

        spline_gy = RectBivariateSpline(x, y, initial_y) 
        initial_y_warp = spline_gy.ev(bb_warp, aa_warp) 
        
        I=[]
        I.append(initial_x_warp.ravel())
        I.append(initial_y_warp.ravel())
        I=np.transpose(I)
        #print(I)
        jacobian = np.array([[1, 0], [0, 1]])

        hessian = np.matmul(I , jacobian)
        H = np.matmul(hessian.T , hessian) 
        dpdash=np.matmul(np.linalg.inv(H) , hessian.T)
        dp = np.matmul(dpdash, errorImg) 
        # Updating the previous parameters
        pos0[0] += dp[0, 0]
        pos0[1] += dp[1, 0]

    p = pos0
    return p 


images = []
images = []




images = [cv2.imread(file) for file in glob.glob(r"DragonBaby/DragonBaby/img/*.jpg")]


print(np.shape(images))
images=images

rectpoints = [149,65,218,147] # Template Rectangular points, calculated manually
width = rectpoints[3] - rectpoints[1] # Calculating the width of the template
length = rectpoints[2] - rectpoints[0] # Calculating the length of the template
rectpoints0 = copy.deepcopy(rectpoints)
frame_original = images[0]
frame_gray_original = cv2.cvtColor(frame_original, cv2.COLOR_BGR2GRAY)
frame_gray_original = cv2.equalizeHist(frame_gray_original)
initialtemp0 = frame_gray_original / 255.
#out = cv2.VideoWriter('Tag2.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (340,360))
out = cv2.VideoWriter('baby.avi',cv2.VideoWriter_fourcc(*'DIVX'), 25, (640,360))
for i in range(0, len(images)-1): # Looping over all the images

    image_index = i
    print(i)
    

        
    frame = images[image_index]
    mainframe= frame
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray) # Converting image into the grayscale
    # Drawing rectangle over the object to track
    cv2.rectangle(mainframe,(int(rectpoints[0]),int(rectpoints[1])),(int(rectpoints[0])+length,int(rectpoints[1])+width),(0,255,0),3)
    cv2.imshow('Baby', frame) # Showing output of the tracking
    out.write(frame)

    frame_next = images[image_index+1]
    frame_gray_next = cv2.cvtColor(frame_next, cv2.COLOR_BGR2GRAY)
    frame_gray_next = cv2.equalizeHist(frame_gray_next)

    #initialtemp0 = frame_gray_original / 255.
    initialtemp1 = frame_gray_next / 255.
    
    p = LKTFunction(initialtemp0, initialtemp1, rectpoints0) # Calling Lucas Kanade Function
    # Updating the rectangular coordinates from the recieved paramters
    rectpoints[0] = rectpoints0[0] + p[0]
    rectpoints[1] = rectpoints0[1] + p[1]
    rectpoints[2] = rectpoints0[2] + p[0]
    rectpoints[3] = rectpoints0[3] + p[1]
    #print(p)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break      # wait for ESC key to exit
#if cv2.waitKey(1) == 27:
out.release()
cv2.destroyAllWindows()
#break

