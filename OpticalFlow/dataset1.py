#!/usr/bin/env python
# coding: utf-8



import numpy as np
import cv2
import copy
from scipy.interpolate import RectBivariateSpline
import os
import glob

 
#                       template     image           rect        x,y_oldiz
def LKTFunction(oldtemp, oldtemp1, rectpoints, pos0=np.zeros(2)):
    
    threshold = 0.001 # Threshold for convergence of the error of the parameters
    x1, y1, x2, y2 = rectpoints[0], rectpoints[1], rectpoints[2], rectpoints[3] # Top-Left and Bottom-Right Corners of the template
    old_y, old_x = np.gradient(oldtemp1) # Calculating difference in intensity
    #print(np.shape(old_x))
    dp = 1 # oldizing the variable for storing error in the parameters

    while np.square(dp).sum() > threshold: # Looping until the solution converges below the threshold
        
        posx, posy = pos0[0], pos0[1] # old Parameters
        x1_warp, y1_warp, x2_warp, y2_warp = x1 + posx, y1 + posy, x2 + posx, y2 + posy # Warped Parameters
        w_t=87
        h_t=36
        x = np.arange(0, oldtemp.shape[0], 1)
        y = np.arange(0, oldtemp.shape[1], 1)
        
        a=np.arange(float(x1),float(x2),float((x2-x1)/w_t))
        b=np.arange(float(y1),float(y2),float((y2-y1)/h_t))
        aa,bb=np.meshgrid(a,b)
        #print(np.shape(aa),"  ",np.shape(bb))
        

        a_warp=np.arange(float(x1_warp),float(x2_warp),float((x2_warp-x1_warp)/w_t))
        b_warp=np.arange(float(y1_warp),float(y2_warp),float((y2_warp-y1_warp)/h_t))
        aa_warp,bb_warp=np.meshgrid(a_warp,b_warp)
        
        spline = RectBivariateSpline(x, y, oldtemp) # Smoothing and Interpolating intensity data over all the template frame
        T = spline.ev(bb, aa) # Evaluating the intensity data over all the interpolated points

        spline1 = RectBivariateSpline(x, y, oldtemp1) # Smoothing and Interpolating intensity data over all the next frame
        warpImg = spline1.ev(bb_warp, aa_warp) # Evaluating the intensity data over all the warped interpolated points

        error = T - warpImg  # Calculating the change in intensity from the template frame to the next frame
        #print(np.shape(error),"error")
        errorImg = np.reshape(error,(h_t*w_t,1))
        #print(np.shape(errorImg))
        spline_gx = RectBivariateSpline(x, y, old_x) # Smoothing and Interpolating intensity gradient data over all the next frame in x-direction
        old_x_warp = spline_gx.ev(bb_warp, aa_warp) # Evaluating intensity gradient data over all the interpolated points in x-direction

        spline_gy = RectBivariateSpline(x, y, old_y) # Smoothing and Interpolating intensity gradient data over all the next frame in y-direction
        old_y_warp = spline_gy.ev(bb_warp, aa_warp) # Evaluating intensity gradient data over all the interpolated points in y-direction
        
        I=[]
        I.append(old_x_warp.ravel())
        I.append(old_y_warp.ravel())
        I=np.transpose(I)
        #print(I)
        jacobian = np.array([[1, 0], [0, 1]])

        hessian = np.matmul(I , jacobian) # oldizing the Jacobian
        H = np.matmul(hessian.T , hessian) # Calculating Hessian Matrix
        dpdash=np.matmul(np.linalg.inv(H) , hessian.T)
        dp = np.matmul(dpdash, errorImg) # Calculating the change in parameters
        # Updating the previous parameters
        pos0[0] += dp[0, 0]
        pos0[1] += dp[1, 0]

    p = pos0
    return p # Returing the updated parameters


images = []
images = []
path = 'Car4\\'



images = [cv2.imread(file) for file in glob.glob(r"Car4/img/*.jpg")]


print(np.shape(images))
images=images

rectpoints = [75,48,181,138] # Template Rectangular points, calculated manually
width = rectpoints[3] - rectpoints[1] # Calculating the width of the template
length = rectpoints[2] - rectpoints[0] # Calculating the length of the template
rectpoints0 = copy.deepcopy(rectpoints)
frame_original = images[0]
frame_gray_original = cv2.cvtColor(frame_original, cv2.COLOR_BGR2GRAY)
frame_gray_original = cv2.equalizeHist(frame_gray_original)
oldtemp0 = frame_gray_original / 255.
out = cv2.VideoWriter('Car.avi',cv2.VideoWriter_fourcc(*'DIVX'), 25, (360,240))
for i in range(0, len(images)-1): # Looping over all the images
    

    i = i
    
    if i == 100:
        rectpoints = [76,55,169,130]
        width = rectpoints[3] - rectpoints[1]
        length = rectpoints[2] - rectpoints[0]
        rectpoints0 = copy.deepcopy(rectpoints)
        frame_original = images[100]
        frame_gray_original = cv2.cvtColor(frame_original, cv2.COLOR_BGR2GRAY)
        frame_gray_original = cv2.equalizeHist(frame_gray_original)
        print("template changed")
        oldtemp0 = frame_gray_original / 255.
#

#
    if i == 200:
        rectpoints = [140,62,211,119]
        width = rectpoints[3] - rectpoints[1]
        length = rectpoints[2] - rectpoints[0]
        rectpoints0 = copy.deepcopy(rectpoints)
        frame_original = images[200]
        frame_gray_original = cv2.cvtColor(frame_original, cv2.COLOR_BGR2GRAY)
        frame_gray_original = cv2.equalizeHist(frame_gray_original)
        print("template changed")
        oldtemp0 = frame_gray_original / 255.
        
        
        
    frame = images[i]
    frame_= frame
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray) # Converting image into the grayscale
    # Drawing rectangle over the object to track
    cv2.rectangle(frame_,(int(rectpoints[0]),int(rectpoints[1])),(int(rectpoints[0])+length,int(rectpoints[1])+width),(0,255,0),3)
    
    cv2.imshow('Car_Tracking', frame) # Showing output of the tracking
    print(np.shape(frame))
    out.write(frame)

    frame_next = images[i+1]
    frame_gray_next = cv2.cvtColor(frame_next, cv2.COLOR_BGR2GRAY)
    frame_gray_next = cv2.equalizeHist(frame_gray_next)

    #oldtemp0 = frame_gray_original / 255.
    oldtemp1 = frame_gray_next / 255.
    
    p = LKTFunction(oldtemp0, oldtemp1, rectpoints0) # Calling Lucas Kanade Function
    # Updating the rectangular coordinates from the recieved paramters
    rectpoints[0] = rectpoints0[0] + p[0]
    rectpoints[1] = rectpoints0[1] + p[1]
    rectpoints[2] = rectpoints0[2] + p[0]
    rectpoints[3] = rectpoints0[3] + p[1]
    print(p)
    if cv2.waitKey(1)   &  0xFF==ord('q'):
        break

cv2.destroyAllWindows()

out.release()
#break

