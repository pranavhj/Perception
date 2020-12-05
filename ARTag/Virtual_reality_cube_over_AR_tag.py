# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 23:29:49 2020

@author: prana
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 09:41:39 2020

@author: prana
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 21:31:00 2020

@author: prana
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 12:41:06 2020

@author: prana
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
import operator

def checkFilter(new_coord_temp,new_coord):
    final_coord=new_coord                         #Passing filtered values
    counter=0
    error=0
    for i in range(len(new_coord)):
        error=error+abs(new_coord_temp[i]-new_coord[i])
        if abs(new_coord_temp[i]-new_coord[i])>50:
            #final_coord[i]=new_coord_temp[i]
            counter=counter+1
    error=error/len(new_coord)
    #print(error)
    if error>40:
        final_coord=new_coord_temp
    if counter>0:
        final_coord=new_coord_temp
    return final_coord
        
#Function to check if the coo-rdinate is a valid coordinate and also if it is it returns 
#whether it is the top left, top right bottom left and bottom right

def checkCorners(thresh, coord):
    dist=8                                              #check the points at this particular distance
    count = 0 
    tleft, tright, bleft, bright = 0, 0, 0, 0
    if coord[0] < 1900 and coord[1] < 1000: 
        if thresh[coord[1]-dist][coord[0]-dist-5] == 255: 
            count = count + 1
            tleft = 1 
        if thresh[coord[1]+dist][coord[0]-dist-5] == 255:
            count = count + 1                                          #check which end of the coord is white
            bleft = 1 
        if thresh[coord[1]-dist][coord[0]+dist+5] == 255: 
            count = count + 1
            tright = 1 
        if thresh[coord[1]+dist][coord[0]+dist+5] == 255: 
            count = count + 1
            bright = 1 
        cv2.circle(thresh,(coord[0],coord[1]), 0, (255,255,255), -1)
        cv2.circle(thresh,(coord[0]-dist-5,coord[1]-dist), 1, (0,0,0), -1)
        cv2.circle(thresh,(coord[0]+dist+5,coord[1]-dist), 2, (0,0,0), -1)   #drawing circles to debug output
        cv2.circle(thresh,(coord[0]-dist-5,coord[1]+dist), 3, (0,0,0), -1)
        cv2.circle(thresh,(coord[0]+dist+5,coord[1]+dist), 4, (0,0,0), -1)
        #cv2.imshow('thresh',thresh)
        #print([coord,count])  
        if count ==3: # If any three areas are white
            
            if tleft == 1 and tright == 1 and bleft == 1: # If top left, top right and bottom left are white
                return True, 'TL' # Then the function returns true and the string 'TL'
            elif tleft == 1 and tright == 1 and bright == 1: # If top left, top right and bottom right are white
                return True, 'TR' # Then the function returns true and the string 'TR'
            elif tleft == 1 and bright == 1 and bleft == 1: 
                return True, 'BL'
            elif tright == 1 and bright == 1 and bleft == 1: 
                return True, 'BR' 
           
        else: 
            return False, None 
    else: 
        return False, None 

    

#Funtion to check whether the element c is presesnt in list lists
def checkElements(lists,c):
    for l in lists:
        if l==c:
            return True
    return False



#Function to calculate Homography between 8 points which are passed as an argument
#returns a 3x3 homography matrix
def homography(p1,p2,p3,p4,q1,q2,q3,q4):
    H=np.array([[p1[0], p1[1], 1, 0, 0, 0, -q1[0]*p1[0], -q1[0]*p1[1], -q1[0]],
            [0, 0, 0, p1[0], p1[1], 1, -q1[1]*p1[0], -q1[1]*p1[1], -q1[1]],
            [p2[0], p2[1], 1, 0, 0, 0, -q2[0]*p2[0], -q2[0]*p2[1], -q2[0]],
            [0, 0, 0, p2[0], p2[1], 1, -q2[1]*p2[0], -q2[1]*p2[1], -q2[1]],
            [p3[0], p3[1], 1, 0, 0, 0, -q3[0]*p3[0], -q3[0]*p3[1], -q3[0]],
            [0, 0, 0, p3[0], p3[1], 1, -q3[1]*p3[0], -q3[1]*p3[1], -q3[1]],
            [p4[0], p4[1], 1, 0, 0, 0, -q4[0]*p4[0], -q4[0]*p4[1], -q4[0]],
            [0, 0, 0, p4[0], p4[1], 1, -q4[1]*p4[0], -q4[1]*p4[1], -q4[1]]])
    u,s,v=np.linalg.svd(H)
    #print(v)
    
    
    h=v[len(H),:]
    h=h/h[-1]
    h=np.resize(h,(3,3))
    
    return h



#Function to form a 8x8 matrix defining all the elements of the detected image_tag
#return the 8x8 matrix which contains 1 where the tag is white and 0 where it is black
def makeMatrix(sol):
    M=np.zeros((8,8))
    width=int(len(sol)/8)
    height=int(len(sol[0])/8)
    #print([width,height])
    for i in range(8):
        for j in range(8):
            mini=sol[width*i:width*(i+1),height*j:height*(j+1)]
            #print(mini.shape)
            
            sum1=0
            for m in mini:
                for n in m:
                    sum1=sum1+n
            M[i][j]=sum1/width/height
            if M[i][j]>120:
                M[i][j]=1
            else:
                M[i][j]=0
    #print(M)
    return M



#Function to calculate the tagid if the tag inputted as a form of 8x8 matrix
#also returns the orientation of the tag so that we know which point is which to project
def Tagid(M):
    p1=M[2][2]
    p2=M[2][5]
    p3=M[5][2]
    p4=M[5][5]
    angle=None
    if p1 ==1  and p2==0  and  p3==0   and p4==0:
        angle=180
        return [True,angle]
    elif p1 ==0  and p2==0  and  p3==0   and p4==1:
        angle=0
        return [True,angle]
    elif p1 ==0  and p2==1  and  p3==0   and p4==0:
        angle=90
        return [True,angle]
    elif p1 ==0  and p2==0  and  p3==1   and p4==0:
        angle=-90
        return [True,angle]
    else:
        return [False,angle]
    
#Function to calculate the projection matrix from the Homography and the Calibration matrix   
def projectionMatrix(H,K):
    h1=H[:,0]
    h2=H[:,1]
    h3=H[:,2]
    Kinv=np.linalg.inv(K)
    lambda_=2/(np.linalg.norm(np.matmul(Kinv,h1))+np.linalg.norm(np.matmul(Kinv,h2)))
    B_tilda=np.matmul(Kinv,H)
    B=lambda_*B_tilda
    flag=1
    if np.linalg.det(B_tilda)<0:
        flag=-1
    B=flag*B
    r1=B[:,0]
    r2=B[:,1]
    r3=np.cross(r1,r2)
    t=B[:,2]
    R=np.column_stack((r1,r2,r3,t))
    P=np.matmul(K,R)
    return P

#Function to calculate the image plane co-odinates of the cube that is to be formed
def Calc_cube(P):
    x1,y1,z1 = np.matmul(P,[0,0,0,1]) # Calculating camera frame coordinates for [0,0,0] 
    x2,y2,z2 = np.matmul(P,[0,300,0,1]) # Calculating camera frame coordinates for [0,300,0] 
    x3,y3,z3 = np.matmul(P,[300,0,0,1]) # Calculating camera frame coordinates for [300,0,0] 
    x4,y4,z4 = np.matmul(P,[300,300,0,1]) # Calculating camera frame coordinates for [300,300,0] 
    x5,y5,z5 = np.matmul(P,[0,0,-300,1]) # Calculating camera frame coordinates for [0,0,-300] 
    x6,y6,z6 = np.matmul(P,[0,300,-300,1]) # Calculating camera frame coordinates for [0,300,-300] 
    x7,y7,z7 = np.matmul(P,[300,0,-300,1]) # Calculating camera frame coordinates for [300,0,-300] 
    x8,y8,z8 = np.matmul(P,[300,300,-300,1]) # Calculating camera frame coordinates for [300,300,-300] 
    
    x1_coord=x1/z1                 #calculate all the image plane co_ordinates of the cube
    y1_coord=y1/z1
    x2_coord=x2/z2
    y2_coord=y2/z2
    x3_coord=x3/z3
    y3_coord=y3/z3
    x4_coord=x4/z4
    y4_coord=y4/z4
    x5_coord=x5/z5
    y5_coord=y5/z5
    x6_coord=x6/z6
    y6_coord=y6/z6
    x7_coord=x7/z7
    y7_coord=y7/z7
    x8_coord=x8/z8
    y8_coord=y8/z8
    
    vector=[x5_coord,                 #A temproray list to return all co_ordinates calculated
    y5_coord,
    x6_coord,
    y6_coord,
    x7_coord,
    y7_coord,
    x8_coord,
    y8_coord,
    x1_coord,
    y1_coord,
    x2_coord,
    y2_coord,
    x3_coord,
    y3_coord,
    x4_coord,
    y4_coord]
    return vector
    
    
        
def drawCube(frame,P,old_coord,contour_counter):
    x1,y1,z1 = np.matmul(P,[0,0,0,1]) # Calculating camera frame coordinates for [0,0,0] 
    x2,y2,z2 = np.matmul(P,[0,300,0,1]) # Calculating camera frame coordinates for [0,300,0] 
    x3,y3,z3 = np.matmul(P,[300,0,0,1]) # Calculating camera frame coordinates for [300,0,0] 
    x4,y4,z4 = np.matmul(P,[300,300,0,1]) # Calculating camera frame coordinates for [300,300,0] 
    x5,y5,z5 = np.matmul(P,[0,0,-300,1]) # Calculating camera frame coordinates for [0,0,-300] 
    x6,y6,z6 = np.matmul(P,[0,300,-300,1]) # Calculating camera frame coordinates for [0,300,-300] 
    x7,y7,z7 = np.matmul(P,[300,0,-300,1]) # Calculating camera frame coordinates for [300,0,-300] 
    x8,y8,z8 = np.matmul(P,[300,300,-300,1]) # Calculating camera frame coordinates for [300,300,-300] 
    alpha=0.2
    x1_coord=x1/z1
    y1_coord=y1/z1
    x2_coord=x2/z2
    y2_coord=y2/z2
    x3_coord=x3/z3
    y3_coord=y3/z3
    x4_coord=x4/z4
    y4_coord=y4/z4                          #Calculate all the image plane co_odinates of the cube
    x5_coord=x5/z5
    y5_coord=y5/z5
    x6_coord=x6/z6
    y6_coord=y6/z6
    x7_coord=x7/z7
    y7_coord=y7/z7
    x8_coord=x8/z8
    y8_coord=y8/z8
    new_coord=[x5_coord,                      #store all the values in the form of a list
    y5_coord,
    x6_coord,
    y6_coord,
    x7_coord,
    y7_coord,
    x8_coord,
    y8_coord,
    x1_coord,
    y1_coord,
    x2_coord,
    y2_coord,
    x3_coord,
    y3_coord,
    x4_coord,
    y4_coord]
    new_coord_temp=new_coord                 #non filtered values
    list_2=[i * alpha for i in list(map(operator.sub,new_coord, old_coord ))] 
    new_coord=list(map(operator.add, old_coord, list_2))
    
    
    new_coord=list(new_coord)
    new_coord=new_coord                        #filtered values
    #
    
    new_coord=checkFilter(new_coord_temp,new_coord)
    #print(new_coord)         
    #print([int(x1/z1),int(y1/z1)])
    
    
    new_coord[8:15]=[
    x1_coord,
    y1_coord,                                         #use filtered values of only the upper layer 
    x2_coord,
    y2_coord,
    x3_coord,
    y3_coord,
    x4_coord,
    y4_coord]
    
    #Marking the filtered points in red
    cv2.circle(frame,(int(new_coord[8]),int(new_coord[9])), 5, (0,0,255), -1) # Drawing a circle at the pixel w.r.t [0,0,0]
    cv2.circle(frame,(int(new_coord[10]),int(new_coord[11])), 5, (0,0,255), -1) # Drawing a circle at the pixel w.r.t [0,300,0]
    cv2.circle(frame,(int(new_coord[12]),int(new_coord[13])), 5, (0,0,255), -1) # Drawing a circle at the pixel w.r.t [300,0,0]
    cv2.circle(frame,(int(new_coord[14]),int(new_coord[15])), 5, (0,0,255), -1) # Drawing a circle at the pixel w.r.t [300,300,0] 
    cv2.circle(frame,(int(new_coord[0]),int(new_coord[1])), 5, (0,0,255), -1) # Drawing a circle at the pixel w.r.t [0,0,-300]
    cv2.circle(frame,(int(new_coord[2]),int(new_coord[3])), 5, (0,0,255), -1) # Drawing a circle at the pixel w.r.t [0,300,-300]
    cv2.circle(frame,(int(new_coord[4]),int(new_coord[5])), 5, (0,0,255), -1) # Drawing a circle at the pixel w.r.t [300,0,-300]
    cv2.circle(frame,(int(new_coord[6]),int(new_coord[7])), 5, (0,0,255), -1)
    
    
    
    #Marking the non filtered points in green
    cv2.circle(frame,(int(new_coord_temp[8]),int(new_coord_temp[9])), 5, (0,255,0), -1) # Drawing a circle at the pixel w.r.t [0,0,0]
    cv2.circle(frame,(int(new_coord_temp[10]),int(new_coord_temp[11])), 5,(0,255,0), -1) # Drawing a circle at the pixel w.r.t [0,300,0]
    cv2.circle(frame,(int(new_coord_temp[12]),int(new_coord_temp[13])), 5, (0,255,0), -1) # Drawing a circle at the pixel w.r.t [300,0,0]
    cv2.circle(frame,(int(new_coord_temp[14]),int(new_coord_temp[15])), 5, (0,255,0), -1) # Drawing a circle at the pixel w.r.t [300,300,0] 
    cv2.circle(frame,(int(new_coord_temp[0]),int(new_coord_temp[1])), 5, (0,255,0), -1) # Drawing a circle at the pixel w.r.t [0,0,-300]
    cv2.circle(frame,(int(new_coord_temp[2]),int(new_coord_temp[3])), 5, (0,255,0), -1) # Drawing a circle at the pixel w.r.t [0,300,-300]
    cv2.circle(frame,(int(new_coord_temp[4]),int(new_coord_temp[5])), 5, (0,255,0), -1) # Drawing a circle at the pixel w.r.t [300,0,-300]
    cv2.circle(frame,(int(new_coord_temp[6]),int(new_coord_temp[7])), 5, (0,255,0), -1)
    
    
    
    
    cv2.line(frame,(int(new_coord[8]),int(new_coord[9])),(int(new_coord[0]),int(new_coord[1])), (0,0,255), 5) 
    # Drawing a line between pixels corresponding to [0,300,0] and [0,300,-300]
    cv2.line(frame,(int(new_coord[10]),int(new_coord[11])),(int(new_coord[2]),int(new_coord[3])), (0,0,255), 5)
    # Drawing a line between pixels corresponding to [300,0,0] and [300,0,-300]
    cv2.line(frame,(int(new_coord[12]),int(new_coord[13])),(int(new_coord[4]),int(new_coord[5])), (0,0,255), 5)
    # Drawing a line between pixels corresponding to [300,300,0] and [300,300,-300]
    cv2.line(frame,(int(new_coord[14]),int(new_coord[15])),(int(new_coord[6]),int(new_coord[7])), (0,0,255), 5)
    # Drawing a line between pixels corresponding to [0,0,0] and [0,300,0]
    
    
    cv2.line(frame,(int(new_coord[8]),int(new_coord[9])),(int(new_coord[10]),int(new_coord[11])), (0,0,255), 5)
    # Drawing a line between pixels corresponding to [0,0,0] and [300,0,0]
    cv2.line(frame,(int(new_coord[8]),int(new_coord[9])),(int(new_coord[12]),int(new_coord[13])), (0,0,255), 5)
    # Drawing a line between pixels corresponding to [0,300,0] and [300,300,0]
    cv2.line(frame,(int(new_coord[14]),int(new_coord[15])),(int(new_coord[10]),int(new_coord[11])), (0,0,255), 5)
    # Drawing a line between pixels corresponding to [300,0,0] and [300,300,0]
    cv2.line(frame,(int(new_coord[14]),int(new_coord[15])),(int(new_coord[12]),int(new_coord[13])), (0,0,255), 5)
    # Drawing a line between pixels corresponding to [0,0,-300] and [0,300,-300]
    
    
    cv2.line(frame,(int(new_coord[0]),int(new_coord[1])),(int(new_coord[2]),int(new_coord[3])), (0,0,255), 5)
    # Drawing a line between pixels corresponding to [0,0,-300] and [300,0,-300]
    cv2.line(frame,(int(new_coord[0]),int(new_coord[1])),(int(new_coord[4]),int(new_coord[5])), (0,0,255), 5)
    # Drawing a line between pixels corresponding to [0,300,-300] and [300,300,-300]
    cv2.line(frame,(int(new_coord[2]),int(new_coord[3])),(int(new_coord[6]),int(new_coord[7])), (0,0,255), 5)
    # Drawing a line between pixels corresponding to [300,0,-300] and [300,300,-300]
    cv2.line(frame,(int(new_coord[4]),int(new_coord[5])),(int(new_coord[6]),int(new_coord[7])), (0,0,255), 5)
    
    
    old_coord=new_coord
    return frame,old_coord            
    
def projectImage(img_1,P1,frame1):
    for i in range(len(img_1)):
            for j in range(len(img_1[0])):
                X=np.matmul(P1,[i,j,0,1])
                X=X/X[-1]
                if X[1]<1080   and   X[1]>0   and  X[0]<1920   and   X[0]>0 : 
                    frame1[int(X[1])][int(X[0])]=img_1[j][i]
    return frame1


def alignPoints(angle,list4):
    list5=[]   
    if angle==90:                              #To always align the points detected with the world img
        list5=[list4[2],list4[0],list4[3],list4[1]]
    elif angle==-90:
        list5=[list4[1],list4[3],list4[0],list4[2]]
    elif angle==180:
        list5=[list4[3],list4[2],list4[1],list4[0]]
    elif angle==0:
        list5=list4
    return list5



img = cv2.imread('ref_marker.png',0)
img_1=cv2.imread('lena.png',0)
c=0
ret=1
cap=cv2.VideoCapture('Tag2.mp4')
#cap=cv2.VideoCapture('Tag1.mp4')
K = np.array([[1406.08415449821,0,0],[2.20679787308599, 1417.99930662800,0],[1014.13643417416, 566.347754321696,1]]).T
sol_frame_dimen=50
P_old=[]
P1_old=[]
angle_old=[0,0,0,0]
frame_num=0
old_coord=[[],[],[],[]]
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('Tag2.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
while (cap.isOpened())    :
    #Video Capture
    
    ret,frame=cap.read()
    frame1=frame
    image=frame
    
    if len(np.shape(frame))!=3:
        break
    blur = cv2.GaussianBlur(frame,(9,9),0) 
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY) 
    ret, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY) 
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) 
    list2 = [] 
    contour_counter=0
    
    for cnt in contours: 
        list1 = [] 
        
        epsilon = 0.1*cv2.arcLength(cnt,True) # Approximating the contour using arcLength method
        cnt = cv2.approxPolyDP(cnt, epsilon, True) # Determining the coordinates of the contour
        if   len(cnt)==4  and    cv2.arcLength(cnt,True)>100            : 
            #frame=cv2.drawContours(frame,cnt,-1,(250,0,0),3)
            
            




            #print(cv2.arcLength(cnt,True))
            for coor in cnt: # Looping over the corners found form the contour

                result, key = checkCorners(thresh, [coor[0][0], coor[0][1]]) 
                if result == True  : # If the checkCorner function returns True
                    list1.append([coor[0][0], coor[0][1], key]) # Then the coordinates are appended into list1
                else:
                    cv2.circle(frame,(coor[0][0], coor[0][1]),10,(255,255,0),-1)
                    
            if len(list1) == 4: # If the length of list1 is 4 i.e. four corners on the AR tag are identified
                list2.append(list1) # Then the corners are appended to list2
               
#    if len(list2)>1:    
#        print(list2)
    #print(contour_num)
    #print(len(list2))
    
    
    
    for list2_num in list2:                                          #To iterate through all tags detected
        #print(list2_num)    
        list3=[]               #to store co_ords only
        
        
        if len(list2_num)>0:                              #for removing tag data to draw
            for c in list2_num:
                #print(c)
                list3.append([c[0],c[1]])
        
        
        for c in list3:
            cv2.circle(frame,(c[0], c[1]),10,(255,0,0),-1)             #marking the detected 4 points in blue   
            #print(list3)    
            #frame=cv2.drawContours(frame,np.array([list3]),-1,(250,0,0),3)  
        #print(list2)   
        if list2_num:
            #list2_num=list2_num[0]
            
            #print(list2)
            list4=[0,0,0,0]                                     #to fill according to tleft tright bleft bright
            for c in list2_num:
                #print(c)
                if c[2]=='TL':
                    list4[0]=[c[0],c[1]]
                if c[2]=='TR':
                    list4[1]=[c[0],c[1]]
                if c[2]=='BL':
                    list4[2]=[c[0],c[1]]
                if c[2]=='BR':
                    list4[3]=[c[0],c[1]]
            
           
        
        #Calculate Homography for projecting the Tag and compensating all orientations encountered  
        H=homography([0,0],[0,sol_frame_dimen-1],[sol_frame_dimen-1,0],[sol_frame_dimen-1,sol_frame_dimen-1],list4[0],list4[1],list4[2],list4[3])
        #H1=np.linalg.inv(H)
        sol=np.zeros((sol_frame_dimen-1,sol_frame_dimen-1))
        im_out=sol
        for i in range(sol_frame_dimen-1):
            for j in range(sol_frame_dimen-1):
                X=np.matmul(H,[i,j,1])
                
                X=X/X[-1]
                #print(X)
                
                if X[1]<1080   and   X[1]>0   and  X[0]<1920   and   X[0]>0 :    
                    #print("hi")
                    #frame[int(X[1])][int(X[0])]=[255,255,255]
                    sol[i][j]=thresh[int(X[1])][int(X[0])]
    
        #cv2.imshow('sol',sol) 
        M=makeMatrix(sol)
        result,angle=Tagid(M)
        #print(result)
        identity=None
        list5=[]
        #print("yes")
        print(result)
        if(result == True): # If true is returned by calculateTagAngle i.e. angle is identifiable 
            if (angle == 0): # If the orientation is 0 degree
                identity = M[3][3] +M[4][3]*8 +M[4][4]*4 + M[3][4]*2 # Calculation of the tag id corresponding to the 0 degree configuration
            elif(angle == 90): # If the orientation is 90 degree
                identity = M[3][3]*2 + M[3][4]*4 + M[4][4]*8 + M[4][3] # Calculation of the tag id corresponding to the 90 degree configuration
            elif(angle == 180): # If the orientation is 180 degree
                identity = M[3][3]*4 + M[4][3]*2 + M[4][4] + M[3][4]*8 # Calculation of the tag id corresponding to the 180 degree configuration
            elif(angle == -90): # If the orientation is -90 degree
                identity = M[3][3]*1 + M[3][4] + M[4][4]*2 + M[4][3]*4 # Calculation of the tag id corresponding to the -90 degree configuration
        
             
                
                
            list5=alignPoints(angle,list4)    
            angle_old[contour_counter]=angle
            
            
            #print(list5)
            
            cv2.putText(frame,str(identity),(list5[0][0],list5[0][1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
            Hcw = homography([0,0],[0,300],[300,0],[300,300],list5[0],list5[1],list5[2],list5[3])  
            #Hcw1 = homography([0,0],[0,511],[511,0],[511,511],list5[0],list5[1],list5[2],list5[3])
            #print(Hcw)
            P=projectionMatrix(Hcw,K)
            #P1=projectionMatrix(Hcw1,K)
    #        
    #       
            
    
            if frame_num==0:
                old_coord[contour_counter]=Calc_cube(P) 
            
            
            frame,old_coord[contour_counter]=drawCube(frame,P,old_coord[contour_counter],contour_counter)
            contour_counter=contour_counter+1
        #what happens if Tagid function returns None i.e. angle could not be calculated. We use the angle we got in 
        #the previous frame and use that to align points found
        else:
            list5=alignPoints(angle_old[contour_counter],list4)
            Hcw = homography([0,0],[0,300],[300,0],[300,300],list5[0],list5[1],list5[2],list5[3])  
            P=projectionMatrix(Hcw,K)
            #print("FilterLoop")
            #frame,old_coord[contour_counter]=drawCube(frame,P,old_coord[contour_counter],contour_counter)
            contour_counter=contour_counter+1
            
        #print(angle_old)
        

      
    #cv2.imshow('frame1',frame1)    
    cv2.imshow('frame',frame)
    out.write(frame)
    frame_num=frame_num+1
    #old_coord=[[],[],[],[]]
    #cv2.imshow('thresh',thresh)
    if cv2.waitKey(1)   &  0xFF==ord('q'):
        break
#ImageProcessor(video_url,img)

cap.release()
cv2.destroyAllWindows()
