# -*- coding: utf-8 -*-
"""
Created on Mon May 20 17:39:04 2019

@author: Guang Jin
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp2d
import time
import cv2
import sys

##### Q1
# 选定方法：画出两个图片，对比着来选择，每次是img1先选一个点，img2选后一个点。两两匹配，最后多一个是为了确定，会舍弃掉。
def correspondences(src, dst):
    plt.figure()
    plt.subplot(121)
    plt.imshow(src)
    plt.subplot(122)
    plt.imshow(dst)
    pos=plt.ginput(-1, -1)
    plt.close()
    return np.array(pos).reshape(-1,2,2)

###########Q1 SIFT
def SIFT(img1, img2, N):
    img1=cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2=cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    # https://pythonprogramming.net/feature-matching-homography-python-opencv-tutorial/
    orb = cv2.ORB_create()        
    kp1, des1 = orb.detectAndCompute(img1,None)      
    kp2, des2 = orb.detectAndCompute(img2,None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)    
    # http://www.voidcn.com/article/p-mapsozvm-bpc.html
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    return src_pts[:N], dst_pts[:N]


##### Q2
def GetA(src, dst):
    def GetRow(src, dst, i):
        L1 = [src[i,0],src[i,1],1,0,0,0,-src[i,0]*dst[i,0],-src[i,1]*dst[i,0]]
        L2 = [0,0,0,src[i,0],src[i,1],1,-src[i,0]*dst[i,1],-src[i,1]*dst[i,1]]
        return [L1,L2]
    A = [ GetRow(src,dst,i) for i in range(len(dst))]
    L = []
    for i in range(len(src)):
        L.extend(GetRow(src,dst,i))
    assert((np.array(A).reshape(-1,8)==np.array(L))).all()
    return np.array(L)
# λp' = Hp
# img1 -> img2
def homography_matrix(pts):
    src = pts[:,0,:]
    dst = pts[:,1,:]
    b = dst.reshape(-1)
    A = GetA(src,dst)
    h =  np.linalg.pinv(A).dot(b)
    H = np.append(h,1).reshape(3,3)
    return H

def homography_matrix_srcdst(src,dst):
    b = dst.reshape(-1)
    A = GetA(src,dst)
    h =  np.linalg.pinv(A).dot(b)
    H = np.append(h,1).reshape(3,3)
    return H

##### Q3
def Homography_Dot(H, pos, Int=True):
    assert pos.shape[0] == 2
    add_one = np.ones((3, pos.shape[1]))
    add_one[:2,:] = pos[::-1,:]
    new_pos = H.dot(add_one)
    new_pos =((new_pos/new_pos[2])[:2])
    if Int:
        new_pos = np.round(new_pos).astype(np.int32)
    return new_pos[::-1,:]

def Get_New_Corner(img, H):
    # 获得边界（拐角）
    a, b = img.shape[0], img.shape[1]
    # reshape 为 3*4 方便进行矩阵乘法，λp' = Hp
    corners = np.array([[0,0],[a-1,0],[0,b-1],[a-1,b-1]]).T  
    new_cor = Homography_Dot(H, corners)
    # 获得边界
    MaxX, MaxY = np.max(new_cor, axis=1)
    MinX, MinY = np.min(new_cor, axis=1)
    return MinX, MinY, MaxX, MaxY


def Warp_img(img, H, fill_value=0, dtype=np.uint8, interpolate = 0):
    a, b = img.shape[0], img.shape[1]
    MinX, MinY, MaxX, MaxY = Get_New_Corner(img, H)
    xBias = -min(0,MinX)
    yBias = -min(0,MinY)
    X = np.arange(min(0,MinX),max(MaxX+1,a))
    Y = np.arange(min(0,MinY),max(MaxY+1,b))
    print("result.shape", (len(X),len(Y)))
    result = np.zeros((len(X),len(Y),img.shape[2]))
    invH = np.linalg.inv(H)
    ip = []
    for i in range(img.shape[2]):
        ip.append(interp2d(range(a),range(b),img[:,:,i].astype(np.float64).T,fill_value=fill_value)) 
    mod = len(X)//10
    count = 0
    def f(x):
        if x<0:
            return 99999
        else:
            return int(x)
    start = time.time()
    for i in X:
        for j in Y:
            x, y = Homography_Dot(invH, np.array([i,j]).reshape(2,-1))
            for k in range(len(ip)):
                try:
                    if interpolate==0:
                        result[i+xBias,j+yBias,k] = img[f(x),f(y),k]#ip[k](x,y)
                    else:
                        result[i+xBias,j+yBias,k] = ip[k](x,y)
                except IndexError:
                    result[i+xBias,j+yBias,k] = fill_value
        count += 1
        if count % mod == 0:
            print(str(count//mod)+"/10"+" time: "+str(time.time()-start)+"s")
    return xBias, yBias, result.astype(dtype)


##### main
if __name__=="__main__":
    Q = input("Which question do you want to reproduce? Choose 1, 2 or 3. 0 is use your own images.")
    
    if Q=="1":
        img1_path = "uttower1.jpg"
        img2_path = "uttower2.jpg"
        img1 = plt.imread(img1_path)
        img2 = plt.imread(img2_path)
        print("Load images successfully!")
#    if(sys.argv[1]==str(0)):
#    print("Please choose serveral pairs of points(>=4)!")
#    pts = correspondences(img1, img2)
#    assert pts.shape[0] >= 4
#    else:
        Choose = input("Your hand is 0. My hand is 1. SIFT is 2.")
        if Choose=="0":
            print("Please choose serveral pairs of points(>=4)!")
            pts = correspondences(img1, img2)
            src = pts[:,0,:]
            dst = pts[:,1,:]
            assert pts.shape[0] >= 4
        elif Choose=="1":
            pts = np.load("uttower_pts.npy")
            src = pts[:,0,:]
            dst = pts[:,1,:]
        elif Choose=="2":
            N = int(input("How many points?"))
            src, dst =SIFT(img1, img2, N)
        else:
            print("Please write 0, 1 or 2! Goodbye!")
            assert False
        print("Load correspondences successfully!")
        Choose = input("Least squares method: 1. RANSAC: 2.")
        if Choose=="1":
            H = homography_matrix_srcdst(src, dst)
        elif Choose=="2":
            ransacReprojThreshold = 4
            H, status = cv2.findHomography(src.reshape(-1,1,2), dst.reshape(-1,1,2), cv2.RANSAC, ransacReprojThreshold)
        else:
            print("Please write 1 or 2! Goodbye!")
            assert False
        print("Begin to compute inter2d...")
        xBias, yBias, result = Warp_img(img1, H)
        result[xBias:img2.shape[0]+xBias,yBias:img2.shape[1]+yBias,:] = img2 ### Q4
        plt.imshow(result.astype(np.uint8))
        plt.show()
    elif Q=="2":
        img1_path = "NJU2.png"
        img2_path = "NJU1.png"
        img1 = plt.imread(img1_path)
        img2 = plt.imread(img2_path)
        print("Load images successfully!")
        
        pts = np.load("NJU.npy")
        pts = pts[:,::-1,:]
        print("Load correspondences successfully!")
        H = homography_matrix(pts)
        print("Begin to compute inter2d...")
        xBias, yBias, result = Warp_img(img1, H, dtype=np.float32)
        result[xBias:img2.shape[0]+xBias,yBias:img2.shape[1]+yBias,:] = img2 ### Q4
        plt.imshow((result*255).astype(np.uint8))
        plt.show()
         
        

    elif Q=="3":
        img1_path = "test1-3-in.jpg"
        img2_path = "test1-3.png" 
        img1 = plt.imread(img1_path)
        img2 = plt.imread(img2_path)
        print("Load images successfully!")
        
        pts = np.load("Tom_pts.npy")
        print("Load correspondences successfully!")
        H = homography_matrix(pts)
        
        print("Begin to compute inter2d...")
        xBias, yBias, result = Warp_img(img1, H, -1, np.int32)
#        result = result.astype(np.uint8)
        img2 = (img2*255).astype(np.uint8)
        # result[xBias:img2.shape[0]+xBias,yBias:img2.shape[1]+yBias,:] = img2 ### Q4
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                for k in range(result.shape[2]):
                    if result[i,j,k]!=-1:
                        img2[i,j,k] = result[i,j,k]       
        plt.imshow(img2)
        plt.show()
    elif Q=="0":
        img1_path = input("img1_path: ")
        img2_path = input("img2_path: ")
        img1 = plt.imread(img1_path)
        img2 = plt.imread(img2_path)
        print("Load images successfully!")
        Choose = input("Your hand is 1. SIFT is 2.")
        if Choose=="1":
            print("Please choose serveral pairs of points(>=4)!")
            pts = correspondences(img1, img2)
            src = pts[:,0,:]
            dst = pts[:,1,:]
            assert pts.shape[0] >= 4
        elif Choose=="2":
            N = int(input("How many points?"))
            src, dst =SIFT(img1, img2, N)
        else:
            print("Please write 1 or 2! Goodbye!")
            assert False
        print("Load correspondences successfully!")
        Choose = input("Least squares method: 1. RANSAC: 2.")
        if Choose=="1":
            H = homography_matrix_srcdst(src, dst)
        elif Choose=="2":
            ransacReprojThreshold = 4
            H, status = cv2.findHomography(src.reshape(-1,1,2), dst.reshape(-1,1,2), cv2.RANSAC, ransacReprojThreshold)
        else:
            print("Please write 1 or 2! Goodbye!")
            assert False
        print("Begin to compute inter2d...")
        xBias, yBias, result = Warp_img(img1, H)
        result[xBias:img2.shape[0]+xBias,yBias:img2.shape[1]+yBias,:] = img2 ### Q4
        plt.imshow(result.astype(np.uint8))
        plt.show()
    else:
        print("Please write 0, 1, 2 or 3! Goodbye!")


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    