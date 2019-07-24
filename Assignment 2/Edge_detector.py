#!/usr/bin/env python
# coding: utf-8

# # 3. Edge detector(40 points)

# In[1]:


import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import cv2
import sys


# In[2]:


def Read_Gray(img_path):
    img = plt.imread(img_path)
    RGB = np.array([0.299, 0.587, 0.114])
    return img.dot(RGB)


# In[ ]:





# ## 3.1 Blur the input image a little, $B_σ(x) = G_σ(x) ∗ I(x)$

# In[3]:


# https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html
def Blur(img, sigma=1):
#     fig = plt.figure()
#     plt.gray()  # show the filtered result in grayscale
#     ax1 = fig.add_subplot(121)  # left side
#     ax2 = fig.add_subplot(122)  # right side
#     ax1.imshow(img)
    result = ndimage.gaussian_filter(img, sigma=sigma)
#     ax2.imshow(result)
#     plt.show()
    return result


# In[ ]:





# ## 3.2 Construct a Gaussian pyramid   $P = Pyramid\{B_σ(x)\}$

# In[4]:


# https://pysource.com/2018/03/14/image-pyramids-opencv-3-4-with-python-3-tutorial-23/
def Pyramid(img, N=3):
    layer = img.copy()
    gaussian_pyramid = [layer]
    for i in range(N):
        layer = cv2.pyrDown(layer)
        gaussian_pyramid.append(layer)
    return gaussian_pyramid


# In[5]:


def Plot(L):
    fig = plt.figure(figsize=(12,16))
    plt.gray()  # show the filtered result in grayscale
    ax = [fig.add_subplot((len(L)+1)//2,2,1+i) for i in range(len(L))]
    for i in range(len(L)):
        ax[i].imshow(L[i])
    plt.show()


# In[6]:


# Plot(Pyramid(Bx))


# In[7]:


def Plot1(L):
    figsize=np.array([12,16])
    for i in L:
        plt.figure(figsize=figsize)
        plt.gray()
        plt.imshow(i)
        plt.show()
        figsize =  figsize / 2


# In[8]:


# Plot1(Pyramid(Bx))


# In[ ]:





# ## 3.3 Subtract an interpolated coarser-level pyramid image from the original resolution blurred image, $S(x) = B_σ(x) − P.InterpolatedLevel(L).$

# In[9]:


def Sub_interpolated(P):
    PL1 = cv2.pyrUp(P[1])
    a, b = min(PL1.shape[0],P[0].shape[0]), min(PL1.shape[1],P[0].shape[1])
    return cv2.subtract(P[0][:a,:b], PL1[:a,:b])


# In[10]:


# Plot([Sub_interpolated(P)])


# In[ ]:





# ## 3.4 count the number of zero crossings along the four edges

# In[11]:


def Count(S):
    C = np.zeros(S.shape)
    for i in range(S.shape[0]-1):
        for j in range(S.shape[1]-1):
            if S[i,j]*S[i+1,j]<0:
                C[i,j] += 1
            if S[i,j+1]*S[i+1,j+1]<0:
                C[i,j] += 1
            if S[i,j]*S[i,j+1]<0:
                C[i,j] += 1
            if S[i+1,j]*S[i+1,j+1]<0:
                C[i,j] += 1
    return C[:-1,:-1]


# In[12]:


# 避免循环
def Count_Quick(S):
    up_left = S[:-1,:-1]
    up_right = S[:-1,1:]
    down_left = S[1:,:-1]
    down_right = S[1:,1:]
    return 0+(up_left*up_right<0)+(up_left*down_left<0)+(down_left*down_right<0)+(down_right*up_right<0)


# In[ ]:





# ## 3.5 When there are exactly two zero crossings, compute their locations using Equation 4.25 and store these edgel endpoints along with the midpoint in the edgel structure
# ## 3.6 For each edgel, compute the local gradient by taking the horizontal and vertical differences between the values of S along the zero crossing edges.¶
# ## 3.7 Store the magnitude of this gradient as the edge strength and either its orientation or that of the segment joining the edgel endpoints as the edge orientation.

# In[13]:


class SEdgel(object):
    def __init__(self, i, j, S):
        self.grad = []
        self.e = self.__check(i,j,S)
        self.midpoint = (self.e[0]+self.e[1])/2     
        self.n = self.e[0]-self.e[1]
        self.theta = np.arctan(self.n[1]/self.n[0])
        self.length = np.linalg.norm(self.n)
        self.strength = np.sqrt(self.grad[0]**2+self.grad[1]**2)
        #np.abs(self.n[1]*i-self.n[0]*j-self.e[0][1]*self.e[1][0]+self.e[0][0]*self.e[1][1])/self.length
        
    
    def __zero_crossing(self, xi, xj, S):
        Si, Sj = S[xi[0],xi[1]], S[xj[0],xj[1]]
        return (xi*Sj-xj*Si)/(Sj-Si)
    
    def __check(self, i, j, S):
        L = []
        if S[i,j]*S[i+1,j]<0:
            L.append(self.__zero_crossing(np.array([i,j]),np.array([i+1,j]),S))
            self.grad.append(abs(S[i+1,j] - S[i,j])) 
        if S[i,j+1]*S[i+1,j+1]<0:
            L.append(self.__zero_crossing(np.array([i,j+1]),np.array([i+1,j+1]),S))
            self.grad.append(abs(S[i,j+1]*S[i+1,j+1])) 
        if S[i,j]*S[i,j+1]<0:
            L.append(self.__zero_crossing(np.array([i,j]),np.array([i,j+1]),S))
            self.grad.append(abs(S[i,j]*S[i,j+1])) 
        if S[i+1,j]*S[i+1,j+1]<0:
            L.append(self.__zero_crossing(np.array([i+1,j]),np.array([i+1,j+1]),S))
            self.grad.append(abs(S[i+1,j]*S[i+1,j+1])) 
        return L


# ## 3.8 Add the edgel to a list of edgels or store it in a 2D array of edgels (addressed by pixel coordinates).

# In[19]:


def get_strength(C, Sx):
    edgels = np.empty(C.shape,dtype=SEdgel)
    strength = np.zeros(C.shape)
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            if C[i,j]==2:
                edgels[i,j] = SEdgel(i,j,Sx)
                strength[i,j] = (edgels[i,j].strength)
            else:
                edgels[i,j] = None
    return strength


# In[26]:


def Show(strength):
    threshold = input("Please input the threshold (input any string to stop):")
    try:
        threshold = float(threshold)
    except ValueError as e:
        raise
    plt.figure()
    plt.gray()
    plt.imshow(np.where(strength>threshold,1,0))
    plt.show()
    return threshold


# In[27]:


def Show_both(img, point):
    img1 = np.where(point>0.5, point, img)
    return (img1)


# In[28]:


def main(img_path):
    img = Read_Gray(img_path)
    print("Load image successfully!")
    Bx = Blur(img,1)
    print("Blur image successfully!")
    P = Pyramid(Bx)
    print("Build Gaussian Pyramid successfully!")
    Sx = Sub_interpolated(P)
    C = Count_Quick(Sx)
    print("Get strength of each pixel...")
    strength = get_strength(C, Sx)
    print("Get strength of each pixel successfully!")
    threshold = 0
    print("Begin to select an appropriate threshold...")
    while True:
        try:
            threshold = Show(strength)
        except ValueError as e:
            break
    point = np.zeros(Bx.shape)
    point[:strength.shape[0],:strength.shape[1]] = np.where(strength>threshold,1,0)
    plt.figure()
    plt.gray()
    plt.imshow(Show_both(Bx, point))
    plt.show()
    plt.imsave(str(threshold)+"_"+img_path,np.where(strength>threshold,1,0))
    print("Save image successfully!")


# In[29]:


if __name__=="__main__":
    main(sys.argv[1])

