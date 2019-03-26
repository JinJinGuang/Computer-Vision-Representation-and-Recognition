#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import cv2


# In[2]:


type_int = [int, np.int, np.int0, np.int16, np.int32, np.int64, np.int8, np.integer, np.uint, np.uint0,
            np.uint16, np.uint32, np.uint64, np.uint8, np.uintc, np.uintp ]


# In[3]:


# 实现 Question3.1(i)
#这个函数输入路径，可以选择得到灰度还是RGB彩色图，默认得到对应的灰度图像
def cvt_color(img_path, gray=True, plot=False):
    img = cv2.imread(img_path)
    if gray:#BGR2GRAY
        img = img.dot(np.array([0.114, 0.587, 0.299])).astype("uint8")
    else:
        img = img[:,:,::-1]
    if plot:
        plt.figure()
        if gray:
            plt.imshow(img,cmap="gray")
            plt.title("Gray Image")
        else:
            plt.imshow(img)
            plt.title("Colorful Image")
    return img

#这个函数返回统计过的count，附加作用可以选择是否要count画图
def Count(img, plot=False):
    
    #确定 image type: float or int? 这里需要 int 来统计
    if type(img[0,0]) not in type_int:
        img = (img*255).astype("uint8")
    #统计数目
    count = np.zeros(256,dtype="int")
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            count[img[i,j]] += 1
    
    #画图
    if plot:
        plt.figure()
        plt.plot(count)
        plt.title("histogram: count")
        
    return count

#这个函数返回计算好的累积分布函数cdf，附加作用可以选择是否要cdf画图
def CDF(count, plot=False):

    #计算 累计分布函数 cdf
    N = np.sum(count)
    cdf = np.empty(256)
    cdf[0] = count[0]/N
    for i in range(1,256):
        cdf[i] = cdf[i-1]+count[i]/N

    #画图
    if plot:
        plt.figure()
        plt.plot(cdf)
        plt.title("CDF")
        plt.ylim((-0.05,1.05))
    
    return cdf

# 实现 Question3.1(ii)
# input: gray image  output: 操作后的image
# 可以选择是否画所有图 plot 只画结果图， all_plot 画所有图
def histogram_equalization(img, alpha=1, plot=False, all_plot=False):
    #step1: 统计数目
    count = Count(img, all_plot)

    #step2: 计算 累计分布函数 cdf
    cdf = CDF(count, all_plot)

    #step3: compensation transfer function: f(I) = c(I)
    def f(I):
        return alpha*cdf[I]+(1-alpha)*I/255

    img = f(img)
    
    #画图
    if plot or all_plot:
        plt.figure()
        plt.imshow(img, cmap="gray")
        plt.title("image by using histogram equalization, alpha is {}".format(alpha))
        #处理后的图的count和cdf
        count = Count(img, all_plot)
        cdf = CDF(count, all_plot)
    return img

# 实现 Question3.1(iii)
# punch直方图均衡化 由于不懂题目所说5%是纯黑纯白各5%还是加起来5%这里变为一个可选参数
# input: gray image 可选纯白比例:top  纯黑比例:bottom
# 可以选择是否画所有图 plot 只画结果图， all_plot 画所有图
def Punch(img, top=0.05, bottom=0.05, plot=False, all_plot=False):
    #step1: 统计数目
    count = Count(img, all_plot)

    #step2: 计算 累计分布函数 cdf
    cdf = CDF(count, all_plot)

    #step3: compensation transfer function f(I)
    def f(I):
        img1 = (cdf[I]-bottom)/(1-top-bottom)
        img2 = np.where(img1<0, 0, img1)
        return np.where(img2>1, 1, img2)
    
    img = f(img)
    
    #画图
    if plot or all_plot:
        plt.figure()
        plt.imshow(img, cmap="gray")
        plt.title("image by using punch histogram equalization\n pure white is {}%, pure black is {}%".format(
                                                                                        top*100, bottom*100))
        #处理后的图的count和cdf
        count = Count(img, all_plot)
        cdf = CDF(count, all_plot)
    return img

# 实现 Question3.1(iv)
# 一开始没搞懂要干嘛，reference: https://zhuanlan.zhihu.com/p/44918476
# input: img, lamda  output: limit_local_gain处理后的img
# 可以选择是否画所有图 plot 只画结果图， all_plot 画所有图
def limit_local_gain(img, lamda, plot=False, all_plot=False):
    #统计数目
    count = Count(img, all_plot)

    #limit local gain
    over = count - lamda
    over = np.where(over<0,0,over)
    over = np.sum(over)
    count = np.where(count>=lamda, lamda, count)
    count += over//256
    
    if all_plot:
        plt.figure()
        plt.plot(count)
        plt.title("histogram: count after limit")
    
    #step2: 计算 累计分布函数 cdf
    cdf = CDF(count, all_plot)
    
    #step3: compensation transfer function: f(I) = c(I)
    def f(I):
        return cdf[I]

    img = f(img)
    
    #画图
    if plot or all_plot:
        plt.figure()
        plt.imshow(img, cmap="gray")
        plt.title("image by using limit_local_gain histogram equalization\n lambda is {}".format(lamda))
        #处理后的图的count和cdf
        count = Count(img, all_plot)
        cdf = CDF(count, all_plot)  
        
    return img
# # 实现 Question3.1(v)
# reference: https://wenku.baidu.com/view/111082efbceb19e8b9f6ba35.html
def gray2color(img_init,img_processed,plot=False):
    img_gray = img_init.dot(np.array([0.299, 0.587, 0.114]))
    
    #确定 image type: float or int?
    if type(img_processed[0,0]) in type_int:
        img_processed = img_processed/255

    new_RGB = (img_processed / img_gray).reshape(img_processed.shape+(1,)) * img_init
    new_RGB = np.where(new_RGB>1,1,new_RGB)
    
    if plot:
        plt.figure()
        plt.imshow(new_RGB)
    return new_RGB


# In[4]:


#3.2 直方图均衡化
def histogram_equalization_f(img, alpha=1):
    #step1: 统计数目
    count = Count(img)

    #step2: 计算 累计分布函数 cdf
    cdf = CDF(count)

    #step3: compensation transfer function: f(I) = c(I)
    def f(I):
        return alpha*cdf[I]+(1-alpha)*I/255
    
    lookup = np.arange(256)
    
    return f(lookup)

# 对每个block做直方图均衡化
def block(img0,M,N):
    lookup = []
    for i in range(0,img0.shape[0],M):
        row = []
        for j in range(0,img0.shape[1],N):
            i_up = min(i+M,img0.shape[0])
            j_up = min(j+N,img0.shape[1])
            row.append(histogram_equalization_f(img0[i:i_up,j:j_up]))
        lookup.append(row)
    return np.array(lookup)

#对lookup padding是为了不用处理边界条件
def padding(lookup):
    lookup1 = np.empty(lookup.shape+np.array([2,2,0]))
    lookup1[1:-1,1:-1,:] = lookup
    lookup1[0,1:-1,:] = lookup[0,:,:]
    lookup1[-1,1:-1,:] = lookup[-1,:,:]
    lookup1[1:-1,0,:] = lookup[:,0,:]
    lookup1[1:-1,-1,:] = lookup[:,-1,:]
    lookup1[0,0,:] = lookup[0,0,:]
    lookup1[0,-1,:] = lookup[0,-1,:]
    lookup1[-1,0,:] = lookup[-1,0,:]
    lookup1[-1,-1,:] = lookup[-1,-1,:]
    return lookup1

# 对每个pixel计算 双线性插值
def pixel(img0,i,j,lookup,M,N):
    def f(mat,u):
        return np.array([[mat[0,0][u],mat[0,1][u]],
                        [mat[1,0][u],mat[1,1][u]]]) 
    mat = lookup[i//M:i//M+2,j//N:j//N+2]
    x = (j - j//N*N - N/2)/N
    y = (i - i//M*M - M/2)/M
    return np.array([[1-x,x]]).dot(
        f(mat,img0[i,j])).dot(
        np.array([1-y,y]).T)

# 局部直方图均衡化主函数
def local_histogram_equalization(img_path,M,N):
    img0 = cvt_color(img_path)

    lookup = block(img0,M,N)
    a,b = img0.shape
    img1 = np.pad(img0,((M//2,(M-a%M)+M//2),(N//2,(N-b%N)+N//2)),"constant")
    lookup1 = padding(lookup)

    img = np.zeros((a,b))
    for i in range(a):
        for j in range(b):
            img[i,j] = pixel(img1,i+M//2,j+N//2,lookup1,M,N)
    return img

