import numpy as np

#padding 保持最后输出 shape same 
#padding 方式：和最近的边界上的pixel相同
def padding(img0,h):
    m = (h.shape[0]-1)//2
    n = (h.shape[1]-1)//2
    img = np.pad(img0,((m,m),(n,n)),"edge")
    return img

#将矩阵切片 方便矢量化
def slice_mat(a, sub_shape):
    #reference: https://stackoverflow.com/questions/43086557/convolve2d-just-by-using-numpy
    view_shape = tuple(np.subtract(a.shape, sub_shape) + 1) + sub_shape
    strides = a.strides + a.strides
    sub_matrices = np.lib.stride_tricks.as_strided(a,view_shape,strides)
    return sub_matrices

#读取图片
def read_image(path):
    import cv2
    img0 = cv2.imread(img_path)
    img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    return img0

def myImageFilter(img0, h, reverse=True):
    #如果输入path读取图片 建议直接输入矩阵
    if(type(img0) is str):
        img0 = read_image(img0)
    #需要翻转吗？做卷积还是相关操作 默认卷积操作
    if(reverse):
        h = h[::-1,::-1]
        
    img1 = padding(img0, h)
    sub_matrices = slice_mat(img1, h.shape)
    a,b,c,d = sub_matrices.shape
    sub_matrices = sub_matrices.reshape(a,b,c*d)
    h = h.reshape(-1)
    return np.dot(sub_matrices,h)
