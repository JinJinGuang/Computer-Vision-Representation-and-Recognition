# README

## 代码使用（复现）方法： ##

运行 code.py 档，依次回答问题即可：

```flow
st=>start: Start
op=>operation: Your Operation
cond=>condition: Yes or No?
cond1=>condition: Reproduce? If no, use your own images.
cond2=>condition: Reproduce 1, 2 or 3: yes is 1, no is 2 or 3
img1=>operation: input your img1's path
img2=>operation: input your img2's path
correspondences1=>operation: correspondences choose your hand, my hand or SIFT to get pts
correspondences2=>operation: homography_matrix by Least squares method or RANSAC and Warp
correspondences=>operation: correspondences, Get homography_matrix and Warp
e=>end

st->op->cond
cond(yes)->e
cond(no)->op

st->cond1
cond1(no)->img1->img2->correspondences1->correspondences2->e
cond1(yes)->cond2
cond2(yes)->correspondences1
cond2(no)->correspondences->e
```

## 函数解释： ##

### 1 Image Mosaics  ###

#### Question 1: correspondences(src, dst) ####

- 传入：src（原图像）、dst（要变换到的目的图像）
- 返回：对应两个图像相同点（需要标定操作）

#### Question 2: homography_matrix(pts) ####

- 根据 Q1 获得的 pts 计算并返回 H 矩阵
- homography_matrix_srcdst(src,dst) 是其传入 src（原图像的坐标点集合），与 dst（对应另一个图像的点）

#### Question 3: Warp_img(img, H, fill_value=0, dtype=np.uint8) ####

- 传入：要变换的原图像 img，H 矩阵（Q2 计算得到）、fill_value：插值时超出的补什么值、dtype：最后返回值类型。
- 返回：变换后的图像

### 2 Automatic Image Mosaics ###

#### Question 1: SIFT(img1, img2, N) ####

- 传入：img1（原图像）、img2（要变换到的目的图像）、N：保留多少对应点
- 返回：对应两个图像相同点（自动标定）

#### Question 2: RANSAC ####

- 使用 H, status = cv2.findHomography(src.reshape(-1,1,2), dst.reshape(-1,1,2), cv2.RANSAC, ransacReprojThreshold)



# Reference： #

[图像取点 ginput](https://www.cnblogs.com/darkknightzh/p/6182474.html)

[图像取点 ginput 官方文档](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.ginput.html)

[Spyder中单独弹出窗口显示figure以及解决动态figure显示的设置](https://blog.csdn.net/yangzijiang666/article/details/79961873)

[画多图(figure)](https://blog.csdn.net/You_are_my_dream/article/details/53440384)f

[GitHub](https://github.com/XS2929/Computer-Vision/tree/master/A3)

[Basic concepts of the homography explained with code](https://docs.opencv.org/3.4.1/d9/dab/tutorial_homography.html)

[Lecture 16: Planar Homographies](http://www.cse.psu.edu/~rtc12/CSE486/lecture16.pdf)

[numpy 往array里添加一个元素](https://www.cnblogs.com/cymwill/p/8118135.html)

[线性方程组求解——基于MTALAB/Octave,Numpy,Sympy和Maxima](https://blog.csdn.net/ouening/article/details/54692458)

[matlab - 将 Matlab interp2移植到 scipy interp2d](https://ask.helplib.com/matlab/post_10465656)

[interp2(X, Y, Z, XI, YI) from Matlab to Python](https://stackoverflow.com/questions/11468367/interp2x-y-z-xi-yi-from-matlab-to-python)

[scipy.interpolate.interp2d](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp2d.html)

[MATLAB interp2](https://ww2.mathworks.cn/help/matlab/ref/interp2.html)

[Homography Examples using OpenCV ( Python / C ++ )](https://www.learnopencv.com/homography-examples-using-opencv-python-c/)