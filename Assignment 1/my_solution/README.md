# 第二题代码使用说明：
请助教调用 Q2.py 中的 myImageFilter 函数，参数与要求相同。
请助教在 main 文档准备好 img1（支持路径 str 类型，和矩阵类型 np.array，建议输入矩阵类型）和 h （矩阵类型）。

示例：
```python
import Q2
img1 = Q2.myImageFilter(img0,h)
```

另有一些注释附在代码旁边。

# 第三题代码说明：
## 实现 Question3.1(i)
- 函数原型 cvt_color(img_path, gray=True, plot=False)
- #这个函数输入路径，可以选择得到灰度还是RGB彩色图，默认得到对应的灰度图像
- 使用示例：
```python
import Q3
img_path = "Lenna.png"
img = Q3.cvt_color(img_path)
```
## 实现 Question3.1(ii)
- 函数原型 histogram_equalization(img, alpha=1, plot=False, all_plot=False)
- input: gray image  output: 操作后的image
- 可以选择是否画所有图 plot 只画结果图， all_plot 画所有图
- 使用示例：
```python
img1 = Q3.histogram_equalization(img)
```

## 实现 Question3.1(iii)
- 函数原型 Punch(img, top=0.05, bottom=0.05, plot=False, all_plot=False)
- punch直方图均衡化 由于不懂题目所说5%是纯黑纯白各5%还是加起来5%这里变为一个可选参数
- input: gray image 可选纯白比例:top  纯黑比例:bottom
- 可以选择是否画所有图 plot 只画结果图， all_plot 画所有图
- 使用示例：
```python
img1 = Q3.Punch(img, 0.05, 0.05)
```

## 实现 Question3.1(iv)
- 函数原型 limit_local_gain(img, lamda, plot=False, all_plot=False)
- 一开始没搞懂要干嘛，reference: https://zhuanlan.zhihu.com/p/44918476
- input: img, lamda  output: limit_local_gain处理后的img
- 可以选择是否画所有图 plot 只画结果图， all_plot 画所有图
- 使用示例：
```python
img1 = Q3.limit_local_gain(img,1000)
```

## 实现 Question3.1(v)
- 函数原型 def gray2color(img_init,img_processed,plot=False)
- input: img_init(最开始的彩色图片), img_processed(经过直方图均衡化等处理后的灰色图片)
- output: 重建的彩色图片
- 可以选择是否把彩色图片画出来
- reference: https://wenku.baidu.com/view/111082efbceb19e8b9f6ba35.html
- 使用示例：
```python
img_color = Q3.cvt_color(img_path, gray=False)
img2 = Q3.gray2color(img_color,img1)
```

## 实现 Question3.2
- 函数原型 local_histogram_equalization(img_path,M,N):
- input: img_path, block 大小： M*N
- output: 局部直方图均衡化的结果
- 附带子函数介绍：histogram_equalization_f(img, alpha=1): 负责返回每个 block 的映射函数；block(img0,M,N): 对每个block做直方图均衡化；padding(lookup): 对lookup padding是为了不用处理边界条件；pixel(img0,i,j,lookup,M,N): 使用双线性插值对每个pixel计算像素值
- 使用示例：
```python
img3 = Q3.local_histogram_equalization(img_path,16,16)
```



















