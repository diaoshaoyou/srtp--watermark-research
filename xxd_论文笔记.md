# On the Effectiveness of Visible Watermarks

### 前景蒙版（alpha matte）

也称前景透明度或透明度蒙版，是前背景分离的结果，是一个灰度图，每一个像素点的灰度值表示原始图像每个像素属于前景物体的程度，白色代表某一个像素确定属于前景，黑色代表某一个像素确定属于背景。

I＝α×F＋(1－α)×B

I: 图像	F: 前景图像	B: 背景图像	alpha matte α

如果再具体一点，针对每个像素，背景颜色为B＝[RB,GB,BB]，前景对象颜色为F＝[αRF, αGF, αBF]，于是matting方程为I＝F＋(1－α)×B。通常在应用中都是求这个方程的解，以获得图像的最后分割结果。

R=α×RF＋(1－α)×RB
G=α×GF＋(1－α)×GB

B= α×BF＋(1－α)×BB

解法：灰度图。令RF＝GF＝BF。



### 高斯滤波

【转自：https://blog.csdn.net/lvquanye9483/article/details/81592574 / https://blog.csdn.net/Ibelievesunshine/article/details/104881204】

#### 含义

高斯滤波就是对整幅图像进行加权平均的过程，每一个像素点的值，都由其本身和邻域内的其他像素值经过加权平均后得到。

#### 高斯函数

<img src="https://upload-images.jianshu.io/upload_images/14512145-cb79bc3d41cc37fd.png?imageMogr2/auto-orient/strip|imageView2/2/w/708/format/webp" alt="img" style="zoom:80%;" />

σ描述正态分布资料数据分布的离散程度，σ越大，数据分布越分散，σ越小，数据分布越集中。也称为是正态分布的形状参数，σ越大，曲线越扁平，反之，σ越小，曲线越瘦高。

#### 高斯核

理论上，高斯分布在所有定义域上都有非负值，这就需要一个无限大的卷积核。实际上，仅需要取均值周围3倍标准差内的值，以外部份直接去掉即可。
**高斯滤波的重要两步就是先找到高斯模板然后再进行卷积。**



对二维高斯分布，不必纠结于系数<img src="https://img-blog.csdnimg.cn/20200315170249883.png" alt="img" style="zoom:50%;" />，因为它只是一个常数！并不会影响互相之间的比例关系，而且最终都要进行归一化，所以在实际计算时我们忽略它而只计算后半部分 <img src="https://img-blog.csdnimg.cn/20200315170316782.png" alt="img" style="zoom:50%;" />

其中(x,y)为掩膜内任一点的坐标，(ux,uy)为掩膜内中心点的坐标，在图像处理中可认为是整数；σ是标准差。

​        例如：要产生一个3×3的高斯滤波器模板，以模板的中心位置为坐标原点进行取样。（x轴水平向右，y轴竖直向下）  

​        <img src="https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNzIyMTQ5OS0yMmEyNjRmM2VmOTVlMzI5LnBuZw?x-oss-process=image/format,png" alt="img" style="zoom:50%;" />

​        （模板在各个位置的坐标，如上图所示↑）

​        这样，将各个位置的坐标代入到高斯函数中，得到的值就是滤波器的系数。

​        如果窗口模板的大小为 (2k+1)×(2k+1)，则：

​        <img src="https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNzIyMTQ5OS01OGM1OTU0MmM5OTU0OTg0LnBuZw?x-oss-process=image/format,png" alt="img" style="zoom: 80%;" />

​        （窗口模板中各个元素的计算公式↑）

​        这样计算出来的模板有两种形式：小数和整数

​        小数形式的模板，就是直接计算得到的值，没有经过任何的处理；

​        整数形式的模板，需要进行归一化处理，将模板左上角的值归一化为1。使用整数的模板时，需要在模板的前面加一个系数，系数为模板中元素和的倒数。

​        例如，标准差![\sigma](https://private.codecogs.com/gif.latex?%5Csigma) =1.3 的 3*3 的整数形式的高斯滤波器如下：

​        <img src="https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNzIyMTQ5OS0xNTI1YWFhMWY0YmJhOWM3LnBuZw?x-oss-process=image/format,png" alt="img" style="zoom:50%;" />

​        （标准差 ![\sigma](https://private.codecogs.com/gif.latex?%5Csigma)=1.3 的8近邻高斯滤波器如图 ↑）

**σ的意义及选取**

​        通过上述的实现过程，不难发现，高斯滤波器模板的生成最重要的参数就是高斯分布的标准差σ。标准差代表着数据的离散程度，如果σ较小，那么生成的模板的中心系数较大，而周围的系数较小，这样对图像的平滑效果就不是很明显；反之，σ较大，则生成的模板的各个系数相差就不是很大，比较类似均值模板，对图像的平滑效果比较明显。



**举个例子**

假定中心点的坐标是（0,0），那么取距离它最近的8个点坐标，为了计算，需要设定σ的值。假定σ=1.5，则模糊半径为1的高斯模板就算如下

<img src="https:////upload-images.jianshu.io/upload_images/14512145-04d41990169b094a.png?imageMogr2/auto-orient/strip|imageView2/2/w/800/format/webp" alt="img" style="zoom: 67%;" />


这个时候我们我们还要确保这九个点加起来为1（这个是高斯模板的特性），这9个点的权重总和等于0.4787147，因此上面9个值还要分别除以0.4787147，得到最终的高斯模板。

<img src="https:////upload-images.jianshu.io/upload_images/14512145-049ada57d888bf79.png?imageMogr2/auto-orient/strip|imageView2/2/w/409/format/webp" alt="img" style="zoom:67%;" />



#### 高斯滤波计算

举个例子：

假设现有9个像素点，灰度值（0-255）的高斯滤波计算如下：

<img src="https:////upload-images.jianshu.io/upload_images/14512145-530497b10b412a95.png?imageMogr2/auto-orient/strip|imageView2/2/w/1075/format/webp" alt="img" style="zoom: 67%;" />

参考来源：（https://blog.csdn.net/nima1994/article/details/79776802）

将这9个值加起来，就是中心点的高斯滤波的值。
对所有点重复这个过程，就得到了高斯模糊后的图像。

由于图像的长宽可能不是滤波器大小的整数倍，因此我们需要在图像的边缘补0，这种方法叫做 zero padding 。



#### opencv函数实现高斯滤波

```python
import cv2
 
img=cv2.imread('../paojie.jpg')
 
#(3, 3)表示高斯滤波器的长和宽都为3，1.3表示滤波器的标准差
 
out=cv2.GaussianBlur(img,(3,3),1.3)
 
cv2.imwrite('out.jpg',out)
 
cv2.imshow('result',out)
 
cv2.waitKey(0)
 
cv2.destroyAllWindows()
```



### Canny边缘检测

【转自：https://www.cnblogs.com/mmmmc/p/10524640.html】

#### **评价标准**

Canny提出了一个对于边缘检测算法的评价标准，包括：

（1）以低的错误率检测边缘，也即意味着需要尽可能准确的捕获图像中尽可能多的边缘。

（2）检测到的边缘应精确定位在真实边缘的中心。

（3）图像中给定的边缘应只被标记一次，并且在可能的情况下，图像的噪声不应产生假的边缘。

　　简单来说就是，检测算法要做到：**边缘要全，位置要准，抵抗噪声的能力要强。** 

#### **实现步骤**

##### step1：高斯平滑滤波

滤波是为了去除噪声，选用高斯滤波也是因为在众多噪声滤波器中，高斯表现最好。一个大小为(2k+1)x(2k+1)的高斯滤波器核（核一般都是奇数尺寸的）的生成方程式由下式给出：

<img src="https://images2017.cnblogs.com/blog/1213858/201708/1213858-20170805214317647-24872875.png" alt="img" style="zoom: 67%;" />

　　下面是一个sigma = 1.4，尺寸为3x3的高斯卷积核的例子，注意矩阵求和值为1（归一化）：

<img src="https://images2017.cnblogs.com/blog/1213858/201708/1213858-20170805214458225-1180132980.png" alt="img" style="zoom:75%;" />

高斯卷积核的大小将影响Canny检测器的性能。尺寸越大，去噪能力越强，因此噪声越少，但图片越模糊，canny检测算法抗噪声能力越强，但模糊的副作用也会导致定位精度不高，一般情况下，推荐尺寸5x5，3x3也行。



##### step2： 计算梯度强度和方向

边缘的最重要的特征是灰度值剧烈变化，如果把灰度值看成二元函数值，那么灰度值的变化可以用二元函数的”导数“（或者称为梯度）来描述。由于图像是离散数据，导数可以用差分值来表示，差分在实际工程中就是灰度差，说人话就是两个像素的差值。一个像素点有8邻域，那么分上下左右斜对角，因此Canny算法使用四个算子来检测图像中的水平、垂直和对角边缘。算子是以图像卷积的形式来计算梯度，比如Roberts，Prewitt，Sobel等，这里选用Sobel算子来计算二维图像在x轴和y轴的差分值（这些数字的由来？），将下面两个模板与原图进行卷积，得出**x和y轴的差分值图**，最后计算该点的梯度G和方向θ

<img src="https://img2018.cnblogs.com/blog/1627009/201903/1627009-20190313190846033-1566639586.png" alt="img" style="zoom: 67%;" />

<img src="https://img2018.cnblogs.com/blog/1627009/201903/1627009-20190313190952686-2066246568.png" alt="img" style="zoom:45%;" />

首先了解opencv的二维滤波函数：dst=cv.filter2D(src, ddepth, kernel[, dst[, anchor[, delta[, borderType]]]]) 

**dst**：　　输出图片

**src**：　　输入图片

**ddepth**： 输出图片的深度, 详见 [combinations](https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#filter_depths),如果填-1，那么就跟与输入图片的格式相同。

**kernel**：   单通道、浮点精度的卷积核。

以下是默认参数：

**anchor**：内核的基准点(anchor)，其默认值为(-1,-1)表示位于kernel的中心位置。基准点即kernel中与进行处理的像素点重合的点。举个例子就是在上面的step1中，e=H*A得到的e是放在原像素的3*3的哪一个位置，一般来说都是放在中间位置，设置成默认值就好。

**delta** ：在储存目标图像前可选的添加到像素的值，默认值为0。

**borderType**：像素向外逼近的方法，默认值是BORDER_DEFAULT,即对全部边界进行计算。

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt


img=cv2.imread("images/luxun.png",cv2.IMREAD_GRAYSCALE)  # 读入图片
sobel_x = np.array([[-1, 0, 1],[-2,0,+2],[-1, 0, 1]])    # sobel的x方向算子
sobel_y = np.array([[1, 2, 1],[0,0,0],[-1, -2, -1]])     # sobel的x方向算子
sobel_x=cv2.flip(sobel_x,-1)          # cv2.filter2D()计算的是相关，真正的卷积需要翻转，再进行相关计算。
sobel_x=cv2.flip(sobel_y,-1)
# cv2.flip()第二个参数：等于0：沿着x轴反转。大于0：沿着y轴反转。小于零：沿着x轴，y轴同时反转

# 卷积 opencv是用滤波器函数实现的
img_x=cv2.filter2D(img,-1, sobel_x)
img_y=cv2.filter2D(img,-1, sobel_y)
# 画图 plt不支持中文，但是可以通过以下方法设置修复
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.subplot(221), plt.imshow(img_x, 'gray'),plt.title('sobel_x')
plt.subplot(222), plt.imshow(img_y, 'gray'),plt.title('sobel_y')
plt.subplot(223), plt.imshow(img_y+img_x, 'gray'),plt.title('sobel')
plt.subplot(224), plt.imshow(img, 'gray'),plt.title('原图')
plt.show()
```

<img src="https://img2018.cnblogs.com/blog/1627009/201903/1627009-20190315200104993-2032123366.png" alt="img" style="zoom: 67%;" />

需要注意一点：在图像处理领域，卷积运算的定义是先将核关于x轴和y轴反转，然在做相关运算。然而工程实践中往往跳过反转，用相关运算代替卷积（比如opencv）。如果你需要严格的卷积运算，应该注意原函数的具体实现方式。sobel算子天生关于中心对称，所以反转与否并不影响结果。

作者自己写的卷积函数：

```python
def conv2d(src,kernel):  # 输入必须为方形卷积核
    # 本函数仍然是相关运算，没有反转。如果非要严格的卷积运算，把下面一行代码的注释取消。
    #kernel=cv2.flip(kernel,-1)
    [rows,cols] = kernel.shape
    border=rows//2                                  # 向下取整 获得卷积核边长
    [rows,cols]=src.shape
    dst = np.zeros(src.shape)                       # 采用零填充再卷积，卷积结果不会变小。
    # print("图像长：",rows,"宽：",cols,"核边界",border)
    # print(border,rows-border,border,cols-border)
    temp=[]
    for i in range(border,rows-border):
        for j in range(border,cols-border):
            temp=src[i-border:i+border+1,j-border:j+border+1]  # 从图像获取与核匹配的图像
            # 切片语法：索引位置包括开头但不包括结尾 [start: end: step]
            dst[i][j]=(kernel*temp).sum()  # 计算卷积
    return dst
```

<img src="https://img2018.cnblogs.com/blog/1627009/201903/1627009-20190320223350715-683812899.png" alt="img" style="zoom: 50%;" />

sobel算子计算的边缘很粗很亮，比较明显，但是不够精确，我们的目标是精确到一个像素宽，至于梯度相位就很难看出什么特征，并且梯度相位实际上是为了下一步打基础的。下面附上代码：



##### step3:非极大值抑制

sobel算子检测出来的边缘太粗了，我们需要**抑制那些梯度不够大的像素点，只保留最大的梯度，从而达到瘦边的目的**。这些梯度不够大的像素点很可能是某一条边缘的过渡点。按照高数上二位函数的极大值的定义，即对点（x0，y0）的某个邻域内所有（x，y）都有f（x，y）≤(f（x0，y0），则称f在（x0，y0）具有一个极大值，极大值为f（x0，y0）。简单方案是判断一个像素点的8邻域与中心像素谁更大，但这很容易筛选出噪声，因此我们需要用梯度和梯度方向来辅助确定。

如下图所示，中心像素C的梯度方向是蓝色直线，那么只需比较中心点C与dTmp1和dTmp2的大小即可。由于这两个点的像素不知道，假设像素变化是连续的，就可以用g1、g2和g3、g4进行线性插值估计。设g1的幅值M(g1)，g2的幅值M(g2),则M(dtmp1)=w*M(g2)+(1-w)*M(g1)  ，其w=distance(dtmp1,g2)/distance(g1,g2)  。也就是利用g1和g2到dTmp1的距离作为权重，来估计dTmp1的值。w在程序中可以表示为tan（θ）来表示，具体又分为四种情况（下面右图）讨论。

<img src="https://img2018.cnblogs.com/blog/1627009/201903/1627009-20190320175657758-344372967.png" alt="img" style="zoom:67%;" />

如下图，经过非极大值抑制可以很明显的看出去除了很多点，边缘也变得很细。在程序实现中，要注意opencv的默认坐标系是从左到右为x轴，从上到下是y轴，原点位于左上方，计算g1、g2、g3、g4的位置的时候，一定要小心。**经过非极大值抑制可以看出来图片的边缘明显变细**，很多看起来黑色的部分其实有值的，只是因为值太小了看不清楚，而这些黑色的部分可能是噪声或者其他原因造成的局部极大值，下一步我们就要用双阈值来限定出强边缘和弱边缘，尽可能的减少噪声的检出。

<img src="https://img2018.cnblogs.com/blog/1627009/201903/1627009-20190320180806448-1983710981.png" alt="img" style="zoom: 67%;" />

```python
# step3:非极大值抑制
anchor=np.where(G!=0)  # 获取非零梯度的位置
Mask=np.zeros(img.shape)

for i in range(len(anchor[0])):
    x=anchor[0][i]
    y=anchor[1][i]
    center_point=G[x,y]
    current_theta=theta[x,y]
    dTmp1=0
    dTmp2=0
    W=0
    if current_theta>=0 and current_theta<45:
        #        g1         第一种情况
        # g4  C  g2
        # g3
        g1 = G[x + 1, y - 1]
        g2 = G[x + 1, y]
        g3 = G[x - 1, y + 1]
        g4 = G[x - 1, y]
        W=abs(np.tan(current_theta*np.pi/180))  # tan0-45范围为0-1
        dTmp1= W*g1+(1-W)*g2
        dTmp2= W*g3+(1-W)*g4

    elif current_theta>=45 and current_theta<90:
        #     g2 g1         第二种情况
        #     C
        # g3  g4

        g1 = G[x + 1, y - 1]
        g2 = G[x, y - 1]
        g3 = G[x - 1, y + 1]
        g4 = G[x, y + 1]
        W = abs(np.tan((current_theta-90) * np.pi / 180))
        dTmp1= W*g1+(1-W)*g2
        dTmp2= W*g3+(1-W)*g4

    elif current_theta>=-90 and current_theta<-45:
        # g1  g2            第三种情况
        #     C
        #     g4 g3
        g1 = G[x - 1, y - 1]
        g2 = G[x, y - 1]
        g3 = G[x + 1, y + 1]
        g4 = G[x, y + 1]
        W = abs(np.tan((current_theta-90) * np.pi / 180))
        dTmp1= W*g1+(1-W)*g2
        dTmp2= W*g3+(1-W)*g4

    elif current_theta>=-45 and current_theta<0:
        # g3                第四种情况
        # g4  C  g2
        #        g1
        g1 = G[x + 1, y + 1]
        g2 = G[x + 1, y]
        g3 = G[x - 1, y - 1]
        g4 = G[x - 1, y]
        W = abs(np.tan(current_theta * np.pi / 180))
        dTmp1= W*g1+(1-W)*g2
        dTmp2= W*g3+(1-W)*g4

    if dTmp1<center_point and dTmp2<center_point:   # 记录极大值结果
            Mask[x,y]=center_point
#Mask=(Mask-Mask.min())/(Mask.max()-Mask.min())*256 #归一化
plt.imshow(Mask,'gray'),plt.title('Mask')
plt.show()
```

##### **step4：用双阈值算法检测和连接边缘**

双阈值法非常简单，我们假设两类边缘：经过非极大值抑制之后的边缘点中，**梯度值超过T1的称为强边缘，梯度值小于T1大于T2的称为弱边缘，梯度小于T2的不是边缘**。可以肯定的是，强边缘必然是边缘点，因此必须将T1设置的足够高，以要求像素点的梯度值足够大（变化足够剧烈），而弱边缘可能是边缘，也可能是噪声，如何判断呢？**当弱边缘的周围8邻域有强边缘点存在时，就将该弱边缘点变成强边缘点**，以此来实现对强边缘的补充。实际中人们发现T1:T2=2:1的比例效果比较好，其中T1可以人为指定，也可以设计算法来自适应的指定，比如定义梯度直方图的前30%的分界线为T1，我实现的是人为指定阈值。检查8邻域的方法叫边缘滞后跟踪，连接边缘的办法还有区域生长法等等。强边缘、弱边缘、综合效果、和opencv的canny函数对比如下：

![img](https://img2018.cnblogs.com/blog/1627009/201903/1627009-20190320220229510-1262182428.png)

作者总结：

实现结果还是很打击的，我检测到的边缘过于断续，没有opencv实现的效果好。查了一下opencv的源码，这里猜测两个可能的原因：源码里梯度的方向被近似到四个角度之一 （0，45，90，135），但我用线性插值的的结果是梯度方向更精确，而过于精确-->过于严格-->容易受到噪声干扰，所以在非极大值抑制这之后，我比opencv少了更多的点，最终导致了边缘不够连续；第二个原因可能是边缘连接算法效果不够好，把图象放大来看，我产生的边缘倾向于对角线上连接，而opencv的边缘倾向于折线连接，因此opencv的边缘更完整连续，而我的边缘更细，更容易断续。



### 论文中的公式

<img src="C:\Users\Doris Cullen\AppData\Roaming\Typora\typora-user-images\1596014303980.png" alt="1596014303980" style="zoom:80%;" />

<img src="C:\Users\Doris Cullen\AppData\Roaming\Typora\typora-user-images\1596014317679.png" alt="1596014317679" style="zoom:80%;" />

J: watermarked image	W: watermark	I: natural image

<img src="C:\Users\Doris Cullen\AppData\Roaming\Typora\typora-user-images\1596014440143.png" alt="1596014440143" style="zoom:80%;" />

<img src="C:\Users\Doris Cullen\AppData\Roaming\Typora\typora-user-images\1596014495712.png" alt="1596014495712" style="zoom: 67%;" />: 图像集

<img src="C:\Users\Doris Cullen\AppData\Roaming\Typora\typora-user-images\1596015590689.png" alt="1596015590689" style="zoom: 67%;" /> 



### 3.1. Initial Watermark Estimation & Detection

#### I. Estimating the Matted Watermark

<img src="C:\Users\Doris Cullen\AppData\Roaming\Typora\typora-user-images\1596031648615.png" alt="1596031648615" style="zoom: 80%;" />

<img src="C:\Users\Doris Cullen\AppData\Roaming\Typora\typora-user-images\1596031718334.png" alt="1596031718334" style="zoom:80%;" />

use Canny with 0.4 threshold

use Poisson reconstruction to obtain the initial matted watermark <img src="C:\Users\Doris Cullen\AppData\Roaming\Typora\typora-user-images\1596031816605.png" alt="1596031816605" style="zoom:50%;" />



#### II. Watermark Detection

use Chamfer Distance to detect the watermark in each of the images



### 3.2. Multi-Image Matting and Reconstruction

arg min F(x,y) ：当F(x,y)取得最小值时，变量x,y的取值

<img src="C:\Users\Doris Cullen\AppData\Roaming\Typora\typora-user-images\1596032915279.png" alt="1596032915279" style="zoom:80%;" /> 

<img src="C:\Users\Doris Cullen\AppData\Roaming\Typora\typora-user-images\1596033025804.png" alt="1596033025804" style="zoom:80%;" /> 				惩罚项

<img src="C:\Users\Doris Cullen\AppData\Roaming\Typora\typora-user-images\1596033035526.png" alt="1596033035526" style="zoom:80%;" /> 										使得水印图像平滑

<img src="C:\Users\Doris Cullen\AppData\Roaming\Typora\typora-user-images\1596033173981.png" alt="1596033173981" style="zoom:80%;" /> 									保持真实性

**Optimization**

<img src="C:\Users\Doris Cullen\AppData\Roaming\Typora\typora-user-images\1596033409943.png" alt="1596033409943" style="zoom:80%;" />

#### I. Image–Watermark Decomposition

<img src="C:\Users\Doris Cullen\AppData\Roaming\Typora\typora-user-images\1596033681758.png" alt="1596033681758" style="zoom: 80%;" />

【！迭代加权最小二乘法】



#### II. Watermark Update

估计全局W -> 使得Eaux最小 -> 求{Wk}平均值

<img src="C:\Users\Doris Cullen\AppData\Roaming\Typora\typora-user-images\1596033828515.png" alt="1596033828515" style="zoom:80%;" />



#### III. Matte Update

<img src="C:\Users\Doris Cullen\AppData\Roaming\Typora\typora-user-images\1596033936877.png" alt="1596033936877" style="zoom:80%;" />



#### Matte and Blend Factor Initialization

$$
α = c · αn
$$

【！自适应阈值】初始化matte

【最小二乘法】初始化c



去水印：利用W和a