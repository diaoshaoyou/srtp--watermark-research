#包含功能：estimate watermark + detect watermark region,在main中调用顺序：1
import sys, os
import cv2
import numpy as np
import warnings
from matplotlib import pyplot as plt
import math
import numpy
import scipy, scipy.fftpack

# Variables
KERNEL_SIZE = 3

def estimate_watermark(foldername):#输出图片梯度和梯度中位数
	"""
	Given a folder, estimate the watermark (grad(W) = median(grad(J)))
	Also, give the list of gradients, so that further processing can be done on it
	"""
	if not os.path.exists(foldername):
		warnings.warn("Folder does not exist.", UserWarning)
		return None

	images = []#image用来存一系列图片J
	for r, dirs, files in os.walk(foldername):#r是root，根目录
		# Get all the images
		for file in files:
			img = cv2.imread(os.sep.join([r, file]))
			if img is not None:
				images.append(img)
			else:
				print("%s not found."%(file))

	# Compute gradients
	print("Computing gradients.")
	gradx = map(lambda x: cv2.Sobel(x, cv2.CV_64F, 1, 0, ksize=KERNEL_SIZE), images)
	grady = map(lambda x: cv2.Sobel(x, cv2.CV_64F, 0, 1, ksize=KERNEL_SIZE), images)

	# Compute median of grads
	print("Computing median gradients.")
	Wm_x = np.median(np.array(gradx), axis=0) 					
	Wm_y = np.median(np.array(grady), axis=0)

	return (Wm_x, Wm_y, gradx, grady)


def PlotImage(image):
	""" 
	PlotImage: Give a normalized image matrix which can be used with implot, etc.
	Maps to [0, 1]
	"""
	im = image.astype(float)
	return (im - np.min(im))/(np.max(im) - np.min(im))


def poisson_reconstruct2(gradx, grady, boundarysrc):
	# Thanks to Dr. Ramesh Raskar for providing the original matlab code from which this is derived
	# Dr. Raskar's version is available here: http://web.media.mit.edu/~raskar/photo/code.pdf

	# Laplacian
	gyy = grady[1:,:-1] - grady[:-1,:-1]
	gxx = gradx[:-1,1:] - gradx[:-1,:-1]
	f = numpy.zeros(boundarysrc.shape)
	f[:-1,1:] += gxx
	f[1:,:-1] += gyy

	# Boundary image
	boundary = boundarysrc.copy()
	boundary[1:-1,1:-1] = 0;

	# Subtract boundary contribution
	f_bp = -4*boundary[1:-1,1:-1] + boundary[1:-1,2:] + boundary[1:-1,0:-2] + boundary[2:,1:-1] + boundary[0:-2,1:-1]
	f = f[1:-1,1:-1] - f_bp

	# Discrete Sine Transform
	tt = scipy.fftpack.dst(f, norm='ortho')
	fsin = scipy.fftpack.dst(tt.T, norm='ortho').T

	# Eigenvalues
	(x,y) = numpy.meshgrid(range(1,f.shape[1]+1), range(1,f.shape[0]+1), copy=True)
	denom = (2*numpy.cos(math.pi*x/(f.shape[1]+2))-2) + (2*numpy.cos(math.pi*y/(f.shape[0]+2)) - 2)

	f = fsin/denom

	# Inverse Discrete Sine Transform
	tt = scipy.fftpack.idst(f, norm='ortho')
	img_tt = scipy.fftpack.idst(tt.T, norm='ortho').T

	# New center + old boundary
	result = boundary
	result[1:-1,1:-1] = img_tt

	return result


def poisson_reconstruct(gradx, grady, kernel_size=KERNEL_SIZE, num_iters=100, h=0.1, 
		boundary_image=None, boundary_zero=True):#返回num_iters次计算后的值
	"""
	Iterative algorithm for Poisson reconstruction. 
	Given the gradx and grady values, find laplacian(拉普拉斯算子), and solve for image
	Also return the squared difference of every step. #每一步差的平方累计值
	h = convergence rate
	"""
   #用Sobel算子进行图像梯度计算：
   #cv2.Sobel(img,cv2.CV_64F, 1, 0, ksize=3),img是源图像，cv2.CV_64F是cv提供的64位float，不用numpy.float64是怕溢出。第一个数字是对X求导即检测X方向是否有边缘，第2个数字是对Y求导。ksize是核的大小(核具体没太了解)
	fxx = cv2.Sobel(gradx, cv2.CV_64F, 1, 0, ksize=kernel_size)#检测X方向的边缘，对X求导
	fyy = cv2.Sobel(grady, cv2.CV_64F, 0, 1, ksize=kernel_size)#检测Y方向的边缘，对Y求导
	laplacian = fxx + fyy#上述步骤相当于直接调用cv2.Laplacian计算拉普拉斯算子
	m,n,p = laplacian.shape#laplacian矩阵的维度，几行+几列+深度

	if boundary_zero == True:
		est = np.zeros(laplacian.shape)#跟拉普拉斯算子矩阵等大的全0矩阵
	else:
		assert(boundary_image is not None)
		assert(boundary_image.shape == laplacian.shape)
		est = boundary_image.copy()

	est[1:-1, 1:-1, :] = np.random.random((m-2, n-2, p))#随机赋值，相当于要从一张噪声图逐渐接近原图
	loss = []#loss数组，存每一步差的平方累计值

	for i in xrange(num_iters):
		old_est = est.copy()
		est[1:-1, 1:-1, :] = 0.25*(est[0:-2, 1:-1, :] + est[1:-1, 0:-2, :] + est[2:, 1:-1, :] + est[1:-1, 2:, :] - h*h*laplacian[1:-1, 1:-1, :])
     #上式在估计原值，若在九宫格中，即V[5]=0.25*(V[2]+V[4]+V[6]+V[8]-laplacian)
		error = np.sum(np.square(est-old_est))#记录前后偏差，用于debug
		loss.append(error)

	return (est)#较粗糙的水印RGB图


def image_threshold(image, threshold=0.5):#转二值图，阈值=0.5
	'''
	Threshold the image to make all its elements greater than threshold*MAX = 1
	'''
	m, M = np.min(image), np.max(image)
	im = PlotImage(image)
	im[im >= threshold] = 1
	im[im < 1] = 0
	return im


def crop_watermark(gradx, grady, threshold=0.4, boundary_size=2):#裁剪水印
	"""
	Crops the watermark by taking the edge map of magnitude of grad(W)
	Assumes the gradx and grady to be in 3 channels
	@param: threshold - gives the threshold param
	@param: boundary_size - boundary around cropped image  #剪下的watermark四周有2像素的边框
	"""
	W_mod = np.sqrt(np.square(gradx) + np.square(grady))
	W_mod = PlotImage(W_mod)
	W_gray = image_threshold(np.average(W_mod, axis=2), threshold=threshold)# W转二值图
	x, y = np.where(W_gray == 1)# where(condition),输出:满足条件的xy坐标。此处认为水印和图像交界像素为白色=1，xy为所有白色像素位置

	xm, xM = np.min(x) - boundary_size - 1, np.max(x) + boundary_size + 1#记得要多保留边框
	ym, yM = np.min(y) - boundary_size - 1, np.max(y) + boundary_size + 1#记得要多保留边框

	return gradx[xm:xM, ym:yM, :] , grady[xm:xM, ym:yM, :]#返回水印所在区域的矩阵，非常粗糙


def normalized(img):
	"""
	Return the image between -1 to 1 so that its easier to find out things like 
	correlation between images, convolutionss, etc.
	Currently required for Chamfer distance for template matching.
	"""
	return (2*PlotImage(img)-1)

def watermark_detector(img, gx, gy, thresh_low=200, thresh_high=220, printval=False):
	"""
	Compute a verbose edge map using Canny edge detector, take its magnitude.
	Assuming cropped values of gradients are given.
	Returns image, start and end coordinates
	"""
	Wm = (np.average(np.sqrt(np.square(gx) + np.square(gy)), axis=2))#average(data,direction),axis=0,每列平均;axis=1,每行平均;axis=2,纵深平均。此处求完就成了单通道的水印大小的矩阵，后续用来卷积

	img_edgemap = (cv2.Canny(img, thresh_low, thresh_high))#Canny边缘检测，用2种阈值来检测出强边缘和弱边缘。只有当强弱边缘相连时才会把弱边缘包含在图像中。Canny算子最不易受噪声干扰
	chamfer_dist = cv2.filter2D(img_edgemap.astype(float), -1, Wm)#！！！这个返回的到底是什么数据类型？？？
   #void filter2D( InputArray src, OutputArray dst, int ddepth,InputArray kernel, Point anchor=Point(-1,-1), double delta=0, int borderType=BORDER_DEFAULT );    ddepth=-1表示输出图像与原图深度相同，Wm是卷积核kernel，是一个对边缘像素加权平均的函数


	rect = Wm.shape#获得图片形状，rect[0]图高，rect[1]图宽，rect[2]通道数
	index = np.unravel_index(np.argmax(chamfer_dist), img.shape[:-1])#倒角距离中最大者的坐标，也就是水印的中心点
	if printval:
		print(index)

	x,y = (index[0]-rect[0]/2), (index[1]-rect[1]/2)#求出水印矩形左上角顶点
	im = img.copy() 
	cv2.rectangle(im, (y, x), (y+rect[1], x+rect[0]), (255, 0, 0))#通过对角线画矩形，圈出水印所在
   #rectangle(img,pt1,pt2,color,thisckness,lineType,shift),pt1是一个左上角顶点，pt2是右下角顶点(Y轴正方向垂直向下，X轴水平向右),color顺序是BGR，thickness表示矩形边框厚度：负数表示全填充。
	return (im, (x, y), (rect[0], rect[1]))#返回水印图，水印左上角顶点，水印长宽
