#1.算出所有图的梯度中位数
#2.用中位数裁剪出极其粗糙的水印图
#3.对裁剪出的水印拉普拉斯算子边缘检测，得到较粗糙的水印图和位置
#4.用较粗糙的水印图位置对某张图片的水印位置进行检测
#5.把所有图都粗糙地裁剪水印
#……
from src import *

gx, gy, gxlist, gylist = estimate_watermark('images/fotolia_processed')#gx、gy是梯度中位数，gxlist、gylist是J梯度

# est = poisson_reconstruct(gx, gy, np.zeros(gx.shape)[:,:,0])
cropped_gx, cropped_gy = crop_watermark(gx, gy)#裁剪出极其粗糙的水印，cropped_gx、cropped_gy是水印(包括边框)所在的区域范围
W_m = poisson_reconstruct(cropped_gx, cropped_gy)#泊松重建，对裁剪后的水印进行拉普拉斯边缘检测

# random photo
img = cv2.imread('images/fotolia_processed/fotolia_137840645.jpg')
im, start, end = watermark_detector(img, cropped_gx, cropped_gy)#找到水印位置，start是水印外框矩形的左上角顶点，end是长和宽

# plt.imshow(im)
# plt.show()
# We are done with watermark estimation
# W_m is the cropped watermark
num_images = len(gxlist)#总共要裁剪的图片数目

J, img_paths = get_cropped_images('images/fotolia_processed', num_images, start, end, cropped_gx.shape)#把该文件夹里的图都裁剪出水印。J是裁剪出的水印图，img_paths是图片路径。返回的J是一个四维矩阵，第一位是图片数，后三维是图片矩阵
# get a random subset of J
idx = [389, 144, 147, 468, 423, 92, 3, 354, 196, 53, 470, 445, 314, 349, 105, 366, 56, 168, 351, 15, 465, 368, 90, 96, 202, 54, 295, 137, 17, 79, 214, 413, 454, 305, 187, 4, 458, 330, 290, 73, 220, 118, 125, 180, 247, 243, 257, 194, 117, 320, 104, 252, 87, 95, 228, 324, 271, 398, 334, 148, 425, 190, 78, 151, 34, 310, 122, 376, 102, 260]
idx = idx[:25]
# Wm = (255*PlotImage(W_m))
Wm = W_m - W_m.min() #使W_m的最小值=0，为什么？？
#matrix.min()找到矩阵最小值;matrix.min(0)选出每个列中最小的组成一个数组;matrix.min(1)选出每个行中最小的组成一个数组

# a_n=0或1,相当于标识了水印的区域。c表示不透明度，每张图相同
alph_est = estimate_normalized_alpha(J, Wm)#估计标准化了的alpha matte(又叫a_n)，也就是求了所有图alpha的中位数，是个二维矩阵
alph = np.stack([alph_est, alph_est, alph_est], axis=2)#shape=(m,n,3), m,n是水印大小
#np.stack(array, axis) array是要堆叠的矩阵，axis是堆叠要沿着的维度。对于axis=1，就是横着切开，对应行横着堆;对于axis=2，就是横着切开，对应行竖着堆;对于axis=0，就是不切开，两个堆一起。 eg.假设A=([1,2,3],[1,2,3],[1,2,3])  B=([4,5,6],[4,5,6],[4,5,6]), 则np.stack((A,B),aixs=2)为
#    1,1,1 
#    2,2,2 
#    3,3,3          
#    - - -       所以shape=(3,3,2)
#    4,4,4
#    5,5,5
#    6,6,6

C, est_Ik = estimate_blend_factor(J, Wm, alph)#估计常数blending factor(又叫c)    

alpha = alph.copy()
for i in xrange(3):
	alpha[:,:,i] = C[i]*alpha[:,:,i]#alpha=c*a_n

Wm = Wm + alpha*est_Ik#据论文，Wm=c*a_n*W −c*a_n*E[Ik],所以求得Wm=c*a_n*W,所以下面要除以c

W = Wm.copy()
for i in xrange(3):
	W[:,:,i]/=C[i]#此处程序中的W=a_n*无水印原图W(论文中的)
Jt = J[:25]
# now we have the values of alpha, Wm, J
# Solve for all images
Wk, Ik, W, alpha1 = solve_images(Jt, W_m, alpha, W)#所有图
# W_m_threshold = (255*PlotImage(np.average(W_m, axis=2))).astype(np.uint8)
# ret, thr = cv2.threshold(W_m_threshold, 127, 255, cv2.THRESH_BINARY)  

