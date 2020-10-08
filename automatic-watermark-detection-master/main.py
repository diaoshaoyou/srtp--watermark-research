from src import *

gx, gy, gxlist, gylist = estimate_watermark('images/fotolia_processed')#gx、gy是梯度中位数，gxlist、gylist是J梯度

# est = poisson_reconstruct(gx, gy, np.zeros(gx.shape)[:,:,0])
cropped_gx, cropped_gy = crop_watermark(gx, gy)#裁剪水印，cropped_gx、cropped_gy是水印(包括边框)所在的区域范围
W_m = poisson_reconstruct(cropped_gx, cropped_gy)#泊松重建

# random photo
img = cv2.imread('images/fotolia_processed/fotolia_137840645.jpg')
im, start, end = watermark_detector(img, cropped_gx, cropped_gy)

# plt.imshow(im)
# plt.show()
# We are done with watermark estimation
# W_m is the cropped watermark
num_images = len(gxlist)#总共要裁剪的图片数目

J, img_paths = get_cropped_images('images/fotolia_processed', num_images, start, end, cropped_gx.shape)#返回的J是一个四维矩阵，第一位是图片数，后三维是图片矩阵
# get a random subset of J
idx = [389, 144, 147, 468, 423, 92, 3, 354, 196, 53, 470, 445, 314, 349, 105, 366, 56, 168, 351, 15, 465, 368, 90, 96, 202, 54, 295, 137, 17, 79, 214, 413, 454, 305, 187, 4, 458, 330, 290, 73, 220, 118, 125, 180, 247, 243, 257, 194, 117, 320, 104, 252, 87, 95, 228, 324, 271, 398, 334, 148, 425, 190, 78, 151, 34, 310, 122, 376, 102, 260]
idx = idx[:25]
# Wm = (255*PlotImage(W_m))
Wm = W_m - W_m.min()

# get threshold of W_m for alpha matte estimate
alph_est = estimate_normalized_alpha(J, Wm)#估计标准化的an，每张图都有，是个数组
alph = np.stack([alph_est, alph_est, alph_est], axis=2)
C, est_Ik = estimate_blend_factor(J, Wm, alph)#估计c

alpha = alph.copy()
for i in xrange(3):
	alpha[:,:,i] = C[i]*alpha[:,:,i]

Wm = Wm + alpha*est_Ik

W = Wm.copy()
for i in xrange(3):
	W[:,:,i]/=C[i]

Jt = J[:25]
# now we have the values of alpha, Wm, J
# Solve for all images
Wk, Ik, W, alpha1 = solve_images(Jt, W_m, alpha, W)
# W_m_threshold = (255*PlotImage(np.average(W_m, axis=2))).astype(np.uint8)
# ret, thr = cv2.threshold(W_m_threshold, 127, 255, cv2.THRESH_BINARY)  

