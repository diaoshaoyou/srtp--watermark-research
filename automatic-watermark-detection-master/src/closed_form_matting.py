#作用：计算a_n
#closed_form_matte函数在watermark_reconstruct中被调用，computeLaplacian在closed_form_matte中被调用
from __future__ import division

import numpy as np
import scipy.sparse
import scipy
from scipy.sparse import *
from numpy.lib.stride_tricks import as_strided


def rolling_block(A, block=(3, 3)):
    shape = (A.shape[0] - block[0] + 1, A.shape[1] - block[1] + 1) + block
    strides = (A.strides[0], A.strides[1]) + A.strides
    return as_strided(A, shape=shape, strides=strides)


# Returns sparse matting laplacian
def computeLaplacian(img, eps=10**(-7), win_rad=1):
    win_size = (win_rad*2+1)**2
    h, w, d = img.shape
    # Number of window centre indices in h, w axes
    c_h, c_w = h - 2*win_rad, w - 2*win_rad
    win_diam = win_rad*2+1

    indsM = np.arange(h*w).reshape((h, w))
    ravelImg = img.reshape(h*w, d)
    win_inds = rolling_block(indsM, block=(win_diam, win_diam))

    win_inds = win_inds.reshape(c_h, c_w, win_size)
    winI = ravelImg[win_inds]

    win_mu = np.mean(winI, axis=2, keepdims=True)
    win_var = np.einsum('...ji,...jk ->...ik', winI, winI)/win_size - np.einsum('...ji,...jk ->...ik', win_mu, win_mu)

    inv = np.linalg.inv(win_var + (eps/win_size)*np.eye(3))

    X = np.einsum('...ij,...jk->...ik', winI - win_mu, inv)
    vals = np.eye(win_size) - (1/win_size)*(1 + np.einsum('...ij,...kj->...ik', X, winI - win_mu))

    nz_indsCol = np.tile(win_inds, win_size).ravel()
    nz_indsRow = np.repeat(win_inds, win_size).ravel()
    nz_indsVal = vals.ravel()
    L = scipy.sparse.coo_matrix((nz_indsVal, (nz_indsRow, nz_indsCol)), shape=(h*w, h*w))
    return L


def closed_form_matte(img, scribbled_img, mylambda=100):#scibbled_img:三通道二值图
    h, w,c  = img.shape
    consts_map = (np.sum(abs(img - scribbled_img), axis=-1)>0.001).astype(np.float64)#consts_map:水印和非水印区像素=1，不确定区=0
    #scribbled_img = rgb2gray(scribbled_img)

    consts_vals = scribbled_img[:,:,0]*consts_map
    D_s = consts_map.ravel()#ravel()把多维数组转换成一维数组W*h
    b_s = consts_vals.ravel()
    # print("Computing Matting Laplacian")
    L = computeLaplacian(img)
    sD_s = scipy.sparse.diags(D_s)#scipy.sparse.diags(diagonals, offsets=0, shape=None, format=None, dtype=None)从对角线构造一个稀疏矩阵
    # print("Solving for alpha")
    x = scipy.sparse.linalg.spsolve(L + mylambda*sD_s, mylambda*b_s)#scipy.sparse.linalg.spsolve(A, b, permc_spec=None, use_umfpack=True)求解Ax=b中的x
    #x是w*h大小的一位数组
    alpha = np.minimum(np.maximum(x.reshape(h, w), 0), 1)#minimum(matrix,standard),选取matrix当中<=standard的数，若无则赋standard。eg. a=[[1,2],[3,4]],minimum(a,1)的结果为[[1,1],[1,1]]    minimum(a,3)的结果为[[1,2],[3,3]]  
    #maximum(x.reshape(h,w),0)是为了消除负数，minimum(…,1)使所有数<=1
    return alpha
