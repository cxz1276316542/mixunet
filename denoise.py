import matplotlib.pyplot as plt
from skimage.restoration import (denoise_wavelet, estimate_sigma)
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio
import skimage.io


import cv2 as cv
import numpy as np


import numpy as np
import time

def resize(src, new_size):
    dst_w, dst_h = new_size # 目标图像宽高
    src_h, src_w = src.shape[:2] # 源图像宽高
    if src_h == dst_h and src_w == dst_w:
        return src.copy()
    scale_x = float(src_w) / dst_w # x缩放比例
    scale_y = float(src_h) / dst_h # y缩放比例

    # 遍历目标图像，插值
    dst = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    for n in range(3): # 对channel循环
        for dst_y in range(dst_h): # 对height循环
            for dst_x in range(dst_w): # 对width循环
                # 目标在源上的坐标（浮点值）
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5
                # 计算在源图上的（i,j）,和（i+1,j+1）的值
                src_x_0 = int(np.floor(src_x))  # floor是下取整
                src_y_0 = int(np.floor(src_y))
                src_x_1 = min(src_x_0 + 1, src_w - 1)
                src_y_1 = min(src_y_0 + 1, src_h - 1)

                # 双线性插值
                value0 = (src_x_1 - src_x) * src[src_y_0, src_x_0, n] + (src_x - src_x_0) * src[src_y_0, src_x_1, n]
                value1 = (src_x_1 - src_x) * src[src_y_1, src_x_0, n] + (src_x - src_x_0) * src[src_y_1, src_x_1, n]
                dst[dst_y, dst_x, n] = int((src_y_1 - src_y) * value0 + (src_y - src_y_0) * value1)
    return dst
# 直接对输入图像转换为灰度图像，然后二值化
def method_1(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    t, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    return binary

# 首先对输入图像进行降噪，去除噪声干扰，然后再二值化
def method_2(image):
    blurred = cv.GaussianBlur(image, (3, 3), 0)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    t, binary = cv.threshold(gray, 0, 100, cv.THRESH_BINARY | cv.THRESH_OTSU)
    return binary

# 图像先均值迁移去噪声，然后二值化的图像
def method_3(image):
    blurred = cv.pyrMeanShiftFiltering(image, 10, 100)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    t, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    return binary

# # 读取图像
# src = cv.imread("D:/vsprojects/images/coins.jpg")
# h, w = src.shape[:2]
#
# # 调用方法
# ret = method_3(src)
#
# result = np.zeros([h, w*2, 3], dtype=src.dtype)
# result[0:h,0:w,:] = src
# result[0:h,w:2*w,:] = cv.cvtColor(ret, cv.COLOR_GRAY2BGR)
#
# # 图像显示
# cv.putText(result, "input", (10, 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 2)
# cv.putText(result, "binary", (w+10, 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 2)
# cv.imshow("result", result)
#
# # 图像存储
# cv.imwrite("D:/vsprojects/images/coins_binary.png", result)
#
# cv.waitKey(0)
# cv.destroyAllWindows()
def waveletDenoiseGray(src):
    # 读取图片，并转换成flaot类型
    src = skimage.io.imread(src)
    # img = skimage.img_as_float(img)
    src = cv.resize(src, (224, 224))
    h, w = src.shape[:2]
    #
    # 调用方法
    ret = method_1(src)

    result = np.zeros([h, w*2, 3], dtype=src.dtype)
    result[0:h,0:w,:] = src
    result[0:h,w:2*w,:] = cv.cvtColor(ret, cv.COLOR_GRAY2BGR)
    # # 往图片添加随机噪声
    # sigma = 0.1
    # imgNoise = random_noise(img, var=sigma ** 2)
    img= result
    # 估计当前的图像的噪声的方差
    # 由于随机噪声的裁切，估计的sigma值将小于指定的sigma的值
    sigma_est = estimate_sigma(img, average_sigmas=True)

    # 对图像分别使用Bayesshink算法和Visushrink算法
    # 输入带噪图像，小波变换模式选择 ，阈值模式，小波分解的级别，小波基，
    imgBayes = denoise_wavelet(img, method='BayesShrink', mode='soft',
                               wavelet_levels=3, wavelet='bior6.8',
                               rescale_sigma=True)

    imgVisushrink = denoise_wavelet(img, method='VisuShrink', mode='soft',
                                    sigma=sigma_est / 3, wavelet_levels=5,
                                    wavelet='bior6.8', rescale_sigma=True)

    # 计算输入和输出之间的PSNR值
    # psnrNoise = peak_signal_noise_ratio(img, imgNoise)
    psnrBayes = peak_signal_noise_ratio(img, imgBayes)
    psnrVisushrink = peak_signal_noise_ratio(img, imgVisushrink)

    # 将降噪图片结果输出出来
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap=plt.cm.gray)
    plt.title('Original Image')

    # plt.subplot(2, 2, 2)
    # plt.imshow(imgNoise, cmap=plt.cm.gray)
    # plt.title('Noise Image')

    plt.subplot(2, 2, 3)
    plt.imshow(imgBayes, cmap=plt.cm.gray)
    plt.title('Denoised Image using Bayes')

    plt.subplot(2, 2, 4)
    plt.imshow(imgVisushrink, cmap=plt.cm.gray)
    plt.title('Denoised Image using Visushrink')

    plt.show()

    # 将PSNR的值打印处理
    print('estimate sigma:', sigma_est)
    # print('PSNR[orignal vs NoiseImgae]:', psnrNoise)
    print('PSNR[orignal vs Denoise[Bayes]]:', psnrBayes)
    print('PSNR[orignal vs Denoise[Visushrink]]:', psnrVisushrink)


def waveletDenoiseRgb(src):
    # 读取图片，并转换成flaot类型
    img = skimage.io.imread(src)
    img = skimage.img_as_float(img)

    # 往图片添加随机噪声
    sigma = 0.15
    imgNoise = random_noise(img, var=sigma ** 2)

    # 估计当前的图像的噪声的方差
    # 由于随机噪声的裁切，估计的sigma值将小于指定的sigma的值,彩色图片需要设定多通道
    sigma_est = estimate_sigma(imgNoise, multichannel=True, average_sigmas=True)

    # 对图像分别使用Bayesshink算法和Visushrink算法
    # 输入带噪图像，小波变换模式选择 ，阈值模式，小波分解的级别，小波基，
    imgBayes = denoise_wavelet(imgNoise, method='BayesShrink', mode='soft',
                               wavelet_levels=3, wavelet='coif5',
                               multichannel=True, convert2ycbcr=True,
                               rescale_sigma=True)

    imgVisushrink = denoise_wavelet(imgNoise, method='VisuShrink', mode='soft',
                                    sigma=sigma_est / 3, wavelet_levels=5,
                                    wavelet='coif5',
                                    multichannel=True, convert2ycbcr=True, rescale_sigma=True)
    # 计算输入和输出之间的PSNR值
    psnrNoise = peak_signal_noise_ratio(img, imgNoise)
    psnrBayes = peak_signal_noise_ratio(img, imgBayes)
    psnrVisushrink = peak_signal_noise_ratio(img, imgVisushrink)

    # 将降噪图片结果输出出来
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap=plt.cm.gray)
    plt.title('Original Image')

    plt.subplot(2, 2, 2)
    plt.imshow(imgNoise, cmap=plt.cm.gray)
    plt.title('Noise Image')

    plt.subplot(2, 2, 3)
    plt.imshow(imgBayes, cmap=plt.cm.gray)
    plt.title('Denoised Image using Bayes')

    plt.subplot(2, 2, 4)
    plt.imshow(imgVisushrink, cmap=plt.cm.gray)
    plt.title('Denoised Image using Visushrink')

    plt.show()

    # 将PSNR的值打印处理
    print('estimate sigma:', sigma_est)
    print('PSNR[orignal vs NoiseImgae]:', psnrNoise)
    print('PSNR[orignal vs Denoise[Bayes]]:', psnrBayes)
    print('PSNR[orignal vs Denoise[Visushrink]]:', psnrVisushrink)


if __name__ == "__main__":
    inputSrc = 'D:/code_cxz/mixformer/PaddleClas-c/dataset/thyroid/train/malignancy/52.jpg'
    # inputSrc='C:/Users/admin/Desktop/project/originPhoto/lena.png'
    waveletDenoiseGray(inputSrc)