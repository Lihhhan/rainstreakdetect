#coding=utf-8
import pywt, sys
import cv2
import numpy as np
from matplotlib import pyplot as plot

img = cv2.imread(sys.argv[1])[:, :, 0]

#detect rain streak with bilateralFilter
def detect(img):
    #滤波
    blur = cv2.bilateralFilter(img, 7, 75, 75) 

    #小波变换找边界
    res = recursion_body(img)
    res2 = recursion_body(blur)

    res = normalized(res)
    res2 = normalized(res2)

    ret, res3 = cv2.threshold(res, 150, 255, cv2.THRESH_BINARY)
    ret, res4 = cv2.threshold(res2, 150, 255, cv2.THRESH_BINARY)


    fig = plot.figure()
    plot.subplot(321), plot.imshow(img, cmap='gray')
    plot.title('origin'), plot.xticks([]), plot.yticks([])
    plot.subplot(323), plot.imshow(res, cmap='gray')
    plot.title('origin idwt2'), plot.xticks([]), plot.yticks([])
    plot.subplot(322), plot.imshow(blur, cmap='gray')
    plot.title('blur'), plot.xticks([]), plot.yticks([])
    plot.subplot(324), plot.imshow(res2, cmap='gray')
    plot.title('blur idwt2'), plot.xticks([]), plot.yticks([])
    plot.subplot(325), plot.imshow(res3, cmap='gray')
    plot.title('origin idwt2 threshold'), plot.xticks([]), plot.yticks([])
    plot.subplot(326), plot.imshow(res4, cmap='gray')
    plot.title('blur idwt2 threshold'), plot.xticks([]), plot.yticks([])

    #求差，二值化
    th_diff = res3 - res4
    diff = res - res2

    ret, diff_th = cv2.threshold(diff,250,255, cv2.THRESH_BINARY)
    #th2 = cv2.adaptiveThreshold(diff,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
    #th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    
    dft1 = dft(th_diff)
    dft2 = dft(diff_th)

    imgs = [diff_th, th_diff, dft1, dft2]
    titles = ['diff_th', 'th_diff', 'dft th_diff', 'dft diff_th']

    fig = plot.figure()
    for i in xrange(4):
        plot.subplot(2, 2, i+1), plot.imshow(imgs[i],'gray')
        plot.title(titles[i]), plot.xticks([]), plot.yticks([])

    plot.show()


#dwt2
def recursion_body(LL, c = 2):
    if c > 0 :
        coeffs = pywt.dwt2(LL, 'bior1.3')
        LL, (LH, HL, HH) = coeffs
        c -= 1
        new_LL = recursion_body(LL, c)
        y, x = LL.shape
        new_LL = cv2.resize(new_LL, (x, y))
        #print new_LL.shape, LL.shape
        return pywt.idwt2((new_LL, (LH, HL, HH)), 'bior1.3') 
    else:
        return np.zeros(LL.shape)    

#归一化
def normalized(img):
    l = (np.max(img) -np.min(img)) / 255
    img -= np.min(img)
    img /= l
    img = np.array(img, np.uint8)
    return img

#dft
def dft(img):
    dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
    return magnitude_spectrum

detect(img)







