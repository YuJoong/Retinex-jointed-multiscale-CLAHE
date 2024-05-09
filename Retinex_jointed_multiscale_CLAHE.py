import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import time

def normalize(img):
    a=np.min(img)
    b=np.max(img)
    normailized_img = ((img - a) / (b-a)) * 255.0

    return normailized_img
def normalize1(img,low_clip=0.01,high_clip=0.99):
    sort_array=np.sort(img,axis=None)
    min_position=int(len(sort_array)*low_clip)
    minvalue=sort_array[min_position]
    max_position=int(len(sort_array)*high_clip)
    maxvalue=sort_array[max_position]
    
    out_img=(img-minvalue)/(maxvalue-minvalue)
    out_img[out_img<0.0]=0.0
    out_img[out_img>1.0]=1.0
    out_img = out_img * 255.0

    return out_img
def ssr(img, sigma):
    
    retinex = (np.log10(img+1.0) - np.log10(cv2.GaussianBlur(img, (0,0), sigma)+1.0))

    retinex = normalize1(retinex)
    return retinex

def msr(img, sigma_list):
    msr = np.zeros(img.shape)

    for sigma in sigma_list:
        msr += ssr(img,sigma)

    msr = msr / len(sigma_list)
     
    return msr

start = time.time()
img = cv2.imread('D:/Image/high_2.jpg', cv2.IMREAD_COLOR)
Lab2 = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
L2,a2,b2 = cv2.split(Lab2) 
L_fused = np.zeros(L2.shape)
a_r = np.zeros(a2.shape)
b_r = np.zeros(b2.shape)
RetinexHE = np.zeros(img.shape)
cc = np.zeros(img.shape)

size_list = [(4,4),(8,8),(16,16)]
clip_limit = 1.0

Lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
L,a,b = cv2.split(Lab) 

L_250 = ssr(L, 250)
L_250 = np.uint8(np.clip(L_250,0,255))
for (x,y) in size_list:
    clahe = cv2.createCLAHE(clip_limit, tileGridSize= (x,y))

    L_250_clahe = clahe.apply(L_250)

    clip_limit += 1.0
    L_fused += L_250_clahe

L_fused = L_fused / len(size_list)
L_fused = np.uint8(L_fused)
RetinexHE = cv2.merge((L_fused,a,b))

img = img.astype(np.float64)+1.0
crf = 46*(np.log10(125*img)-np.log10(np.sum(img,axis=2,keepdims=True)))

RetinexHE = cv2.cvtColor(RetinexHE, cv2.COLOR_LAB2BGR)

RetinexHE_CC = RetinexHE*crf
RetinexHE_CC = 192*(RetinexHE_CC-(-30))
RetinexHE_CC = 255*(RetinexHE_CC-np.min(RetinexHE_CC))/(np.max(RetinexHE_CC)-np.min(RetinexHE_CC))
RetinexHE_CC = np.uint8(RetinexHE_CC)

Lab_CC = cv2.cvtColor(RetinexHE_CC, cv2.COLOR_BGR2LAB)
L_CC, a_CC, b_CC = cv2.split(Lab_CC)
result = cv2.merge((L_fused,a_CC,b_CC))
result = cv2.cvtColor(result,cv2.COLOR_LAB2BGR)
cv2.imwrite('D:/Retinex HE/time_test2.jpg',result)
RetinexHE = RetinexHE / 255
RetinexHE_CC = RetinexHE_CC/255
result = result/255

end = time.time()
cv2.imshow('RetinexHE', result)
print(f"{end-start:.5f} sec")

cv2.waitKey(0)
cv2.destroyAllWindows()