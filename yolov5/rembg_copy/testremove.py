import glob
import numpy as np
import cv2
import os
import numba
from numba import jit
         
B = os.listdir('/home/MMI22jiho/deepsort/yolov5/rembg_copy/tes')
@jit
def makeGEI(c, GEI, k,height, width):
    for j in range(height):
        for i in range(width):
            if c[j,i]>1:
                GEI[j,i] = GEI[j,i] + 255/k
            else:
                c[j,i]=0
    return GEI

for filename in B:
    path = "tes/%s/*.png" %filename
    GEI = np.zeros((128,88), np.uint8)
    for bb,file in enumerate (glob.glob(path)):
        image_read = cv2.imread(file)
        c = cv2.cvtColor(image_read, cv2.COLOR_BGR2GRAY)
        c = cv2.resize(c, dsize=(88,128), interpolation=cv2.INTER_CUBIC)
        height, width = c.shape
        k = len(glob.glob(path))
        if k>15:
           makeGEI(c, GEI, k, height, width)
           cv2.imwrite('GEIext/A/test/{}.png'.format(filename),GEI)








    #     for j in range(height):
    #         for i in range(width):
    #             if c[j,i]>1:
    #                 GEI[j,i] = GEI[j,i] + 255/len(glob.glob(path))
    #             else:
    #                 c[j,i]=0
    # cv2.imwrite('GEIext/A/test/{}.png'.format(filename),GEI)

# path = "tes/*.png" 
# GEI = np.zeros((128,88), np.uint8)
# for bb,file in enumerate (glob.glob(path)):
#     image_read = cv2.imread(file)
#     c = cv2.cvtColor(image_read, cv2.COLOR_BGR2GRAY)
#     c = cv2.resize(c, dsize=(88,128), interpolation=cv2.INTER_CUBIC)
#     height, width = c.shape
#     k = len(glob.glob(path))
#     if k>5:
#         makeGEI(c, GEI, k, height, width)
#         cv2.imwrite('/home/MMI22jiho/deepsort/yolov5/rembg_copy/GEIext/A/test/A.png',GEI)



# import shutil
# shutil.rmtree("/home/MMI22jiho/deepsort/yolov5/rembg_copy/GEIext/tas")
# os.mkdir("/home/MMI22jiho/deepsort/yolov5/rembg_copy/GEIext/tas")
# A = os.listdir('/home/MMI22jiho/deepsort/yolov5/rembg_copy/tes')
# @jit
# def makeGEI(c, GEI,k,height, width):
#     for j in range(height):
#         for i in range(width):
#             if c[j,i]>1:
#                 GEI[j,i] = GEI[j,i] + 255/k
#             else:
#                 c[j,i]=0
#     return GEI
# GEI = np.zeros((128,88), np.uint8)
# for filename in A:
#     os.mkdir("/home/MMI22jiho/deepsort/yolov5/rembg_copy/GEIext/tas/%s"%filename)
#     K = os.listdir('/home/MMI22jiho/deepsort/yolov5/rembg_copy/tes/%s'%filename)
#     for i in range(0,len(K)):
#         v = K[i]
#         v = v[:-4]
#         K[i] = v
#     K.sort(key=int)
#     for i in range(0,len(K)):
#         K[i] = K[i] +'.png'
#     for length in range(0,len(K)):
#         GEI = np.zeros((128,88), np.uint8)
#         for file in K[0:length]:
#             image_read = cv2.imread('/home/MMI22jiho/deepsort/yolov5/rembg_copy/tes/%s/%s'%(filename,file))
#             c = cv2.cvtColor(image_read, cv2.COLOR_BGR2GRAY)
#             c = cv2.resize(c, dsize=(88,128), interpolation=cv2.INTER_CUBIC)
#             height, width = c.shape
#             if len(K)>5:
#                 makeGEI(c,GEI,length,height,width)
#                 cv2.imwrite('GEIext/tas/%s/{}.png'.format(length)%filename,GEI)



# oa=15               
# import shutil
# shutil.rmtree("/home/MMI22jiho/deepsort/yolov5/rembg_copy/GEIext/tas")
# os.mkdir("/home/MMI22jiho/deepsort/yolov5/rembg_copy/GEIext/tas")
# A = os.listdir('/home/MMI22jiho/deepsort/yolov5/rembg_copy/tes')
# @jit
# def makeGEI(c, GEI,k,height, width):
#     for j in range(height):
#         for i in range(width):
#             if c[j,i]>1:
#                 GEI[j,i] = GEI[j,i] + 255/k
#             else:
#                 c[j,i]=0
#     return GEI
# GEI = np.zeros((128,88), np.uint8)
# for filename in A:
#     os.mkdir("/home/MMI22jiho/deepsort/yolov5/rembg_copy/GEIext/tas/%s"%filename)
#     K = os.listdir('/home/MMI22jiho/deepsort/yolov5/rembg_copy/tes/%s'%filename)
#     for i in range(0,len(K)):
#         v = K[i]
#         v = v[:-4]
#         v = v[8:]
#         K[i] = v
#         print[K]
    # K.sort(key=int)
    # for i in range(0,len(K)):
    #     K[i] = K[i] +'.png'
    # GEI = np.zeros((128,88), np.uint8)
    # for file in K[oa:len(K)]:
    #     image_read = cv2.imread('/home/MMI22jiho/deepsort/yolov5/rembg_copy/tes/%s/%s'%(filename,file))
    #     c = cv2.cvtColor(image_read, cv2.COLOR_BGR2GRAY)
    #     c = cv2.resize(c, dsize=(88,128), interpolation=cv2.INTER_CUBIC)
    #     height, width = c.shape
    #     if len(K)>5:
    #         makeGEI(c,GEI,len(K)-oa,height,width)
    #         cv2.imwrite('GEIext/tas/%s/{}.png'.format(filename)%filename,GEI)



# oa=0               
# import shutil
# shutil.rmtree("/home/MMI22jiho/deepsort/yolov5/rembg_copy/GEIext/tas")
# os.mkdir("/home/MMI22jiho/deepsort/yolov5/rembg_copy/GEIext/tas")
# A = os.listdir('/home/MMI22jiho/deepsort/yolov5/rembg_copy/tes')
# @jit
# def makeGEI(c, GEI,k,height, width):
#     for j in range(height):
#         for i in range(width):
#             if c[j,i]>1:
#                 GEI[j,i] = GEI[j,i] + 255/k
#             else:
#                 c[j,i]=0
#     return GEI
# GEI = np.zeros((128,88), np.uint8)
# for filename in A:
#     os.mkdir("/home/MMI22jiho/deepsort/yolov5/rembg_copy/GEIext/tas/%s"%filename)
#     K = os.listdir('/home/MMI22jiho/deepsort/yolov5/rembg_copy/tes/%s'%filename)

#     for i in range(0,len(K)):
#         v = K[i]
#         v = v[:-4]
#         v = v[4:]
#         K[i] = v
#     K.sort(key=int)
#     for i in range(0,len(K)):
#         K[i] = 'test' +K[i] +'.png'
#         print(K[i])
#     for length in range(oa,len(K)-oa):
#         GEI = np.zeros((128,88), np.uint8)
#         for file in K[0:length]:
#             image_read = cv2.imread('/home/MMI22jiho/deepsort/yolov5/rembg_copy/tes/%s/%s'%(filename,file))
#             c = cv2.cvtColor(image_read, cv2.COLOR_BGR2GRAY)
#             c = cv2.resize(c, dsize=(88,128), interpolation=cv2.INTER_CUBIC)
#             height, width = c.shape
#             if len(K)>5:
#                 makeGEI(c,GEI,length,height,width)
#                 cv2.imwrite('GEIext/tas/%s/{}.png'.format(length)%filename,GEI)





















# for filename in B:
#     path = "tes/%s/*.png" %filename
#     GEI = np.zeros((128,88), np.uint8)
#     for bb,file in enumerate (glob.glob(path)):
#         image_read = cv2.imread(file)
#         c = cv2.cvtColor(image_read, cv2.COLOR_BGR2GRAY)
#         c = cv2.resize(c, dsize=(88,128), interpolation=cv2.INTER_CUBIC)
#         height, width = c.shape
#         k = len(glob.glob(path))
#         if k>5:
#            makeGEI(c, GEI, k, height, width)
#            cv2.imwrite('GEIext/A/test/{}.png'.format(filename),GEI)






 





  