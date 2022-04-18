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
        if k>5:
            makeGEI(c, GEI, k, height, width)
            cv2.imwrite('GEIext/A/test/{}.png'.format(filename),GEI)
    #     for j in range(height):
    #         for i in range(width):
    #             if c[j,i]>1:
    #                 GEI[j,i] = GEI[j,i] + 255/len(glob.glob(path))
    #             else:
    #                 c[j,i]=0
    # cv2.imwrite('GEIext/A/test/{}.png'.format(filename),GEI)




  