import os
A = os.listdir('/home/MMI22jiho/deepsort/yolov5/rembg_copy/yolov5s_osnet_ibn_x1_0_MSMT17/crops/person')
for filename in A:
    os.mkdir("tes/%s"%filename)
    os.system("rembg p /home/MMI22jiho/deepsort/yolov5/rembg_copy/yolov5s_osnet_ibn_x1_0_MSMT17/crops/person/%s tes/%s" %(filename, filename))
