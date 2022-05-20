import os
if os.path.exists('/home/MMI22jiho/deepsort/yolov5/rembg_copy/yolov5s_osnet_ibn_x1_0_MSMT17/crops/person'):
    A = os.listdir('/home/MMI22jiho/deepsort/yolov5/rembg_copy/yolov5s_osnet_ibn_x1_0_MSMT17/crops/person')

    for filename in A:
        os.mkdir("tes/%s"%filename)
        os.system("rembg p /home/MMI22jiho/deepsort/yolov5/rembg_copy/yolov5s_osnet_ibn_x1_0_MSMT17/crops/person/%s tes/%s" %(filename, filename))


# A = os.listdir('/home/MMI22jiho/deepsort/yolov5/rembg_copy/yolov5s_osnet_ibn_x1_0_MSMT17/crops/person')
# for filename in A:
#     os.mkdir("sil/%s"%filename)
#     os.mkdir("tes/%s"%filename)
#     B = (os.listdir('/home/MMI22jiho/deepsort/yolov5/rembg_copy/yolov5s_osnet_ibn_x1_0_MSMT17/crops/person/%s' %filename))
#     if len(B)>15:
#         for file in B[5:20]:
#             os.system("cp /home/MMI22jiho/deepsort/yolov5/rembg_copy/yolov5s_osnet_ibn_x1_0_MSMT17/crops/person/%s/%s sil/%s" %(filename, file, filename))
#     else:
#         for file in B[0:len(B)]:
#             os.system("cp /home/MMI22jiho/deepsort/yolov5/rembg_copy/yolov5s_osnet_ibn_x1_0_MSMT17/crops/person/%s/%s sil/%s" %(filename, file, filename))
#     os.system("rembg p /home/MMI22jiho/deepsort/yolov5/rembg_copy/sil/%s tes/%s" %(filename, filename))

# A = os.listdir('/home/MMI22jiho/deepsort/yolov5/rembg_copy/yolov5s_osnet_ibn_x1_0_MSMT17/crops/person')
# for filename in A:
#     os.mkdir("sil/%s"%filename)
#     os.mkdir("tes/%s"%filename)
#     B = (os.listdir('/home/MMI22jiho/deepsort/yolov5/rembg_copy/yolov5s_osnet_ibn_x1_0_MSMT17/crops/person/%s' %filename))
#     if len(B)>15:
#         for file in B[0:len(B)]:
#             os.system("cp /home/MMI22jiho/deepsort/yolov5/rembg_copy/yolov5s_osnet_ibn_x1_0_MSMT17/crops/person/%s/%s sil/%s" %(filename, file, filename))
#     else:
#         for file in B[0:len(B)]:
#             os.system("cp /home/MMI22jiho/deepsort/yolov5/rembg_copy/yolov5s_osnet_ibn_x1_0_MSMT17/crops/person/%s/%s sil/%s" %(filename, file, filename))
#     os.chdir("/home/MMI22jiho/yolact")
#     os.system("python eval.py --trained_model=weights/yolact_resnet50_54_800000.pth --score_threshold=0.10 --top_k=15 --images=/home/MMI22jiho/deepsort/yolov5/rembg_copy/sil/%s:/home/MMI22jiho/deepsort/yolov5/rembg_copy/tes/%s" %(filename, filename))
#     os.chdir("/home/MMI22jiho/deepsort/yolov5/rembg_copy")


# import shutil
# shutil.rmtree("/home/MMI22jiho/deepsort/yolov5/rembg_copy/tes")
# os.mkdir("tes")
# shutil.rmtree("/home/MMI22jiho/deepsort/yolov5/rembg_copy/sil")
# os.mkdir("sil")
# shutil.rmtree("/home/MMI22jiho/deepsort/yolov5/rembg_copy/gait")
# os.mkdir("gait")

# A = os.listdir('/home/MMI22jiho/deepsort/yolov5/rembg_copy/yolov5s_osnet_ibn_x1_0_MSMT17/crops/person')
# t=0
# for filename in A:
#     os.mkdir("sil/%s"%filename)
#     os.mkdir("tes/%s"%filename)
#     os.mkdir("gait/%s"%filename)
#     B = (os.listdir('/home/MMI22jiho/deepsort/yolov5/rembg_copy/yolov5s_osnet_ibn_x1_0_MSMT17/crops/person/%s' %filename))
#     for length in range(0,len(B)-1):
#         t=1
#         print(t)
#         for file in B[0:t]:
#             ab = str(t)
#             os.system("cp /home/MMI22jiho/deepsort/yolov5/rembg_copy/yolov5s_osnet_ibn_x1_0_MSMT17/crops/person/%s/%s sil/%s/%s" %(filename, file, filename,ab))
#             os.system("rembg p /home/MMI22jiho/deepsort/yolov5/rembg_copy/sil/%s/%s gait/%s" %(filename,ab, filename))
#             t=t+1
#             print(t)


