# import the library opencv
import cv2
# globbing utility.
import glob
# select the path
import os

path = "tes/*.png"
for bb,file in enumerate (glob.glob(path)):
  image_read = cv2.imread(file)
  # conversion numpy array into rgb image to show 
  c = cv2.cvtColor(image_read, cv2.COLOR_BGR2GRAY)
  # writing the images in a folder output_images

  height, width = c.shape
  for j in range(height):
      for i in range(width):
            if c[j,i]>1:
                c[j,i]=255
            else:
                c[j,i]=0

  cv2.imwrite('silh/{}.png'.format(bb), c)


  