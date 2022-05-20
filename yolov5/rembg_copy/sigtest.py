import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.utils.data import DataLoader 
import torchvision 
import torchvision.transforms as transforms
import cv2
import math
from moviepy.editor import *
import os
import shutil
os.environ['MKL_THREADING_LAYER'] = 'GNU'

trans = transforms.Compose([ transforms.ToTensor() ])
a=0
b=0
c=0
d=0
e=0
f=0
g=0
h=0
k=0
p=0
o=0
oa=0
ob=0
oc=0
od=0
oe=0
of=0
device = 'cuda' 
#if torch.cuda.is_available() else 'cpu' 
torch.manual_seed(777) 
if device == 'cuda': 
  torch.cuda.manual_seed_all(777)
if len(os.listdir('GEIext/A/test')) >0:  
    test_data = torchvision.datasets.ImageFolder(root = "GEIext/A", transform = trans) 

    test_set = DataLoader(dataset = test_data, batch_size = 1) 

    class CNN(nn.Module): 
        def __init__(self): 
            super(CNN, self).__init__() 

            self.conv1 = nn.Conv2d(3,18,7, stride =1)
            self.pool = nn.MaxPool2d(2,2)
            self.conv2 = nn.Conv2d(18, 45, 5, stride=1, padding=2)
            self.fc1 = nn.Linear(45*20*30,1024)   
            self.fc2 = nn.Linear(1024,512)
            self.fc3 = nn.Linear(512,100)
            self.fc4 = nn.Linear(100,30)
            self.fc5 = nn.Linear(30,4)

        def num_flat_features(self,x):
            size = x.size()[1:]
            num_features=1
            for s in size:
              num_features *= s

            return num_features

        def forward(self,x):

         x=self.pool(F.relu(self.conv1(x)))
         x=self.pool(F.relu(self.conv2(x)))
         x=x.view(-1,self.num_flat_features(x))
         x=F.dropout((self.fc1(x)))
         # x=self.fc2(x)
         x=F.dropout((self.fc2(x)))
         x=(F.relu(self.fc3(x)))
         x=(F.relu(self.fc4(x)))
         x=F.softmax(self.fc5(x), dim=1)
         return x

    net = CNN().to(device) 
    net = torch.load('A_299.pt')
    net.eval()
    if torch.cuda.is_available():
        net.cuda()

    with torch.no_grad(): 
        for num, data in enumerate(test_set): 
            imgs, label = data 
            imgs = imgs.to(device) 
            label = label.to(device) 
            prediction = net(imgs) 
            correct_prediction = torch.argmax(prediction, 1)
            if (label) == 0:
                if correct_prediction == 0:
                    a=a+1
                if correct_prediction == 1:
                    b= b+1
                if correct_prediction == 2:
                    c= c+1
                if correct_prediction == 3:
                    d= d+1
            if label == 1:
                if correct_prediction == 0:
                    e = e+1
                if correct_prediction == 1:
                    f = f+1
                if correct_prediction == 2:
                    g= g+1
                if correct_prediction == 3:
                    h = h+1
            if label == 2:
                if correct_prediction ==0:
                    k= k+1
                if correct_prediction ==1:
                    p= p+1
                if correct_prediction ==2:
                    o = o+1
                if correct_prediction ==3:
                    oa = oa+1
            if label == 3:
                if correct_prediction ==0:
                    ob= ob+1
                if correct_prediction ==1:
                    oc= oc+1
                if correct_prediction ==2:
                    od = od+1
                if correct_prediction ==3:
                    oe = oe+1

    # with torch.no_grad(): 
    #     for num, data in enumerate(test_set): 
    #         imgs, label = data 
    #         imgs = imgs.to(device) 
    #         label = label.to(device) 
    #         prediction = net(imgs) 
    #         correct_prediction = torch.argmax(prediction, 1)
    #       # if correct_prediction == label:
    #       #   a = a+1
    #         if (correct_prediction) == 0:
    #              a= a+1
    #         if (correct_prediction) == 1:
    #              b= b+1
    #         if (correct_prediction) == 2:
    #              c= c+1
    #         if (correct_prediction) == 3:
    #              d= d+1
            
# print("real child male:{}".format(a+b+c+d))
# print("predict: child male:{}, child female:{}, adult male:{}, adult female:{}".format(a,b,c,d))
# print("real child female:{}".format(e+f+g+h))
# print("predict: child male:{}, child female:{}, adult male:{}, adult female:{}".format(e,f,g,h))
# print("real adult male: {}".format(k+p+o+oa))
# print("predict: child male:{}, child female:{}, adult male:{}, adult female:{}".format(k,p,o,oa))
# print("real child {}".format(ob+oc+od+oe))
# print("predict: child male:{}, child female:{}, adult male:{}, adult female:{}".format(ob,oc,od,oe))

# print("child male:{}".format(a))
# print("child female:{}".format(b))
# print("adult male:{}".format(c))
# print("adult female:{}".format(d))


print("child:{}, adult:{}".format(a+b+e+f+k+p+ob+oc,c+d+g+h+o+oa+od+oe))
A=os.listdir("/home/MMI22jiho/videocut/")
for filename in A:
  if (a+b+e+f+k+p+ob+oc!=0):

    videoclip = VideoFileClip("/home/MMI22jiho/videocut/%s" %filename)
    audioclip = AudioFileClip("/home/MMI22jiho/child.mp3")
    videoclip.audio = audioclip
    videoclip.write_videofile("/home/MMI22jiho/videoclip/%s" %filename)

  else:
    if(c+d+g+h+o+oa+od+oe!=0):

      videoclip = VideoFileClip("/home/MMI22jiho/videocut/%s" %filename)
      audioclip = AudioFileClip("/home/MMI22jiho/adult.mp3")
      videoclip.audio = audioclip
      videoclip.write_videofile("/home/MMI22jiho/videoclip/%s" %filename)
    else:
      videoclip = VideoFileClip("/home/MMI22jiho/videocut/%s" %filename)
      videoclip.write_videofile("/home/MMI22jiho/videoclip/%s" %filename)