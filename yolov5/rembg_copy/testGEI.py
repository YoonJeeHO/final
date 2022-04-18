import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.utils.data import DataLoader 
import torchvision 
import torchvision.transforms as transforms



trans = transforms.Compose([ transforms.ToTensor() ])

device = 'cuda' 
#if torch.cuda.is_available() else 'cpu' 
torch.manual_seed(777) 
if device == 'cuda': 
  torch.cuda.manual_seed_all(777)
test_data = torchvision.datasets.ImageFolder(root = "GEIext/A", transform = trans) 

test_set = DataLoader(dataset = test_data, batch_size = 1) 

class CNN(nn.Module): 
  def __init__(self): 
    super(CNN, self).__init__() 

    self.conv1 = nn.Conv2d(3,18,7, stride =1)
    self.conv2 = nn.Conv2d(18, 45, 5, stride=1, padding=2)
    self.fc1 = nn.Linear(45*20*30,1024)
    self.fc2 = nn.Linear(1024,3)

  def num_flat_features(self,x):
    size = x.size()[1:]
    num_features=1
    for s in size:
      num_features *= s

    return num_features

  def forward(self,x):

    x=F.max_pool2d(F.relu(self.conv1(x)),(2,2))
    x = F.max_pool2d(F.relu(self.conv2(x)),(2,2))
    x=x.view(-1,self.num_flat_features(x))
    x=F.dropout(self.fc1(x))
    x=F.softmax(self.fc2(x),dim=1)
    return x
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()

#         #input = 3, output = 6, kernal = 5
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         #kernal = 2, stride = 2, padding = 0 (default)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         #input feature, output feature
#         self.fc1 = nn.Linear(16 * 19 * 29, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 3)

#     # 값 계산
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 19 * 29)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

net = CNN().to(device) 
net = torch.load('model_600_99.pt')
net.eval()
if torch.cuda.is_available():
    net.cuda()
a=0
b=0
c=0
d=0
e=0
f=0
g=0
h=0
k=0
with torch.no_grad(): 
    for num, data in enumerate(test_set): 
        imgs, label = data 
        imgs = imgs.to(device) 
        label = label.to(device) 
        prediction = net(imgs) 
        correct_prediction = torch.argmax(prediction, 1)
        # if correct_prediction == label:
        #   a = a+1
        if (correct_prediction) == 0:
            a= a+1
        if (correct_prediction) == 1:
            b= b+1
        if (correct_prediction) == 2:
            c= c+1

print("{} Pedestrian is child".format(a))
print("{} Pedestrian is adult".format(b))
print("{} Pedestrian is old".format(c))







# with torch.no_grad(): 
#     for num, data in enumerate(test_set): 
#         imgs, label = data 
#         imgs = imgs.to(device) 
#         label = label.to(device) 
#         prediction = net(imgs) 
#         correct_prediction = torch.argmax(prediction, 1)
#         if (label) == 0:
#             if correct_prediction == 0:
#               a=a+1
#             if correct_prediction == 1:
#               b= b+1
#             if correct_prediction == 2:
#               c= c+1
#         if label == 1:
#             if correct_prediction == 0:
#               d = d+1
#             if correct_prediction == 1:
#               e = e+1
#             if correct_prediction == 2:
#               f= f+1
#         if label == 2:
#             if correct_prediction ==0:
#               g= g+1
#             if correct_prediction ==1:
#               h= h+1
#             if correct_prediction ==2:
#               k = k+1
# print("real child {}".format(a+b+c))
# print("predict: child {}, Adult {}, Old {}".format(a,b,c))
# print("real Adult {}".format(d+e+f))
# print("predict: child {}, Adult {}, Old {}".format(d,e,f))
# print("real Old {}".format(g+h+k))
# print("predict: child {}, Adult {}, Old {}".format(g,h,k))