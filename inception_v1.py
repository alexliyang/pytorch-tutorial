import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

class Inception(nn.Module):
    def __init__(self, input_channels):
        self.input_channels = input_channels
        super(Inception, self).__init__()
        self.branch1_1_1 = nn.Conv2d(self.input_channels, 16, kernel_size=1)
        
        self.branch2_1_1 = nn.Conv2d(self.input_channels, 16, kernel_size=1)
        self.branch2_5_5 = nn.Conv2d(16, 24, kernel_size=5, padding=2)
        
        self.branch3_1_1 = nn.Conv2d(self.input_channels, 16, kernel_size=1)
        self.branch3_3_3_1 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3_3_3_2 = nn.Conv2d(24, 24, kernel_size=3, padding=1)
        
        self.branch4_1_1 = nn.Conv2d(self.input_channels, 24, kernel_size=1)
        
    def forward(self, input_x):
        branch1 = self.branch1_1_1(input_x)
        
        branch2 = self.branch2_5_5(self.branch2_1_1(input_x))
        
        branch3 = self.branch3_3_3_2(self.branch3_3_3_1(self.branch3_1_1(input_x)))
        
        branch4 = self.branch4_1_1(F.avg_pool2d(input_x, kernel_size=3, stride=1, padding=1))
        
        outputs = [branch1, branch2, branch3, branch4]
        
        return torch.cat(outputs, 1)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.inception1 = Inception(10)# output 88 channels
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5)
        self.inception2 = Inception(20)
        
        self.max_pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(1408, 10)
                
    def forward(self, input_x):
        in_size = input_x.size(0)
        # construct neural networks
        conv1 = self.max_pool(F.relu(self.conv1(input_x)))
        inception1 = self.inception1(conv1)      
        conv2 = self.max_pool(F.relu(self.conv2(inception1)))
        inception2 = self.inception2(conv2)
        x = inception2.view(in_size, -1)
        fc = self.fc(x)
        
        return F.log_softmax(fc)

if __name__=="__main__":
    batch_size = 64
    epochs = 10
    
    train_loader = DataLoader(dataset=datasets.MNIST('./data/mnist', train=True, transform=transforms.Compose([transforms.ToTensor()]), download=True), batch_size=batch_size, shuffle=True)
    
    test_loader = DataLoader(dataset=datasets.MNIST('./data/mnist', train=False, transform=transforms.Compose([transforms.ToTensor()]), download=True), batch_size=batch_size, shuffle=True)
    
    model = Net()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    
    model.train()
    for epoch in range(epochs):
        for i, train_data in enumerate(train_loader):
            train_x, train_y = train_data
            train_x, train_y = Variable(train_x), Variable(train_y)
            
            optimizer.zero_grad()
            output = model(train_x)
            loss = F.nll_loss(output, train_y)
            loss.backward()
            optimizer.step()
            
            if np.mod(i,100)==0:
                print("epoch{}[{}/{}]:loss:{:.6f}".format(epoch, i, len(train_loader.dataset)/batch_size, loss.data[0]))
    
    model.eval()
    accuracy = 0
    for i, test_data in enumerate(test_loader):
        test_x, test_y = test_data
        test_x, test_y = Variable(test_x), Variable(test_y)
        
        outputs = model(test_x)
        pred = outputs.data.max(dim=1)[1]
        batch_accuracy = pred.eq(test_y.data.view_as(pred)).cpu().sum()
        accuracy = accuracy + batch_accuracy
        print("batch_accuracy:{}/{}".format(batch_accuracy, batch_size))
    print("accuracy:{}/{}".format(accuracy, len(test_loader.dataset)))
