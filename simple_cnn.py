import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

import numpy as np

class LeNet_5(nn.Module):
    def __init__(self):
        super(LeNet_5, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, stride=1)
        self.max_pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, stride=1)
        self.max_pool2 = nn.MaxPool2d(2)
        #print(self.max_pool2.size)
        #in_size = self.max_pool2.shape(1)*self.max_pool2.size(2)*self.max_pool2.size(3)
        
        self.l1 = nn.Linear(20*4*4, 10)
        
    def forward(self, input_x):
        in_size = input_x.size(0)
        out1 = self.max_pool1(F.relu(self.conv1(input_x)))
        out2 = self.max_pool2(F.relu(self.conv2(out1)))
        out2 = out2.view(in_size, -1)
        out3 = F.relu(self.l1(out2))
        pred = F.log_softmax(out3)
        return pred

if __name__=="__main__":
    batch_size = 64
    epochs = 10
    
    train_loader = DataLoader(dataset=datasets.MNIST('./data/mnist', train=True, transform=transforms.Compose([transforms.ToTensor()]), download=True), batch_size=batch_size, shuffle=True)
    
    test_loader = DataLoader(dataset=datasets.MNIST('./data/mnist', train=False, transform=transforms.Compose([transforms.ToTensor()]), download=True), batch_size=batch_size, shuffle=True)
    
    model = LeNet_5()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    # training
    model.train()
    for epoch in range(epochs):
        for i, data in enumerate(train_loader):
            input_x, input_y = data
            input_x, input_y = Variable(input_x), Variable(input_y)
            
            output = model(input_x)
            loss = F.nll_loss(output, input_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if np.mod(i, 100)==0:
                print("epoch{}:{}/{}: loss:{},".format(epoch, i, len(train_loader.dataset)/batch_size, loss.data[0]))

    # testing
    model.eval()
    accuracy = 0
    for i, test_data in enumerate(test_loader):
        test_x, test_y = test_data
        test_x, test_y = Variable(test_x), Variable(test_y)
        
        test_out = model(test_x)
        pred = test_out.data.max(dim=1)[1]
        batch_accuracy = pred.eq(test_y.data.view_as(pred)).cpu().sum()
        accuracy = accuracy + batch_accuracy
        print("batch_accuracy:{}/{}".format(batch_accuracy,batch_size))
        
    print("total_accuracy:{}/{}".format(accuracy,len(test_loader.dataset)))
