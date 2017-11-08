import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

import numpy as np

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = nn.Linear(784, 520)
        self.l2 = nn.Linear(520, 320)
        self.l3 = nn.Linear(320, 240)
        self.l4 = nn.Linear(240, 120)
        self.l5 = nn.Linear(120, 10)
        #self.relu = F.relu()
        #self.sigmoid = F.sigmoid()
        
    def forward(self, input_x):
        out1 = F.relu(self.l1(input_x))
        out2 = F.relu(self.l2(out1))
        out3 = F.relu(self.l3(out2))
        out4 = F.relu(self.l4(out3))
        out5 = F.relu(self.l5(out4))
        y_pred = F.log_softmax(out5)
        
        return y_pred

if __name__=="__main__":
    batch_size = 64
    epochs = 10
    
    train_loader = DataLoader(dataset=datasets.MNIST('./data/mnist', train=True, transform=transforms.Compose([transforms.ToTensor()]), download=True), batch_size=batch_size, shuffle=True)
    
    test_loader = DataLoader(dataset=datasets.MNIST('./data/mnist', train=False, transform=transforms.Compose([transforms.ToTensor()]), download=True), batch_size=batch_size, shuffle=True)
    
    model = Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    model.train()
    for epoch in range(epochs):
        for i, data in enumerate(train_loader):
            input_x, labels_y = data
            input_x, labels_y = Variable(input_x), Variable(labels_y)
            input_x = input_x.view(-1, 784)
            optimizer.zero_grad()
            output = model(input_x)
            #print(output)
            loss = F.nll_loss(output, labels_y)
            loss.backward()
            # update parameters
            optimizer.step()
            if np.mod(i,100)==0:
                print('train epoch{}[{}/{}]:loss:{:.6f}'.format(epoch, i, len(train_loader.dataset)/batch_size, loss.data[0]))
                

    accuracy = 0
    model.eval()
    for i, test_data in enumerate(test_loader):
        test_x, test_y = test_data
        test_x, test_y = Variable(test_x), Variable(test_y)
        test_x = test_x.view(-1, 784)
        test_out = model(test_x)
        
        #total_loss = F.nll_loss(test_out, test_y)
        pred = test_out.data.max(dim=1)[1]
        accuracy = accuracy + pred.eq(test_y.data.view_as(pred)).cpu().sum()
        
        print('accuracy:{}'.format(accuracy))
    print('accuracy rate:{}/{}'.format(accuracy,len(test_loader.dataset)))
