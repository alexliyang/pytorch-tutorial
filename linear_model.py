import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

class DiabetesDataset(Dataset):
    def __init__(self):
        xy = np.loadtxt('data.csv', delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:,0:-1])
        self.y_data = torch.from_numpy(xy[:,-1])

    def __getitem__(self,index):
        return self.x_data[index], self.y_data[index]
        
    def __len__(self):
        return self.len

class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.l1 = torch.nn.Linear(8,6)
        self.l2 = torch.nn.Linear(6,4)
        self.l3 = torch.nn.Linear(4,1)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, input_x):
        out1 = self.sigmoid(self.l1(input_x))
        out2 = self.sigmoid(self.l2(out1))
        out3 = self.sigmoid(self.l3(out2))
        y_pred = out3
        return y_pred

if __name__=="__main__":
    model = Model()
    dataset = DiabetesDataset()
    train_loader = DataLoader(dataset=dataset,
                              batch_size=2,
                              shuffle=True,
                              num_workers=2)
    
    cross_entropy = torch.nn.BCELoss(size_average=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(10):
        for i, data in enumerate(train_loader,0):
            input_x, labels_y = data
            input_x, labels_y = Variable(input_x), Variable(labels_y.float())
            y_pred = model(input_x)
            # calculate loss
            loss = cross_entropy(y_pred, labels_y)
            print('loss:',i,loss.data)
            # zero the grad
            optimizer.zero_grad()
            # calculate gradients
            loss.backward()
            # update parameters
            optimizer.step()
