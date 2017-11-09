import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

# Hyper Parameters
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.01

# MNIST Dataset
train_dataset = dsets.MNIST(root='./data/mnist',
                            train=True, 
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data/mnist',
                           train=False, 
                           transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=True)

# LSTM Model (Many-to-One)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # num_layers indicates numbers of LSTM unit 
        # batch_first = True indicates input_data with [batch_size, seq_len, input_dim], False indicates [seq_len, batch_size, input_dim]
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Set initial states 
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) 
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        # Forward propagate RNN
        out, _ = self.lstm(x, (h0, c0))  
        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])
        return out

rnn = RNN(input_size, hidden_size, num_layers, num_classes)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
    
# Train the Model
model.train()
for epoch in range(num_epochs):
    for i, train_data in enumerate(train_loader):
        input_x, input_y = train_data
        input_x, input_y = Variable(input_x.view(-1, sequence_length, input_size)), Variable(input_y)        
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = rnn(input_x)
        loss = criterion(outputs, input_y)
        loss.backward()
        optimizer.step()        
        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
                   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))

# Test the Model
correct = 0
total = 0
model.eval()
for i, test_data in enumerate(test_loader):
    test_x, test_y = test_data
    test_x, test_y = Variable(test_x.view(-1, sequence_length, input_size)), Variable(test_y)
    outputs = rnn(test_x)
    predicted = outputs.data.max(dim=1)[1]
    total = total + test_y.size(0)
    correct =  correct + predicted.eq(test_y.data.view_as(predicted)).cpu().sum()
    print("batch_accuracy:{}/{}".format(correct, total))
print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total)) 

# Save the Model
torch.save(rnn.state_dict(), './model/rnn.pkl')
