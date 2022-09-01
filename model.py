from torch import nn
from torchinfo import summary

class ClassiferCNN(nn.Module):
  def __init__(self):
    super(ClassiferCNN, self).__init__()

    self.conv1 = nn.Conv2d(3,32,kernel_size=3,stride=1)
    self.conv2 = nn.Conv2d(32,32,kernel_size=3,stride=1)
    self.relu = nn.ReLU()
    self.maxpool = nn.MaxPool2d(2,2)
    self.dropout = nn.Dropout(0.25)
    self.linear1 = nn.Linear(32*73*73, 256)
    self.out = nn.Linear(256 , 2)
  def forward(self,x):
    x = self.conv1(x)
    x = self.relu(x)
    x = self.conv2(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.dropout(x)
    x = x.view(-1, 32 * 73 * 73)
    x = self.linear1(x)
    x = self.dropout(x)
    x = self.out(x)
    return x

if __name__ == "__main__":
  model = ClassiferCNN()
  summary(model,input_size=(1,3,150,150))