from pickletools import optimize
from tokenize import Double
import numpy as np
from torch import torch, nn , optim , cuda
from model import ClassiferCNN
from dataset import ClassDataset
from torch.utils.data import DataLoader

def main():
  classes = ["car","motorbike"]
  num_classes = len(classes)
  image_size = 150
  batch_size = 8
  epochs = 10

  #npyから各データセットを作成
  X_train, X_test, y_train, y_test = np.load("./data/imagefiles.npy",allow_pickle=True)
  dataset_train = ClassDataset(X_train,y_train)
  dataset_test = ClassDataset(X_test,y_test)

  train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,num_workers=2)
  test_loader  = DataLoader(dataset_test, batch_size=batch_size, shuffle=True,num_workers=2)
  # GPUの確認
  device = 'cuda' if cuda.is_available() else 'cpu'
  print("Use of",device)

  #モデル定義
  model = ClassiferCNN().double().to(device)

  # optimizer/criterion
  optimizer = optim.SGD(model.parameters(),lr=0.1)
  criterion = nn.CrossEntropyLoss().to(device)

  # 学習
  train_loss_list = []
  train_acc_list = []
  val_loss_list = []
  val_acc_list = []
  train_loss = 0
  train_acc = 0
  val_loss = 0
  val_acc = 0

  for epoch in range(epochs):
    print("Epoch: {}".format(epoch+1))

    model.train() #学習モードセット
    for i , (images, labels) in enumerate(train_loader):
      images = images.to(device)
      labels = labels.to(device)
      optimizer.zero_grad() #勾配初期化

      #モデルに渡してloss計算
      outputs = model(images)
      loss = criterion(outputs, labels)

      train_loss += loss.item()
      train_acc += (outputs.max(1)[1] == labels).sum().item()
      #逆伝搬 + 最適化
      loss.backward()
      optimizer.step()
    # スコア平均値計算
    avg_train_loss = train_loss / len(train_loader.dataset)
    avg_train_acc  = train_acc  / len(train_loader.dataset)

    #テスト
    model.eval()
    with torch.no_grad():
      for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        val_loss += loss.item()
        val_acc += (outputs.max(1)[1] == labels).sum().item()
    avg_val_loss = val_loss / len(test_loader.dataset)
    avg_val_acc = val_acc / len(test_loader.dataset)

    print ('Epoch [{}/{}], loss: {loss:.4f} val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}'
                   .format(epoch+1, epochs, i+1, loss=avg_train_loss, val_loss=avg_val_loss, val_acc=avg_val_acc))
    train_loss_list.append(avg_train_loss)
    train_acc_list.append(avg_train_acc)
    val_loss_list.append(avg_val_loss)
    val_acc_list.append(avg_val_acc)

# 1 hot vecをクラスで分割する np_utils.to_categorialのpytorch版処理
def to_categorical(y, num_classes):
    return np.eye(num_classes, dtype='uint8')[y]

if __name__ =="__main__":
  main()