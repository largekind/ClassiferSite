from pickletools import optimize
from tokenize import Double
import numpy as np
from torch import torch, nn , optim , cuda
from model import ClassiferCNN
from dataset import ClassDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms , models
from torchvision.datasets import ImageFolder
from torchinfo import summary

def main():
  classes = ["car","motorbike"]
  num_classes = len(classes)
  image_size = 150
  batch_size = 8
  epochs = 20
  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
  ])

  #npyから各データセットを作成
  #X_train, X_test, y_train, y_test = np.load("./data/imagefiles.npy",allow_pickle=True)
  images = ImageFolder( "./data", transform = transform)
  #dataset_train = ClassDataset(X_train,y_train,transform)
  #dataset_test = ClassDataset(X_test,y_test,transform)
  # 学習データ、検証データに 8:2 の割合で分割する。
  train_size = int(0.8 * len(images))
  val_size = len(images) - train_size
  dataset_train, dataset_test = torch.utils.data.random_split(
      images, [train_size, val_size]
  )
  # データローダの設定
  train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,num_workers=2)
  test_loader  = DataLoader(dataset_test, batch_size=batch_size, shuffle=True,num_workers=2)
  # GPUの確認
  device = 'cuda' if cuda.is_available() else 'cpu'
  print("Use of",device)

  #モデル定義
  model = models.efficientnet_b7(pretrained=True).to(device)
  #転移学習用に値を固定
  for param in model.features.parameters():
    param.requires_grad = False
  # 分類部分の構造を再作成
  model.classifier = torch.nn.Sequential(
                            torch.nn.Dropout(p=0.2, inplace=True),
                            torch.nn.Linear(in_features=2560,
                            out_features=2,
                            bias=True)
                            ).to(device)
  # モデルの表示
  print(summary(model,
        input_size=(32, 3, 224, 224),
        verbose=0,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
  ))
  # optimizer/criterion
  optimizer = optim.Adam(model.parameters(),lr=0.1)
  criterion = nn.CrossEntropyLoss().to(device)
  # 学習
  train_loss_list = []
  train_acc_list = []
  val_loss_list = []
  val_acc_list = []
  for epoch in range(epochs):
    print("Epoch: {}".format(epoch+1))
    train_loss = 0
    train_acc = 0
    val_loss = 0
    val_acc = 0

    model.train() #学習モードセット
    for i , (images, labels) in enumerate(tqdm(train_loader)):
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
  #モデル保存
  model_path = 'model.pth'
  torch.save(model.state_dict(), model_path)

# 1 hot vecをクラスで分割する np_utils.to_categorialのpytorch版処理
def to_categorical(y, num_classes):
    return np.eye(num_classes, dtype='uint8')[y]

if __name__ =="__main__":
  main()