from torchvision import transforms , models
from torchinfo import summary
from torch import torch, cuda , nn
import sys
from PIL import Image
import numpy as np

# GPUの確認
device = 'cuda' if cuda.is_available() else 'cpu'
print("Use of",device)
# パラメータ/ラベルの読み込み
param = torch.load('model.pth')
classes = ["car","motorbike"]
#モデル定義
model = models.efficientnet_b7(pretrained=True).to(device)
# 分類部分の構造を再作成
model.classifier = torch.nn.Sequential(
                        torch.nn.Dropout(p=0.2, inplace=True),
                        torch.nn.Linear(in_features=2560,
                        out_features=2,
                        bias=True)
                        ).to(device)
# 重みの再読み込み
model.load_state_dict(param)
# 評価モードにする
model = model.eval()

#コマンドラインで渡されたパスをテンソル化
data = Image.open(sys.argv[1]).convert("RGB")
data = transforms.Compose([transforms.ToTensor(),
                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]),
  ])(data)
data = data.unsqueeze_(0).to(device)
print(data.shape)

# 予測
with torch.no_grad():
  outputs = nn.Softmax(dim=1)(model(data))
  pred = classes[torch.argmax(outputs)]
  print(pred," ",outputs)
