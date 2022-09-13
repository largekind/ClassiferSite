from django.db import models
from torch import torch, cuda , nn
from PIL import Image
import io , base64

class Photo(models.Model):
  image = models.ImageField(upload_to='photos')
  device = 'cuda' if cuda.is_available() else 'cpu'

  IMAGE_SIZE = 224 #画像サイズ
  MODEL_FILE_PATH = './carbike/ml_models/model.pth'

  #分類するクラス
  classes = ["car","motorbike"]

  def load_model(self):
    device = self.device
    # パラメータ/ラベルの読み込み
    param = torch.load(self.MODEL_FILE_PATH)

    #モデル定義
    from torchvision import models
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
    return model

  #推定処理
  def predict(self):
    print("Start predict")
    # モデルの読み込み
    device = self.device
    model = self.load_model()

    # 画像データの取得
    img_data = self.image.read()
    img_bin = io.BytesIO(img_data)

    #コマンドラインで渡されたパスをテンソル化
    data = Image.open(img_bin).convert("RGB").resize((self.IMAGE_SIZE,self.IMAGE_SIZE))

    from torchvision import transforms
    data = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225]),
    ])(data)
    data = data.unsqueeze_(0).to(device)

    # 予測
    with torch.no_grad():
      outputs = nn.Softmax(dim=1)(model(data))
      pred = self.classes[torch.argmax(outputs)]
      print(pred," ",outputs)

      return pred, outputs
  def image_src(self):
    with self.image.open() as img:
      #画像をBase64の文字列へ変更
      base64_img = base64.b64encode(img.read()).decode()

      return 'data:' + img.file.content_type +';base64,' + base64_img