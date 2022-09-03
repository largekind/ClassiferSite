from PIL import Image
import os , glob
import numpy as np
from sklearn import model_selection

#パラメータ初期化
classes = ["car","motorbike"]
num_classes = len(classes)
image_size = 224

#画像のnumpy配列
X = []
Y = []

for index , classlabel in enumerate(classes):
  photo_dir = "./data/" + classlabel
  files = glob.glob(photo_dir + "/*.jpg")
  for _ , file in enumerate(files):
    image = Image.open(file)

    image = image.convert("RGB") #RGBに統一
    image = image.resize((image_size,image_size))

    data = np.asarray(image).astype('float') #np配列化
    # リストへ追加
    X.append(data)
    Y.append(index)

X = np.array(X)
Y = np.array(Y)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,Y)
xy = (X_train ,X_test , y_train, y_test)

np.save("./data/imagefiles.npy",xy)
