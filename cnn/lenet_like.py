# ライブラリをインポート
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# DataLoaderの作成
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# ネットワーク構築
from torch import nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
    def forward(self, x):
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # nn.Conv2d(input_channel, out_channel, kernel, stride, padding)
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)     #6@24x24
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3)    #16@10x10
        self.fc1 = nn.Linear(400, 320)
        self.fc2 = nn.Linear(320, 10)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))      #6@12x12
        x = F.relu(F.max_pool2d(self.conv2(x), 2))      #16@5x5
        #print(x.shape)
        x = x.view(-1, 400)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

#デバイスの設定
device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 画像の読み込みと正規化
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X / 255
y = [float(s) for s in y]

# 画像の表示
plt.imshow(X[10].reshape(28,28), cmap=plt.cm.gray)
print("{:.0f}".format(y[10]))

# データの分割
# 学習用のデータ行列, 検証用のデータ行列, 学習用のラベル, 評価用のラベル = train_test_split(データ行列, ラベル, 検証用データの割合, 乱数シード値)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/7, random_state=0)
# nn.Conv2dに入力できるよう行列の形状を変形する
X_train = X_train.reshape((len(X_train), 1, 28, 28)) # (# of samples, channel, height, width)
X_test = X_test.reshape((len(X_test), 1, 28, 28))

# GPUで動作させるためのデータ形式であるTensorを作成
# 基本はnumpyと類似している
X_train = torch.Tensor(X_train)
X_test = torch.Tensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# 同じ要素数の2つのTensorを入力し，TensorDatasetによりペアを作る
ds_train = TensorDataset(X_train, y_train)
ds_test = TensorDataset(X_test, y_test)

# DataLoaderにTensorDatasetとミニバッチサイズを入力し，バッチサイズ分のデータを生成する
loader_train = DataLoader(ds_train, batch_size=64, shuffle=True)
loader_test = DataLoader(ds_test, batch_size=64, shuffle=False)


model=Net()
print(model)

# 誤差関数と最適化手法の設定
from torch import optim
model = Net().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 学習の設定
def train(epoch):
  model.train()
  for data, targets in loader_train:
    #data, targets = data.cuda(), targets.cuda()
    optimizer.zero_grad()
    outputs = model(data)
    loss = loss_fn(outputs, targets)
    loss.backward()
    optimizer.step()
  print("epoch{}:終了\n".format(epoch))
  
  
  # 検証の設定
def test():
  model.eval()
  correct = 0
  with torch.no_grad():
    for data, targets in loader_test:
      #data, targets = data.cuda(), targets.cuda()
      outputs = model(data)
      _, predicted = torch.max(outputs.data, 1)
      correct += predicted.eq(targets.data.view_as(predicted)).sum()
  data_num = len(loader_test.dataset)
  print('\nテストデータの正解率：{}/{}({:.0f}%)\n'.format(correct, data_num, 100. * correct /data_num))
  
  
# 学習前のテストデータ正解率
test()

# 3epoch学習後のテストデータ正解率
for epoch in range(3):
  train(epoch)
test()