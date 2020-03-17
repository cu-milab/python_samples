## PyTorchによる手書き数字認識のサンプルコード
PyTorchによる手書き数字認識のサンプルコードです．

### 手書き数字画像データセットMNIST
MNIST(Mixed National Institute of Standards and Technology database)とは，手書き数字画像60,000枚とテスト画像10,000枚を集めた画像データセットです．さらに，手書きの数字「0〜9」に正解ラベルも用意されています．画像認識のチュートリアルで使われることが多いデータセットです． 詳細は[こちら](http://yann.lecun.com/exdb/mnist/)をご覧ください．

### 実習
今回は畳み込みニューラルネットワークの基礎と呼ばれるLeNet[1]に近いネットワークにより手書き数字画像を分類します．


まずはライブラリをインポートします．
```
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
```

次にデバイスの設定をします．
今回はCPUを使用しますが，GPUを使用する場合にはコメントを解除してください．
```
#デバイスの設定
device = "cpu"

#GPUを使用する場合は下記を実行
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

まずは手書き数字画像を読み込み，画素値を0から1の範囲に正規化します．
また，ラベルの値を実数にキャストしています．
```
# 画像の読み込みと正規化
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X / 255
y = [float(s) for s in y]
```

試しに画像を表示しましょう．
今回はMatplotlibライブラリのimshow関数によって画像を表示します．
ここでは10番目の画像を表示します．
他の画像を表示したい場合にはXのインデックス番号を任意の数値に変えてください．

```
# 画像の表示
plt.imshow(X[10].reshape(28,28), cmap=plt.cm.gray)
print("{:.0f}".format(y[10]))
```

PyTorchで学習するためには画像等のデータをDataLoaderという形式に変換する必要があります．
DataLoaderは基本的にはNumPyのndarray配列と類似した働きをしますが，異なる点はGPUを用いて演算できる点です．
手書き文字画像データベースであるMNISTのデータを学習用と検証用に分割し，それぞれをDataLoaderの形式に変換します．

```
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
```

次にネットワークを定義します．
ネットワークは，入力層1層，畳み込み層とプーリング層を2回繰り返し，その後全結合層2層を経て出力層1層とします．
1回目の畳み込み層は，入力チャンネルが1，カーネル数が6，カーネルサイズが5x5と定義し，畳み込みを行った後に活性化関数ReLuに入力されます．
そして計算結果は2x2のプーリング層により処理されます．
今回はプーリング方法としてマックスプーリングを採用しています．
2回目の畳み込み層は，入力チャンネルが6，カーネル数が16，カーネルサイズが3x3，活性化関数はReLu，プーリング層は2x2のマックスプーリングと定義します．

畳み込みとプーリングを2回繰り返した後に全結合層に入力します．
今回は全結合層(Full Connected = fc)は2層用意してます．
1層目の全結合層は入力ユニットが400，出力ユニットが320，2層目の全結合層は入力ユニットが320，出力ユニットが10です．
2層目の出力ユニットは分類クラス数と同じ数にする必要があるので注意してください．
全結合層を経た後に

```
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # nn.Conv2d(input_channel, out_channel, kernel, stride, padding)
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)     #1@28x28 => 6@24x24
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3)    #6@12x12 => 16@10x10
        self.fc1 = nn.Linear(400, 320)                  #16 x 5 x 5 = 400 => 320
        self.fc2 = nn.Linear(320, 10)                   #320 => 10
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)      #6@24x24 => 6@12x12
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)      #16@10x10 => 16@5x5
        #print(x.shape)
        x = x.view(-1, 400)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model=Net()
print(model)
```

誤差関数と最適化手法を設定します
誤差関数はクロスエントロピー，最適化手法はAdamを使用します．
```
# 誤差関数と最適化手法の設定
from torch import optim
model = Net().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
```

学習1回あたりの処理を設定します．
```
# 学習の設定
def train(epoch):
  model.train()
  for data, targets in loader_train:
    data, targets = data.cuda(), targets.cuda()
    optimizer.zero_grad()
    outputs = model(data)
    loss = loss_fn(outputs, targets)
    loss.backward()
    optimizer.step()
  print("epoch{}:終了\n".format(epoch))
```

モデルにデータを与えて識別した結果の精度を確認する関数を定義します．
```
# 検証の設定
def test():
  model.eval()
  correct = 0
  with torch.no_grad():
    for data, targets in loader_test:
      data, targets = data.cuda(), targets.cuda()
      outputs = model(data)
      _, predicted = torch.max(outputs.data, 1)
      correct += predicted.eq(targets.data.view_as(predicted)).sum()
  data_num = len(loader_test.dataset)
  print('\nテストデータの正解率：{}/{}({:.0f}%)\n'.format(correct, data_num, 100. * correct /data_num))
```

試しに学習前の正解率を確認します．
```
# 学習前のテストデータ正解率
test()
```
10000個のデータにおいて663個のデータが正解，正解率は7%と表示されました．
なお，畳み込みニューラルネットワークの初期の重みは乱数であるため，
学習する度に正解率は変わります．
次に3epoch学習して正解率を確認しましょう．
```
# 3epoch学習後のテストデータ正解率
for epoch in range(3):
  train(epoch)
test()
```
10,000個のデータにおいて9,819個のデータが正解，正解率は98%と表示されました．

### 参考文献
[1] Y. LeCun et.al, "Gradient-Based Learning Applied to Document Recognition", IEEE, 1998.
