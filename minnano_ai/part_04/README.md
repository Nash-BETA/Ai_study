# 手順書(流れ)

## ライブラリ及び、データのインポート
### numpy
    数値計算を効率的に行うライブラリ

### matplotlib
    グラフ描写ができるライブラリ

### sklearn
    葉のデータ

## データの取得
```ex
from sklearn import datasets

iris = datasets.load_iris()
iris_data = iris.data
```

## 数値の整理
 + 平均値の算出及び、平均値を中央に

## 生成した値をリストに収める

## シグモイド関数の作成
 + 入力された値を0.0から1.0の範囲で数値に変換し出力する
 + S字の曲線を作る

## 数式

$f(x)=\frac{1}{1 + e^{-x}}  (a>0)$
```pythonで表現
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))
```
#### 補足
np.exp(x) はe の x 乗を返す

pythonでのべき乗は「**」で表すがnumpyの関数で簡単に作れる
通常の例（5 ** x）→5のX乗


## ニューロンの初期設定
### init(コンストラクト)
 + ニューロンへの入力時と出力時の初期設定を行う

###