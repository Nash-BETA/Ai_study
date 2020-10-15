import numpy as np
import matplotlib.pyplot as plt
# サンプルデータセットの読み込み
from sklearn import datasets

iris = datasets.load_iris()
iris_data = iris.data
sl_data = iris_data[:100, 0] # 葉の縦の長さ
sw_data = iris_data[:100, 1] # 葉の額の幅

## 葉の縦の長さ
# numpyを使ってリストの平均値を取得
sl_ave = np.average(sl_data)
# リスト全体の数値に平均値を引く(引くことで平均値の値が0になりそれ以上が正の値、それ以外が負の値になる)
sl_data -= sl_ave

## 葉の縦の長さ
# numpyを使ってリストの平均値を取得
sw_ave = np.average(sw_data)
# リスト全体の数値に平均値を引く(引くことで平均値の値が0になりそれ以上が正の値、それ以外が負の値になる)
sw_data -= sw_ave

# 1枚1枚のリストを作成リストを作成して
input_data = []
for i in range(100):
    input_data.append([sl_data[i],sw_data[i]])

# シグモイド関数
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# ニューロンの設定(各層に配置するニューロンの設定)
class Neuron:
    # ニューロンの入力時と出力時の初期設定
    def __init__(self):
        self.input_sum = 0.0
        self.output = 0.0

    def set_input(self,inp):
        self.input_sum += inp

    def get_output(self):
        self.output = sigmoid(self.input_sum)
        return self.output

    def reset(self):
        self.input_sum = 0
        self.output = 0

class NeuralNetwork:
    def __init__(self):
        # 中間層にはニューロンが2つあり、1つあたり2つ入力(受け取る)される
        self.w_im = [[4.0, 4.0], [4.0, 4.0]]
        # 出力層にはニューロンが1つで入力(受け取る)が2つある状態
        self.w_mo = [[1.0, -1.0]]  # 入力:2 ニューロン数:1

        # バイアス
        # 中間層にはニューロン数が2つなので2つバイアスを入力される
        self.b_m = [2.0, -2.0]
        # 出力層にはニューロン数が1つなので1つバイアスを入力される
        self.b_o = [-0.5]

        ## 各層の宣言
        # 入力層（最初は2つの入力がある）の宣言。初期値の宣言
        self.input_layer = [0.0, 0.0]
        # 中間層(Neuronインスタンスを入力) ニューロンは二つ
        self.middle_layer = [Neuron(), Neuron()]
        # 出力層(Neuronインスタンスを入力) ニューロンは一つ
        self.output_layer = [Neuron()]

    #ニューラルネットワークの実行処理
    def commit(self,input_data):
        self.input_layer[0] = input_data[0]  # 入力層は値を受け取るのみ
        self.input_layer[1] = input_data[1]
        self.middle_layer[0].reset()
        self.middle_layer[1].reset()
        self.output_layer[0].reset()

        # 入力層→中間層
        self.middle_layer[0].set_input(self.input_layer[0] * self.w_im[0][0])
        self.middle_layer[0].set_input(self.input_layer[1] * self.w_im[0][1])
        self.middle_layer[0].set_input(self.b_m[0])

        self.middle_layer[1].set_input(self.input_layer[0] * self.w_im[1][0])
        self.middle_layer[1].set_input(self.input_layer[1] * self.w_im[1][1])
        self.middle_layer[1].set_input(self.b_m[1])

        # 中間層→出力層
        self.output_layer[0].set_input(self.middle_layer[0].get_output() * self.w_mo[0][0])
        self.output_layer[0].set_input(self.middle_layer[1].get_output() * self.w_mo[0][1])
        self.output_layer[0].set_input(self.b_o[0])

        return self.output_layer[0].get_output()

# ニューラルネットワークのインスタンス
neural_network = NeuralNetwork()

# 実行
st_predicted = [[], []]  # Setosa(花の種類)
vc_predicted = [[], []]  # Versicolor(花の種類)
for data in input_data:
    # 出力が0.5より小さければsetosa
    if neural_network.commit(data) < 0.5:
        st_predicted[0].append(data[0]+sl_ave)
        st_predicted[1].append(data[1]+sw_ave)
    # 出力が0.5以上ならばVersicolor
    else:
        vc_predicted[0].append(data[0]+sl_ave)
        vc_predicted[1].append(data[1]+sw_ave)

# 分類結果をグラフ表示
plt.scatter(st_predicted[0], st_predicted[1], label="Setosa")
plt.scatter(vc_predicted[0], vc_predicted[1], label="Versicolor")
plt.legend()

# 葉の長さ
plt.xlabel("Sepal length (cm)")
# 葉の横幅
plt.ylabel("Sepal width (cm)")
plt.title("Predicted")
plt.show()
