import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
iris_data = iris.data
#2つの花の葉の長さ
sl_data = iris_data[:100, 0]
#2つの花の葉の横幅
sw_data = iris_data[:100, 1]

#平均値を取得して平均値で引くことで0＝平均値にする
sl_ave = np.average(sl_data)
sl_data -= sl_ave

sw_ave = np.average(sw_data)
sw_data -= sw_ave

input_data = []
for i in range(100):  # iには0から99までが入る
    input_data.append([sl_data[i], sw_data[i]])

#シグモイド関数の定義
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

#ニューロンの初期設定
class Neuron:
    def __init__(self):
        self.input_sum = 0.0
        self.output = 0.0

    # ニューロンに値を渡す時に使用
    def set_input(self, inp):
        self.input_sum += inp

    # ニューロンから値を出力する時に使用(入力層ではその値を中間層に渡すだけなので使われない)
    # 渡された値をシグモイド関数を当ててるだけ
    def get_output(self):
        #出力時にシグモイド関数を用いて0から1の値にする
        self.output = sigmoid(self.input_sum)
        return self.output

    def reset(self):
        self.input_sum = 0
        self.output = 0

class NeuralNetwork:
    def __init__(self):
        # 中間層にはニューロンが3つあり、1つあたり2つ入力(受け取る)される
        self.n_mid = [[4.0,4.0],[4.0,4.0],[4.0,4.0]]
        # 出力層にはニューロンが1つあり、1つあたり3つ入力(受け取る)される
        self.n_end = [[1.0, -1.0, 1.0]]

        # 各層のバイアス
        self.b_mid = [3.0,0.0,-3.0]
        self.b_end = [-0.5]

        # 各層の宣言
        self.input_layer = [0.0, 0.0]
        self.middle_layer = [Neuron(), Neuron(), Neuron()]
        self.output_layer = [Neuron()]

    def commit(self,input_data):
        self.input_layer[0] = input_data[0]
        self.input_layer[1] = input_data[1]

        self.middle_layer[0].reset()
        self.middle_layer[1].reset()
        self.middle_layer[2].reset()

        self.output_layer[0].reset()

        self.middle_layer[0].set_input(self.input_layer[0] * self.n_mid[0][0])
        self.middle_layer[0].set_input(self.input_layer[1] * self.n_mid[0][1])
        self.middle_layer[0].set_input(self.b_mid[0])

        self.middle_layer[1].set_input(self.input_layer[0] * self.n_mid[1][0])
        self.middle_layer[1].set_input(self.input_layer[1] * self.n_mid[1][1])
        self.middle_layer[1].set_input(self.b_mid[1])

        self.middle_layer[2].set_input(self.input_layer[0] * self.n_mid[2][0])
        self.middle_layer[2].set_input(self.input_layer[1] * self.n_mid[2][1])
        self.middle_layer[2].set_input(self.b_mid[2])

        # 中間層→出力層
        self.output_layer[0].set_input(self.middle_layer[0].get_output() * self.n_end[0][0])
        self.output_layer[0].set_input(self.middle_layer[1].get_output() * self.n_end[0][1])
        self.output_layer[0].set_input(self.middle_layer[2].get_output() * self.n_end[0][2])
        self.output_layer[0].set_input(self.b_end[0])

        return self.output_layer[0].get_output()

# ニューラルネットワークのインスタンス
neural_network = NeuralNetwork()

# 実行
st_predicted = [[], []]  # Setosa
vc_predicted = [[], []]  # Versicolor
for data in input_data:
    if neural_network.commit(data) < 0.5:
        st_predicted[0].append(data[0]+sl_ave)
        st_predicted[1].append(data[1]+sw_ave)
    else:
        vc_predicted[0].append(data[0]+sl_ave)
        vc_predicted[1].append(data[1]+sw_ave)

# 分類結果をグラフ表示
plt.scatter(st_predicted[0], st_predicted[1], label="Setosa")
plt.scatter(vc_predicted[0], vc_predicted[1], label="Versicolor")
plt.legend()

plt.xlabel("Sepal length (cm)")
plt.ylabel("Sepal width (cm)")
plt.title("Predicted")
plt.show()