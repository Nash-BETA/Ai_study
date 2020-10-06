# 人工知能、機械学習系の勉強用レポジトリ

## 備忘録
### コンストラクタ
下記の二つほぼ同意義

'''python
class Neuron:
    # クラスをインスタンス化する際に呼ばれる
    def __init__(self,input_sum):
        self.input_sum = input_sum

    def set_input(self,inp):
        self.input_sum += inp
        print(self.input_sum)

class NeuralNetwork:
    def __init__(self):
        # インスタンス化
        self.neuron = Neuron(0.0)
'''

'''python
class Neuron:
    # クラスをインスタンス化する際に呼ばれる
    def __init__(self):
        self.input_sum = 0.0

    def set_input(self,inp):
        self.input_sum += inp
        print(self.input_sum)

class NeuralNetwork:
    def __init__(self):
        # インスタンス化
        self.neuron = Neuron()
'''


### 配列とリスト