class Neuron:
    # クラスをインスタンス化する際に呼ばれる
    def __init__(self,input_sum):
        self.input_sum = input_sum

    def set_input(self,inp):
        # インスタンス変数(内容は保持される)
        self.input_sum += inp
        print(self.input_sum)

class NeuralNetwork:
    def __init__(self):
        # インスタンス化
        self.neuron = Neuron(0.0)

    def commit(self,input_data):
        # input_dataのループ
        for data in input_data:
            # Neuronクラスのset_inputメソッド
            self.neuron.set_input(data)

neural_network = NeuralNetwork()

input_data = [1.0, 2.0, 3.0]
neural_network.commit(input_data)