class Neuron:
    def __init__(self):
        self.input_sum = 0.0
        self.output = 0.0

    def set_input(self,inp):
        self.input_sum += inp

    def get_output(self):
        # set_inputで加算されたself.input_sumをoutputに代入して返してる
        self.output = self.input_sum
        return self.output

class NeuralNetwork:
    def __init__(self):
        self.neuron = Neuron()

    def commit(self, input_data):
        for data in input_data:
            self.neuron.set_input(data)
        return self.neuron.get_output()

# ニューラルネットワークのインスタンス
neural_network = NeuralNetwork()

# 実行
input_data = [1.0, 2.0, 3.0]
print(neural_network.commit(input_data))