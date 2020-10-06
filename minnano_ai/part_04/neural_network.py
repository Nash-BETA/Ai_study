import numpy as np
import matplotlib.pyplot as plt
# サンプルデータセットの読み込み
from sklearn import datasets

iris = datasets.load_iris()
iris_data = iris.data
sl_data = iris_data[:100, 0] # SetosaとVersicolor、Sepal length
sw_data = iris_data[:100, 1] # SetosaとVersicolor、Sepal width

print(iris_data)