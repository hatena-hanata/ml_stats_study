from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


class MyData():
    def __init__(self, data_type='clf', class_num=2):
        if data_type == 'clf':
            iris = load_iris()
            target = iris.target
            self.y = target[target!=class_num]
            # 説明変数は2つだけ
            self.X = iris.data[target != 2][:, :2]
        elif data_type == 'reg':
            pass
        
    def get_dataset(self):
        return self.X, self.y

    def plot_data(self):
        X, y = self.get_dataset()
        plt.scatter(X[y==0][:, 0], X[y==0][:, 1], color='red')
        plt.scatter(X[y==1][:, 0], X[y==1][:, 1], color='blue')
        plt.show()
        
    def plot_boundary(self, coef, intercept):
        X, y = self.get_dataset()
        x_min = X[:, 0].min()
        x_max = X[:, 0].max()
        y_min = -1 * coef[0] / coef[1] * x_min - intercept / coef[1]
        y_max = -1 * coef[0] / coef[1] * x_max - intercept / coef[1]
        plt.scatter(X[y==0][:, 0], X[y==0][:, 1], color='red')
        plt.scatter(X[y==1][:, 0], X[y==1][:, 1], color='blue')
        plt.plot([x_min, x_max], [y_min, y_max], color='green')
