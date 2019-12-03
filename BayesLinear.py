import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

class bayesLR:
    def __init__(self, path):
        self.path = path

    def loadata(self):
        data = pd.read_csv(self.path, delimiter='\t', header=None,)
        X = data.iloc[:, 0].values
        t = data.iloc[:, -1].values
        return X.reshape(len(X), 1), t.reshape(len(t), 1)#将数据重新组织,X t整成一个列向量


    def design_matrix(self, M):                       #M为阶数;设置基函数的矩阵
        X, t =self.loadata()
        #X_trans = X
        design_matrix = np.empty(shape=[len(X), M])        #返回一个新的空矩阵,M列
        for i in range(0, M):
            design_matrix[:, i] = [x**i for x in X]
            return design_matrix

    def lossFunc(self, M, W, belta, alpha):
        X, t = self.loadata()
        Lossfunc = -belta*0.5*np.linalg.norm((t-np.dot(self.design_matrix(M), W)))**2-alpha*0.5*np.linalg.norm(W)**2
        return Lossfunc

    def trainModel(self,M,alpha, belta, learning_rate):   #优化model，SGD
        X, t = self.loadata()
        W = np.random.random((M, 1))
        i = 0
        while True:
            i += 1
            loss_last = self.lossFunc(W, M, alpha, belta)
            print('epoch为:', i, 'w为：', W, 'lossFunc为: ', self.lossFunc(W, M, alpha, belta))
            gradient = -belta * np.dot(self.design_matrix(M).T, (t - np.dot(self.design_matrix(M), W))) - alpha * W
            print('-------------')
            W = W - learning_rate * gradient
            loss = self.lossFunc(W, M, alpha, belta)
            if loss - loss_last <= 1:
                break
        return W

    def calculateW(self, alpha, belta, M):
        X, t = self.loadata()
        matrix = self.design_matrix(M)
        return np.dot(np.dot(np.linalg.inv(alpha/belta*np.eye(M)+np.dot(matrix.T,matrix)), matrix.T), t)

    def drawpic(self, M, W):
        X, t = self.loadata()
        fig = plt.figure().add_subplot(1, 1, 1)
        fig.scatter(X, t, c='green', marker='.')#画散点图
        fig.set_title('bayesLinearRegression')
        fig.set_xlabel('X')
        fig.set_ylabel('output_tn')
        fig.plot(X, np.dot(self.design_matrix(M), W), c='black')#画拟合曲线
        plt.show()
        return


if __name__ == '__main__':
    file = bayesLR(path='C:/Users/ADD\Desktop\研究生\code\BL/testSet.txt')
    W_final = file.calculateW(alpha=2, belta=25, M=9)
    file.drawpic(M=9, W=W_final)
