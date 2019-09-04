'''
By:shenqiti
2019/9/4

DELM 实现

'''

import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np
from sklearn.datasets import load_iris  # 数据集
from sklearn.model_selection import train_test_split  # 数据集的分割函数
from sklearn.preprocessing import StandardScaler  # 数据预处理
from sklearn import metrics
from sklearn.model_selection import cross_validate
# 引入包含数据验证方法的包
from sklearn import metrics


class HiddenLayer:
    def __init__(self, x, num):
        row = x.shape[0]
        columns = x.shape[1]
        rnd = np.random.RandomState(4444)
        self.w = rnd.uniform(-1, 1, (columns, num))
        self.b = np.zeros([row, num], dtype=float)
        for i in range(num):
            rand_b = rnd.uniform(-0.4, 0.4)
            for j in range(row):
                self.b[j, i] = rand_b
        # self.h = self.sigmoid(np.dot(x, self.w) + self.b)
        self.h = self.relu(np.dot(x, self.w) + self.b)
        self.H_ = np.linalg.pinv(self.h)
        # print(self.H_.shape)

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))


    def relu(self,x):
        return np.where(x < 0, 0, x)

    def regressor_train(self, T):
        C = 2
        I = len(T)
        sub_former = np.dot(np.transpose(self.h), self.h) + I / C
        all_m = np.dot(np.linalg.pinv(sub_former), np.transpose(self.h))
        T = T.reshape(-1, 1)
        self.beta = np.dot(all_m, T)
        return self.beta

    def classifisor_train(self, T):
        en_one = OneHotEncoder()
        T = en_one.fit_transform(T.reshape(-1, 1)).toarray()  # 独热编码之后一定要用toarray()转换成正常的数组
        C = 3
        I = len(T)
        sub_former = np.dot(np.transpose(self.h), self.h) + I / C
        all_m = np.dot(np.linalg.pinv(sub_former), np.transpose(self.h))
        self.beta = np.dot(all_m, T)
        return self.beta

    def regressor_test(self, test_x):
        b_row = test_x.shape[0]
        # h = self.sigmoid(np.dot(test_x, self.w) + self.b[:b_row, :])
        h = self.relu(np.dot(test_x, self.w) + self.b[:b_row, :])
        result = np.dot(h, self.beta)
        return result

    def classifisor_test(self, test_x):
        b_row = test_x.shape[0]
        # h = self.sigmoid(np.dot(test_x, self.w) + self.b[:b_row, :])
        h = self.relu(np.dot(test_x, self.w) + self.b[:b_row, :])
        result = np.dot(h, self.beta)
        result = [item.tolist().index(max(item.tolist())) for item in result]
        return result


