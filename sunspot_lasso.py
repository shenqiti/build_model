'''
By:shenqiti
2019/9/4

'''

import numpy as np # 快速操作结构数组的工具
import matplotlib.pyplot as plt  # 可视化绘制
from sklearn import preprocessing
from sklearn.linear_model import Lasso,LassoCV,LassoLarsCV   # Lasso回归,LassoCV交叉验证实现alpha的选取，LassoLarsCV基于最小角回归交叉验证实现alpha的选取
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
#利用传统特征
inputpath1 = 'D:/data/lasso/train_feature_deal.csv'
inputpath2 = 'D:/data/lasso/test_feature_deal.csv'
#利用lasso筛选后的特征
# inputpath1 = './data/new_train_feature_2.csv'
# inputpath2 = './data/new_test_feature_2.csv'
outputpath = 'D:/data/resault/gbdt_result_2.csv'
#读取变量
INPUT_PATH = "D:/data/train_processed.csv"


label = []
feature = []

with open (inputpath1) as f:
    for line in f:
        ll = [float(item) for item in line.strip().split(",")]
        label.append(ll[1])
        feature.append(ll[2:])
X_train, X_test, y_train, y_test = train_test_split(
    feature, label, test_size=0.33, random_state=10)


def read_csv():
    f1 = open(inputpath1,'r')
    f2 = open(inputpath2,'r')
    #把数据变成矩阵
    train_feature = np.loadtxt(f1,delimiter=',',skiprows=1)#跳过第一行
    test_feature = np.loadtxt(f2, delimiter=',', skiprows=1)  # 跳过第一行
    train_feature_matrix =np.array(train_feature)
    test_feature_matrix = np.array(test_feature)
    #选出x,y
    x_train= train_feature_matrix[:,1:]#除去第一列
    y_train= train_feature_matrix[:,0]#选取第一列
    x_test = test_feature_matrix[:,:]
    #标准化数据
    x_train_sc= preprocessing.scale(x_train)
    x_test_sc = preprocessing.scale(x_test)
    #尝试把测试集和训练集放一起标准化,但是一起标准化的结果没有分开标准化好
    # combine_matrix = np.concatenate((x_train, x_test), axis=0)#把矩阵在列上合并，np.hstack()在行上合并
    # combine_matrix_st = preprocessing.scale(combine_matrix)
    # x_train_sc = combine_matrix_st[0:2126]
    # x_test_sc = combine_matrix_st[2126:]
    return(x_train_sc,y_train,x_test_sc)

#使用lasso建模
def lasso(x_train,y_train,x_test):
    #model = Lasso(alpha=0.01)  # 调节alpha可以实现对拟合的程度
    #model = LassoCV(max_iter=3000)  # LassoCV自动调节alpha可以实现选择最佳的alpha,0.0295。
    model = LassoLarsCV()  # LassoLarsCV自动调节alpha可以实现选择最佳的alpha
    print(x_train.shape);print(y_train.shape)
    model.fit(x_train, y_train)  # 线性回归建模
    print('系数矩阵:\n', model.coef_)
    print('线性回归模型:\n', model)
    print('最佳的alpha：',model.alpha_)
    predicted = model.predict(x_test)
    print(predicted.shape)
    return(predicted)

#使用多层感知机mlp建模
def mlp(x_train,y_train,x_test):
    model = MLPRegressor()  # LassoLarsCV自动调节alpha可以实现选择最佳的alpha
    model.fit(x_train, y_train)  # 线性回归建模
    predicted = model.predict(x_test)
    return (predicted)

#使用GBDT建模
def gbdt(x_train,y_train,x_test):
    model = GradientBoostingRegressor()
    model.fit(x_train, y_train)  # 线性回归建模
    predicted = model.predict(x_test)
    MSE = mean_squared_error(label, predicted)
    MAE = mean_absolute_error(label, predicted)
    R2 = r2_score(label, predicted)
    result_rep = "MSE: %s, MAE: %s, R2: %s" % (MSE, MAE, R2)
    print(result_rep)


    return (predicted)

def save_result(predicted):
    fid0 = open(outputpath, 'w')
    fid0.write("time,prediction" + "\n")
    i=1
    for item in predicted:
        fid0.write(str(i) + "," + str(item) + "\n")
        i = i + 1
    fid0.close()





if __name__ == '__main__':


    x_train,y_train,x_test = read_csv()
    #predicted = lasso(x_train, y_train, x_test)
    #predicted = mlp(x_train, y_train, x_test)
    predicted = gbdt(x_train, y_train, x_test)
    save_result(predicted)




