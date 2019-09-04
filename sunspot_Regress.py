'''
By:shenqiti
2019/9/4

'''


import numpy as np  # 快速操作结构数组的工具
import matplotlib.pyplot as plt  # 可视化绘制
from sklearn import preprocessing
# Lasso回归,LassoCV交叉验证实现alpha的选取，LassoLarsCV基于最小角回归交叉验证实现alpha的选取
from sklearn.linear_model import Lasso, LinearRegression, Ridge, BayesianRidge, Perceptron, LassoLarsCV, LassoLarsIC, \
    LassoCV, RidgeCV, ElasticNetCV, ARDRegression, BayesianRidge, HuberRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor

# 利用传统特征
inputpath1 = './data/train_feature_deal.csv'
inputpath2 = './data/test_feature_deal.csv'
# 利用lasso筛选后的特征
# inputpath1 = './data/new_train_feature_2.csv'
# inputpath2 = './data/new_test_feature_2.csv'
outputpath = './result2/gbdt_result_2.csv'

# Kesa's CSV
outputpath_Gbdt = './result2/gbdt_result_Gbdt.csv'
outputpath_Lasso = './result2/gbdt_result_Lasso.csv'
outputpath_mlp = './result2/gbdt_result_mlp.csv'
outputpath_etr = './result2/gbdt_result_etr.csv'
outputpath_rfr = './result2/gbdt_result_rfr.csv'
outputpath_All = './result2/gbdt_result_all.csv'


# 读取变量
def read_csv():
    f1 = open(inputpath1, 'r')
    f2 = open(inputpath2, 'r')

    # 把数据变成矩阵
    train_feature = np.loadtxt(f1, delimiter=',', skiprows=1)  # 跳过第一行
    test_feature = np.loadtxt(f2, delimiter=',', skiprows=1)  # 跳过第一行
    train_feature_matrix = np.array(train_feature)
    test_feature_matrix = np.array(test_feature)

    # 选出x,y
    x_train = train_feature_matrix[:, 1:]  # 除去第一列
    y_train = train_feature_matrix[:, 0]  # 选取第一列
    x_test = test_feature_matrix[:, :]

    # 标准化数据
    x_train_sc = preprocessing.scale(x_train)
    x_test_sc = preprocessing.scale(x_test)

    # 尝试把测试集和训练集放一起标准化,但是一起标准化的结果没有分开标准化好
    # combine_matrix = np.concatenate((x_train, x_test), axis=0)#把矩阵在列上合并，np.hstack()在行上合并
    # combine_matrix_st = preprocessing.scale(combine_matrix)
    # x_train_sc = combine_matrix_st[0:2126]
    # x_test_sc = combine_matrix_st[2126:]
    return (x_train_sc, y_train, x_test_sc)


### --------------模型区域
# 使用lasso建模
def lasso(x_train, y_train, x_test):
    # model = Lasso(alpha=0.01)  # 调节alpha可以实现对拟合的程度
    # model = LassoCV(max_iter=3000)  # LassoCV自动调节alpha可以实现选择最佳的alpha,0.0295。
    model = LassoLarsCV()  # LassoLarsCV自动调节alpha可以实现选择最佳的alpha
    print(x_train.shape);
    print(y_train.shape)
    model.fit(x_train, y_train)  # 线性回归建模
    print('系数矩阵:\n', model.coef_)
    print('线性回归模型:\n', model)
    print('最佳的alpha：', model.alpha_)
    predicted = model.predict(x_test)
    print(predicted.shape)
    return (predicted)


# 使用多层感知机mlp建模
def mlp(x_train, y_train, x_test):
    model = MLPRegressor()  # LassoLarsCV自动调节alpha可以实现选择最佳的alpha
    model.fit(x_train, y_train)  # 线性回归建模
    predicted = model.predict(x_test)
    return (predicted)


# 使用GBDT建模
def gbdt(x_train, y_train, x_test):
    model = GradientBoostingRegressor()
    model.fit(x_train, y_train)  # 线性回归建模
    predicted = model.predict(x_test)
    return (predicted)


# 使用极端随机森林建模
def etr(x_train, y_train, x_test):
    model = ExtraTreeRegressor()
    model.fit(x_train, y_train)  # 线性回归建模
    predicted = model.predict(x_test)
    return (predicted)


# 使用随机森林建模RandomForestRegressor
def rfr(x_train, y_train, x_test):
    # model = RandomForestRegressor()
    model = HuberRegressor()
    model.fit(x_train, y_train)  # 线性回归建模
    predicted = model.predict(x_test)
    return (predicted)


### --------------模型区域

# 一个结果的导出函数
def save_result(predicted):
    fid0 = open(outputpath, 'w')
    fid0.write("time,prediction" + "\n")
    i = 1
    for item in predicted:
        fid0.write(str(i) + "," + str(item) + "\n")
        i = i + 1
    fid0.close()


# Copied:一个结果的导出函数
def save_result_k(predicted, outputPath):
    fid0 = open(outputPath, 'w')
    fid0.write("time,prediction" + "\n")
    i = 1
    for item in predicted:
        fid0.write(str(i) + "," + str(item) + "\n")
        i = i + 1
    fid0.close()


# 合并表
def merge_results():
    # 安全性检测
    if collect_Res.__len__() == 0:
        return print("结果集合为空")

    # 安全性检测
    size = np.size(collect_Res[0])
    for predict in collect_Res:
        if size != np.size(predict):
            return print("三个表的数据量不同")

    # 写入文件
    fileAll = open(outputpath_All, "w")
    fileAll.write("time,prediction_GBDT,prediction_Lasso,prediction_mlp,prediction_rfr" + "\n")

    # i 是数据的行数
    # j 是模型的数量
    for i in range(size):
        fileAll.write(str(i))
        for j in range(collect_Res.__len__()):
            fileAll.write("," + str(collect_Res[j][i]))
        fileAll.write("\n")
    fileAll.close()


# ---------------正片---------------

global time_save  # 储存次数
collect_Res = []  # 结果的存储集合

if __name__ == '__main__':
    time_save = 0

    # 赋值
    x_train, y_train, x_test = read_csv()
    # predicted = lasso(x_train, y_train, x_test)
    # predicted = mlp(x_train, y_train, x_test)

    # Gbdt
    predicted_GBDT = gbdt(x_train, y_train, x_test)
    # print predicted
    print('predicted_GBDT\'s Size:')
    print(np.size(predicted_GBDT))
    # print(predicted)
    print(predicted_GBDT[0])
    save_result_k(predicted_GBDT, outputpath_Gbdt)
    collect_Res.append(predicted_GBDT)

    # Lasso
    predicted_Lasso = lasso(x_train, y_train, x_test)
    # print predicted
    print('predicted_Lasso:\'s Size:')
    print(np.size(predicted_Lasso))
    # print(predicted)
    print(predicted_Lasso[0])
    save_result_k(predicted_Lasso, outputpath_Lasso)
    collect_Res.append(predicted_Lasso)

    # Mlp
    predicted_mlp = mlp(x_train, y_train, x_test)
    # print predicted
    print('predicted_mlp:\'s Size:')
    print(np.size(predicted_mlp))
    # print(predicted)
    print(predicted_mlp[0])
    save_result_k(predicted_mlp, outputpath_mlp)
    collect_Res.append(predicted_mlp)

    # # ExtraTreeRegressor
    # predicted_etr = etr(x_train, y_train, x_test)
    # # print predicted
    # print('predicted_etr:\'s Size:')
    # print(np.size(predicted_etr))
    # # print(predicted)
    # print(predicted_etr[0])
    # save_result_k(predicted_etr, outputpath_etr)
    # collect_Res.append(predicted_etr)

    # RandomForestRegressor
    predicted_rfr = rfr(x_train, y_train, x_test)
    # print predicted
    print('predicted_rfr:\'s Size:')
    print(np.size(predicted_rfr))
    # print(predicted)
    print(predicted_rfr[0])
    save_result_k(predicted_rfr, outputpath_rfr)
    collect_Res.append(predicted_rfr)

    # print collect_Res.count()
    print(collect_Res.__len__())
    merge_results()
