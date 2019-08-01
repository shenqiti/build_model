'''

其它常用分类算法
1.k近邻（kNN）
2.随机森林算法
3.决策树
4.逻辑斯蒂回归
5.朴素贝叶斯

'''

# #1 kNN算法
#
# import pandas as pd
# import numpy as np
# from sklearn import neighbors
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from sklearn.model_selection import train_test_split
# import warnings
# warnings.filterwarnings("ignore")
#
#
# # INPUT_PATH = './data_finall/day_data/feature_xg.csv '
# # INPUT_PATH = './data_finall/day_data/hour2day_norm_PCA2.csv'  # 归一化后PCA
# # INPUT_PATH = './data_finall/day_data/hour2day_norm_PCA3.csv'
# # INPUT_PATH = './data_finall/day_data/data_aday_2014_train.csv'
# INPUT_PATH = './data_finall/day_data/data_aday_2014_train_norm_PCA.csv'
# # INPUT_PATH = './data_finall/day_data/hour2day_norm.csv'
# # INPUT_PATH = './data_finall/day_data/hour2day_feature_cross_la.csv'
# data = pd.read_csv(INPUT_PATH)
#
# data = np.array(data)
# x, y = np.split(data, (13,), axis=1)
# x = x[:, :13]
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=10)
# for i in range(1,8):
#
#     #k近邻分类，这里选择i个近邻
#     knn=neighbors.KNeighborsClassifier(i,weights='uniform')
#     #可以使用fit 或者cross_val_score 函数来得到结果
#     knn.fit(x_train,y_train)
#     y_predict = knn.predict(x_test)
#     MSE = mean_squared_error(y_predict, y_test)
#     MAE = mean_absolute_error(y_predict, y_test)
#     R2 = r2_score(y_predict, y_test)
#     print("KNN分类器%s个近邻的结果为，MSE:%s,MAE:%s,R2:%s" %(i,MSE,MAE,R2))
#
#
#
# #######################################################################################################
#
#
# #模型2：随机森林算法
#
# #设定随机森林分类模型
# from sklearn import ensemble
# import numpy as np
# import pandas as pd
#
#
#
# # INPUT_PATH = './data_finall/day_data/feature_xg.csv '
# # INPUT_PATH = './data_finall/day_data/hour2day_norm_PCA3.csv'
# # INPUT_PATH = './data_finall/day_data/data_aday_2014_train.csv'
# INPUT_PATH = './data_finall/day_data/data_aday_2014_train_norm_PCA.csv'
# # INPUT_PATH = './data_finall/day_data/hour2day_norm.csv'
# # INPUT_PATH = './data_finall/day_data/hour2day_feature_cross_la.csv'
# data = pd.read_csv(INPUT_PATH)
#
# data = np.array(data)
# x, y = np.split(data, (13,), axis=1)
# x = x[:, :13]
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=10)
#
# for i in range(1,15):
#     #设定随机森林分类模型
#     rf=ensemble.RandomForestClassifier(i)
#     rf.fit(x_train,y_train)
#     y_predict = rf.predict(x_test)
#     MSE = mean_squared_error(y_predict, y_test)
#     MAE = mean_absolute_error(y_predict, y_test)
#     R2 = r2_score(y_predict, y_test)
#     print("%s个随机森林的分类结果为为：MSE:%s,MAE:%s,R2:%s" % (i,MSE, MAE, R2))
#
#
#
# #######################################################################################################
#
# #模型3：决策树
#
# from sklearn import tree
# import pandas as pd
# import numpy as np
# #设定X，y值
#
# # INPUT_PATH = './data_finall/day_data/feature_xg.csv '
# # INPUT_PATH = './data_finall/day_data/hour2day_norm_PCA3.csv'
# # INPUT_PATH = './data_finall/day_data/data_aday_2014_train.csv'
# INPUT_PATH = './data_finall/day_data/data_aday_2014_train_norm_PCA.csv'
# # INPUT_PATH = './data_finall/day_data/hour2day_norm.csv'
# # INPUT_PATH = './data_finall/day_data/hour2day_feature_cross_la.csv'
# data = pd.read_csv(INPUT_PATH)
#
# data = np.array(data)
# x, y = np.split(data, (13,), axis=1)
# x = x[:, :13]
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=10)
#
# dt=tree.DecisionTreeClassifier()
# #训练模型，并得到准确率
# dt.fit(x_train,y_train)
# y_predict = dt.predict(x_test)
# MSE = mean_squared_error(y_predict, y_test)
# MAE = mean_absolute_error(y_predict, y_test)
# R2 = r2_score(y_predict, y_test)
# print("决策树的分类结果为为：MSE:%s,MAE:%s,R2:%s" % (MSE, MAE, R2))
#
# #######################################################################################################
#
#
# # 模型4 逻辑斯蒂回归
#
# import numpy as np
# import pandas as pd
# from sklearn import linear_model
# import warnings
# warnings.filterwarnings("ignore")
#
#
# # INPUT_PATH = './data_finall/day_data/feature_xg.csv '
# # INPUT_PATH = './data_finall/day_data/hour2day_norm_PCA3.csv'
# # INPUT_PATH = './data_finall/day_data/data_aday_2014_train.csv'
# INPUT_PATH = './data_finall/day_data/data_aday_2014_train_norm_PCA.csv'
# # INPUT_PATH = './data_finall/day_data/hour2day_norm.csv'
# # INPUT_PATH = './data_finall/day_data/hour2day_feature_cross_la.csv'
# data = pd.read_csv(INPUT_PATH)
#
# data = np.array(data)
# x, y = np.split(data, (13,), axis=1)
# x = x[:, :13]
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=10)
#
# lm=linear_model.LogisticRegression()
#
# lm.fit(x_train,y_train)
# y_predict = lm.predict(x_test)
#
# MSE = mean_squared_error(y_predict, y_test)
# MAE = mean_absolute_error(y_predict, y_test)
# R2 = r2_score(y_predict, y_test)
# print("逻辑斯蒂回归的分类结果为：MSE:%s,MAE:%s,R2:%s" % (MSE, MAE, R2))
#
#
#
#
# #######################################################################################################
#
# #模型5：朴素贝叶斯
#
#
#
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import GaussianNB
#
#
# # INPUT_PATH = './data_finall/day_data/feature_xg.csv '
# # INPUT_PATH = './data_finall/day_data/hour2day_norm_PCA3.csv'
# # INPUT_PATH = './data_finall/day_data/data_aday_2014_train.csv'
# INPUT_PATH = './data_finall/day_data/data_aday_2014_train_norm_PCA.csv'
# # INPUT_PATH = './data_finall/day_data/hour2day_norm.csv'
# # INPUT_PATH = './data_finall/day_data/hour2day_feature_cross_la.csv'
# data = pd.read_csv(INPUT_PATH)
#
# data = np.array(data)
# x, y = np.split(data, (13,), axis=1)
# x = x[:, :13]
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=10)
#
#
#
# clf = GaussianNB()
# clf.fit(x_train, y_train)
# y_predict = clf.predict(x_test)
#
# MSE = mean_squared_error(y_predict, y_test)
# MAE = mean_absolute_error(y_predict, y_test)
# R2 = r2_score(y_predict, y_test)
# print("朴素贝叶斯的分类结果为：MSE:%s,MAE:%s,R2:%s" % (MSE, MAE, R2))
#
#
# #######################################################################################################

# 支持向量机
from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")
INPUT_PATH = './data_finall/day_data/feature_xg.csv '
# INPUT_PATH = './data_finall/day_data/hour2day_norm_PCA3.csv'
# INPUT_PATH = './data_finall/day_data/data_aday_2014_train.csv'
# INPUT_PATH = './data_finall/day_data/hour2day_norm.csv'
# INPUT_PATH = './data_finall/day_data/hour2day_feature_cross_la.csv'
# INPUT_PATH = './data_finall/day_data/data_aday_2014_train_norm_PCA.csv'
data = pd.read_csv(INPUT_PATH)

data = np.array(data)
x, y = np.split(data, (5,), axis=1)
x = x[:, :5]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=10)

clf = svm.SVC(kernel='rbf', C=2, gamma=2.1)    #待优化的参数
clf.fit(x_train, y_train)
y_predict = clf.predict(x_test)

MSE = mean_squared_error(y_predict, y_test)
MAE = mean_absolute_error(y_predict, y_test)
R2 = r2_score(y_predict, y_test)


print("支持向量机的结果为：MSE:%s,MAE:%s,R2:%s" % (MSE, MAE, R2))

