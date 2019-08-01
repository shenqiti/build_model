'''

绘制不同分类器的误分类图
'''


'''

其它常用分类算法
1.k近邻（kNN）
2.随机森林算法
3.决策树
4.逻辑斯蒂回归
5.朴素贝叶斯
6.SVM

'''

from sklearn import svm
from sklearn import ensemble
from sklearn import neighbors
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import tree
import numpy as np
from sklearn import linear_model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# INPUT_PATH = './data_finall/day_data/feature_xg.csv '
# INPUT_PATH = './data_finall/day_data/data_aday_2014_111.csv'
INPUT_PATH = './data_finall/day_data/data_aday_2014_train_norm_PCA.csv '



data = pd.read_csv(INPUT_PATH)

data = np.array(data)
x, y = np.split(data, (13,), axis=1)
x = x[:, :13]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=10)

#1 kNN算法

for i in range(1,2):

    #k近邻分类，这里选择i个近邻
    knn=neighbors.KNeighborsClassifier(i,weights='uniform')
    #可以使用fit 或者cross_val_score 函数来得到结果
    knn.fit(x_train,y_train)
    y_predict = knn.predict(x_test)
    MSE = mean_squared_error(y_predict, y_test)
    MAE = mean_absolute_error(y_predict, y_test)
    R2 = r2_score(y_predict, y_test)
    print("KNN分类器%s个近邻的结果为，MSE:%s,MAE:%s,R2:%s" %(i,MSE,MAE,R2))
    num = []
    CNT = []
    cnt = 0
    for i in range(0, len(y_predict)):

        if y_predict[i] != y_test[i]:
            cnt = cnt + 1
            CNT.append(cnt)
        else:
            CNT.append(cnt)
    for i in range(0, len(CNT)):
        num.append(i)
    # for i in range(0,len(CNT)):
    #     CNT[i] = CNT[i]/cnt
    # plt.scatter(num,CNT)
    l1 = plt.plot(num, CNT)




#######################################################################################################


#模型2：随机森林算法

#设定随机森林分类模型

data = np.array(data)
x, y = np.split(data, (13,), axis=1)
x = x[:, :13]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=10)


for i in range(2,3):
    #设定随机森林分类模型
    rf=ensemble.RandomForestClassifier(i)
    rf.fit(x_train,y_train)
    y_predict = rf.predict(x_test)
    MSE = mean_squared_error(y_predict, y_test)
    MAE = mean_absolute_error(y_predict, y_test)
    R2 = r2_score(y_predict, y_test)
    print("%s个随机森林的分类结果为为：MSE:%s,MAE:%s,R2:%s" % (i,MSE, MAE, R2))
    num = []
    CNT = []
    cnt = 0
    for i in range(0, len(y_predict)):

        if y_predict[i] != y_test[i]:
            cnt = cnt + 1
            CNT.append(cnt)
        else:
            CNT.append(cnt)
    for i in range(0, len(CNT)):
        num.append(i)
    # for i in range(0,len(CNT)):
    #     CNT[i] = CNT[i]/cnt
    # plt.scatter(num,CNT)
    l2 = plt.plot(num, CNT)




#######################################################################################################

#模型3：决策树


#设定X，y值
data = np.array(data)
x, y = np.split(data, (13,), axis=1)
x = x[:, :13]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=10)

dt=tree.DecisionTreeClassifier()
#训练模型，并得到准确率
dt.fit(x_train,y_train)
y_predict = dt.predict(x_test)
MSE = mean_squared_error(y_predict, y_test)
MAE = mean_absolute_error(y_predict, y_test)
R2 = r2_score(y_predict, y_test)
print("决策树的分类结果为为：MSE:%s,MAE:%s,R2:%s" % (MSE, MAE, R2))
num = []
CNT = []
cnt = 0
for i in range(0, len(y_predict)):

    if y_predict[i] != y_test[i]:
        cnt = cnt + 1
        CNT.append(cnt)
    else:
        CNT.append(cnt)
for i in range(0, len(CNT)):
    num.append(i)
# for i in range(0,len(CNT)):
#     CNT[i] = CNT[i]/cnt
# plt.scatter(num,CNT)

l3 = plt.plot(num, CNT)


#######################################################################################################


# 模型4 逻辑斯蒂回归
data = np.array(data)
x, y = np.split(data, (13,), axis=1)
x = x[:, :13]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=10)

lm=linear_model.LogisticRegression()

lm.fit(x_train,y_train)
y_predict = lm.predict(x_test)

MSE = mean_squared_error(y_predict, y_test)
MAE = mean_absolute_error(y_predict, y_test)
R2 = r2_score(y_predict, y_test)
print("逻辑斯蒂回归的分类结果为：MSE:%s,MAE:%s,R2:%s" % (MSE, MAE, R2))
num = []
CNT = []
cnt = 0
for i in range(0, len(y_predict)):

    if y_predict[i] != y_test[i]:
        cnt = cnt + 1
        CNT.append(cnt)
    else:
        CNT.append(cnt)
for i in range(0, len(CNT)):
    num.append(i)
# for i in range(0,len(CNT)):
#     CNT[i] = CNT[i]/cnt
# plt.scatter(num,CNT)

l4 = plt.plot(num, CNT)


#######################################################################################################

#模型5：朴素贝叶斯
data = np.array(data)
x, y = np.split(data, (13,), axis=1)
x = x[:, :13]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=10)



clf = GaussianNB()
clf.fit(x_train, y_train)
y_predict = clf.predict(x_test)

MSE = mean_squared_error(y_predict, y_test)
MAE = mean_absolute_error(y_predict, y_test)
R2 = r2_score(y_predict, y_test)
print("朴素贝叶斯的分类结果为：MSE:%s,MAE:%s,R2:%s" % (MSE, MAE, R2))

num = []
CNT = []
cnt = 0
for i in range(0, len(y_predict)):

    if y_predict[i] != y_test[i]:
        cnt = cnt + 1
        CNT.append(cnt)
    else:
        CNT.append(cnt)
for i in range(0, len(CNT)):
    num.append(i)
# for i in range(0,len(CNT)):
#     CNT[i] = CNT[i]/cnt
# plt.scatter(num,CNT)

l5 = plt.plot(num, CNT)


#
# #######################################################################################################

# 支持向量机
data = np.array(data)
x, y = np.split(data, (13,), axis=1)
x = x[:, :13]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=10)

clf = svm.SVC(kernel='rbf', C=2, gamma=2.08)    #待优化的参数
clf.fit(x_train, y_train)
y_predict = clf.predict(x_test)

MSE = mean_squared_error(y_predict, y_test)
MAE = mean_absolute_error(y_predict, y_test)
R2 = r2_score(y_predict, y_test)


print("支持向量机的结果为：MSE:%s,MAE:%s,R2:%s" % (MSE, MAE, R2))
num = []
CNT = []
cnt = 0

for i in range(0, len(y_predict)):

    if y_predict[i] != y_test[i]:
        cnt = cnt + 1
        CNT.append(cnt)
    else:
        CNT.append(cnt)
for i in range(0, len(CNT)):
    num.append(i)
# for i in range(0,len(CNT)):
#     CNT[i] = CNT[i]/cnt
# plt.scatter(num,CNT)
l6 = plt.plot(num, CNT)

plt.title('All classfier with 2014.csv')
plt.legend(['KNN','RandomForest','DecisionTree','LogisticRegression','GaussianNB','SVM'])
#预测结果可视化.....
#多分类器可视化
plt.show()

