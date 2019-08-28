
#随机森林特征选择法 —— Gini Importance
'''
原理： 
使用Gini指数表示节点的纯度，Gini指数越大纯度越低。然后计算每个节点的Gini 指数 - 子节点的Gini 指数之和，记为Gini decrease。
最后将所有树上相同特征节点的Gini decrease加权的和记为Gini importance，该数值会在0-1之间，该数值越大即代表该节点（特征）重要性越大。
'''
import pandas
import numpy as np
from sklearn import ensemble
from sklearn.preprocessing import LabelEncoder

iris =pandas.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)
iris.columns=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species']

le = LabelEncoder()
le.fit(iris['Species'])
rf = ensemble.RandomForestClassifier()
features = ['PetalLengthCm','PetalWidthCm','SepalLengthCm','SepalWidthCm']
y = np.array(le.transform(iris['Species']))
X = np.array(iris[features])

#Gini importance
rf.fit(X,y)
print(rf.feature_importances_)




#随机森林特征选择法 —— Mean Decrease Accuracy
'''
原理：主要思路是打乱每个特征的特征值顺序，并且度量顺序变动对模型的精确率的影响。
很明显，对于不重要的变量来说，打乱顺序对模型的精确率影响不会太大，但是对于重要的变量来说，打乱顺序就会降低模型的精确率。
'''

from sklearn.metrics import accuracy_score
from sklearn.model_selection import ShuffleSplit

rs = ShuffleSplit(n_splits=10,test_size=0.1)
scores = np.zeros((10,4))
count = 0
for train_idx, test_idx in rs.split(X):
    X_train , X_test = X[train_idx] , X[test_idx]
    y_train , y_test = y[train_idx] , y[test_idx]
    r = rf.fit(X_train,y_train)
    acc = accuracy_score(y_test,rf.predict(X_test))
    for i in range(len(features)):
        X_t = X_test.copy()
        np.random.shuffle(X_t[:, i])
        shuff_acc = accuracy_score(y_test,rf.predict(X_t))
        scores[count,i] = ((acc-shuff_acc)/acc)
    count += 1
print(np.mean(scores,axis=0))


#线性回归特征选择：L1正则化Lasso
'''
什么是正则化？：监督机器学习问题无非就是“minimize your error while regularizing your parameters”，
也就是在规则化参数的同时最小化误差。最小化误差是为了让我们的模型拟合我们的训练数据，而规则化参数是防止我们的模型过分拟合我们的训练数据。

正则化的作用： 
1、约束参数，降低模型复杂度。 
2、规则项的使用还可以约束我们的模型的特性。这样就可以将人对这个模型的先验知识融入到模型的学习当中，
强行地让学习到的模型具有人想要的特性，例如稀疏、低秩、平滑等等。

L1范数：向量中各个元素绝对值之和，也有个美称叫“稀疏规则算子”（Lasso regularization）。

L1范数的作用：由于L1范数的天然性质，对L1优化的解是一个稀疏解，因此L1范数也被叫做稀疏规则算子。
通过L1可以实现特征的稀疏，去掉一些没有信息的特征，例如在对用户的电影爱好做分类的时候，
用户有100个特征，可能只有十几个特征是对分类有用的，大部分特征如身高体重等可能都是无用的，利用L1范数就可以过滤掉。
'''

import pandas
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder

iris = pandas.read_csv("iris.csv")
le = LabelEncoder()
le.fit(iris['Species'])
lm = linear_model.Lasso(0.02)
features = ['PetalLengthCm','PetalWidthCm','SepalLengthCm','SepalWidthCm']
y = np.array(le.transform(iris['Species']))
X = np.array(iris[features])

lm.fit(X,y)
print(lm.coef_)



