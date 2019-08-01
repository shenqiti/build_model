
'''
特征选择：从原始特征中选择出一些最有效特征以降低数据集维度的过程
先把pm2.5打上标签才能进行特征选择。
参考:https://static.dcxueyuan.com/content/disk/train/other/219c9338-5568-4526-917a-7dc277300d3f.html
'''
#

#####不建议在此数据集使用！！！！！！#########
#### 亲测   不如直接训练原数据 ###########
######### 😭    😭    😭 ###############

import pandas
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")
INPUT_DATA = './data_pro/data_pro_label.csv'

iris =pandas.read_csv(INPUT_DATA,header=None)
iris.columns=['DEWP','TEMP','PRES','Iws','Is','Ir','pm2.5']
le = LabelEncoder()
le.fit(iris['pm2.5'])
lm = linear_model.LogisticRegression()
features = ['DEWP','TEMP','PRES','Iws','Is','Ir']
y = le.transform(iris['pm2.5'])

selected_features = []
rest_features = features[:]
best_acc = 0
while len(rest_features)>0:
    temp_best_i = ''
    temp_best_acc = 0
    for feature_i in rest_features:
        temp_features = selected_features + [feature_i,]
        X = iris[temp_features]
        scores = cross_val_score(lm,X,y,cv=5 , scoring='accuracy')
        acc = np.mean(scores)
        if acc > temp_best_acc:
            temp_best_acc = acc
            temp_best_i = feature_i
    print("select",temp_best_i,"acc:",temp_best_acc)
    if temp_best_acc > best_acc:
        best_acc = temp_best_acc
        selected_features += [temp_best_i,]
        rest_features.remove(temp_best_i)
    else:
        break
print("best feature set: ",selected_features,"acc: ",best_acc)
