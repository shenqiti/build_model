'''

SVM分类学习：

根据算力等现实要求，我们最后选择采用rbf核函数进行学习

本程序用于对rbg核函数的搜索半径gama以及惩罚因子C进行组合，从而选取结果较好的一组
'''


from sklearn import svm,metrics
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score
import pandas as pd
from sklearn.model_selection import cross_val_score
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")




# para_kernel_get = ["rbf", "poly", "linear"]
para_kernel_get = ["rbf"]
para_C_get = [1,2]
gama = [4.0+i*0.05 for i in range(0,20)]
# gama = [1,2,3,4,5,6,7,8,9,10]
for para_kernel in para_kernel_get:
    for para_C in para_C_get:
        for each in gama:
            # INPUT_PATH = './data_finall/day_data/feature_all.csv'     #原数据集   大致最优：0.49274597452208013，
            # INPUT_PATH = './data_finall/day_data/feature_jb.csv'   #脚本数据集，以后没必要出现了，还没原数据好呢！
            # INPUT_PATH = './data_finall/day_data/feature_xg.csv'   #分析出来的数据集，值得鼓励！大致最优：0.613496778091019
            #INPUT_PATH = './data_finall/day_data/xg_PCA.csv'   #对相关性分析以后进行PCA操作之后的数据
            # INPUT_PATH = './data_finall/day_data/hour2day_norm.csv'  #归一化后
            INPUT_PATH = './data_finall/day_data/hour2day_norm_PCA.csv'
            # INPUT_PATH = './data_finall/day_data/hour2day_feature_cross_la.csv'
            # INPUT_PATH = './data_finall/day_data/data_aday_2014_train_norm.csv'
            data = pd.read_csv(INPUT_PATH)
            data = np.array(data)


            x, y = np.split(data, (6,), axis=1)    #根据csv文件选择x，y列
            x = x[:, :6]
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=10)

            print("开始训练....")

            clf = svm.SVC(kernel=para_kernel, C=para_C, gamma=each)
            clf.fit(x_train, y_train)
            y_predict = clf.predict(x_test)

            score = cross_val_score(clf,x,y,cv=4,scoring='accuracy')
            precision = precision_score(y_test, y_predict, average="macro")
            recall = recall_score(y_test, y_predict, average="macro")
            f1 = f1_score(y_test, y_predict, average="macro")

            print("gama：%s,惩罚因子：%s,准确率：%s，召回率：%s，F1值：%s,交叉检验：%s" %  (each,para_C,precision, recall, f1,score))
#             num = []
#             CNT = []
#             cnt = 0
#             for i in range(0,len(y_predict)):
#
#                 if y_predict[i] != y_test[i]:
#                     cnt = cnt +1
#                     CNT.append(cnt)
#                 else:
#                     CNT.append(cnt)
#             for i in range(0,len(CNT)):
#                 num.append(i)
#             # for i in range(0,len(CNT)):
#             #     CNT[i] = CNT[i]/cnt
#             # plt.scatter(num,CNT)
#             plt.plot(num,CNT)
# plt.title('C=1')
# plt.legend(['gama=1','gama=2','gama=3','gama=4','gama=5','gama=6','gama=7','gama=8','gama=9','gama=10'])
# plt.show()
#
# #预测结果可视化.....
# #多分类器可视化
#
