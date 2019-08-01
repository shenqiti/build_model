'''

SVM预测：

将数据PCA处理后送入SVM

'''

from sklearn import svm,metrics
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")


para_kernel_get = ["rbf"]
para_C_get = 1
gama = 4.4
for para_kernel in para_kernel_get:

    # INPUT_PATH = './data_finall/day_data/train.csv'   #用来训练的模型
    # INPUT_PATH2 = './data_finall/day_data/test.csv'    #用来预测的因素
    # OUTPUT_PATH = './data_finall/day_data/result.csv'

    INPUT_PATH = './data_finall/day_data/hour2day_norm_PCA.csv'   #用来训练的模型
    INPUT_PATH2 = './data_finall/day_data/data_Scenarios_Analysis_norm_PCA.csv'    #用来预测的因素
    OUTPUT_PATH = './data_finall/day_data/result.csv'

    data = pd.read_csv(INPUT_PATH)
    data = np.array(data)
    x, y = np.split(data, (6,), axis=1)
    x = x[:, :6]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=10)
    print("开始训练....")
    clf = svm.SVC(kernel=para_kernel, C=para_C_get, gamma=gama)
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    precision = precision_score(y_test, y_predict, average="macro")
    recall = recall_score(y_test, y_predict, average="macro")
    f1 = f1_score(y_test, y_predict, average="macro")
    MSE = mean_squared_error(y_predict, y_test)
    MAE = mean_absolute_error(y_predict, y_test)
    R2 = r2_score(y_predict, y_test)
    print("MSE:%s,MAE:%s,R2:%s" %(MSE,MAE,R2))
    print("gama：%s,惩罚因子：%s,准确率：%s，召回率：%s，F1值：%s" %  (gama,para_C_get,precision, recall, f1))

#预测阶段
    #为什么会自动跳过第一行？？？？
    feature = pd.read_csv(INPUT_PATH2)
    feature= np.array(feature)
    predict_label = clf.predict(feature)

#将预测结果保存下来
    fn = open(OUTPUT_PATH,'w')
    for i in range(0,len(predict_label)):
        fn.write(str(predict_label[i]))
        fn.write('\n')
    print("预测结果写入完成！")
    fn.close()
