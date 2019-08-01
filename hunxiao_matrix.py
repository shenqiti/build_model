#混淆矩阵

from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np


INPUT_PATH1 = './data_pro/Analysis/label2.csv'  # 用来预测的因素
INPUT_PATH2 = './data_pro/Analysis/test1_label_predict2.csv'



y_true = pd.read_csv(INPUT_PATH1)
y_pred = pd.read_csv(INPUT_PATH2)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

C=confusion_matrix(y_true, y_pred)
print(C, end='\n\n')
