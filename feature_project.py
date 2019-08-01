'''

特征工程

url = https://segmentfault.com/a/1190000014799038

url = https://www.cnblogs.com/jasonfreak/p/5448385.html

'''

from sklearn.preprocessing import StandardScaler,MinMaxScaler,Normalizer,Binarizer

import pandas as pd
import numpy as np
INPUT_DATA = './data_finall/day_data/hour2day.csv'
OUPUT_PATH = './data_finall/day_data/hour2day.csv_norm111.csv'

#标准化，返回值为标准化后的数据
data = pd.read_csv(INPUT_DATA)
print(StandardScaler().fit_transform(data))

#区间缩放，返回值为缩放到[0, 1]区间的数据
print(MinMaxScaler().fit_transform(data))

#归一化，返回值为归一化后的数据
print(Normalizer().fit_transform(data))


#二值化，阈值设置为3，返回值为二值化后的数据
print(Binarizer(threshold=3).fit_transform(data))
