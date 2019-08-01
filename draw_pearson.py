import warnings

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

warnings.filterwarnings('ignore')


# INPUT_PATH = './data_finall/day_data/hour2day.csv'
# INPUT_PATH = './data_finall/day_data/hour2day_PCA.csv'
# INPUT_PATH = './data_finall/day_data/hour2day_norm.csv'
INPUT_PATH = './data_finall/hour_data/777.csv'
# columns = ['TEMP','PRES','Iws','Is','Ir','pm2.5','DEWP']   #hour2day.csv
# columns = ['pm2.5','DEWP','TEMP','PRES','Iws','Is','Ir']    # hour2day_PCA.csv
kc_train=pd.read_csv(INPUT_PATH)
# kc_train.drop('date',axis=1,inplace=True)

print(kc_train.describe())
#绘制各个特征的分布柱状图

kc_train.hist(figsize=(19,15),bins=50,grid=False)


plt.show()

# continuous_cols=['DEWP','TEMP','PRES','Iws','Is','Ir']
# for col in continuous_cols:
#     #sns.jointplot(x=col,y='pm2.5',data=kc_train,height=4,kind='reg')  #,绘制带边缘直方图的散点图,添加回归和内核密度拟合：
#     # sns.jointplot(x=col,y='pm2.5',data=kc_train,kind='hex')  #使用六边形区域用关节直方图替换散点图：
#     sns.jointplot(x=col,y='pm2.5',data=kc_train,kind = "kde", space = 0, color = "g")
#     #用密度估计值替换散点图和直方图，并将边缘轴与关节轴紧密对齐
#     plt.show()
# plt.figure(figsize=(12,6))
# # kc_train.corr(method='spearman')['pm2.5'][['DEWP','TEMP','PRES','Iws']].sort_values(ascending=False).plot("barh",figsize=(12,6),title=
# #         'Variable Correlation(spearman) with pm2.5(After PCA)')
# # #打印的是所有列之间的一个对称矩阵相关关系
# # #method:{'pearson','kendall','spearman'} 默认为pearson
# # plt.show()


