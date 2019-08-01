import pandas as pd
import  seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

INPUT_PATH = './data_finall/hour_data/777.csv'

data = pd.read_csv(INPUT_PATH)

#绘制相关系数矩阵热力图
# columns = ['date','pm2.5','DEWP','TEMP','PRES','Iws','Is','Ir']
# kc_train=pd.read_csv(INPUT_PATH,names=columns,encoding='ISO-8859-1')
# kc_train.drop('date',axis=1,inplace=True)

continuous_cols=['1','2','3','4','5','6']
sns.set(font_scale=0.7)
plt.title("The heatmap of the relationship between 6 components")
sns.heatmap(data[continuous_cols].corr(),annot=True,vmin=0,vmax=1)
plt.show()

#
# sns.countplot(x="pm2.5",data=data[1:10])
# sns.barplot(x='pm2.5',y='Iws',data=data[1:10])
# sns.boxplot(x='pm2.5',y='Iws',data=data[1:10])
# plt.show()
