'''
By:shenqiti
2019/11/8
K-means 对国家聚类
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

INPUT_PATH = 'D:/python_code/NEW_step/k-means.xls'
INPUT_PATH2 = 'D:/python_code/NEW_step/K_means_Country.xlsx'
OUTPUT_PATH = 'D:/python_code/NEW_step/k-means_result.xls'
data = np.array(pd.read_excel(INPUT_PATH))
data2 = pd.read_excel(INPUT_PATH2)
name = np.array(data2["Country or Area"])

n_clusters = 4
estimator = KMeans(n_clusters=n_clusters) #构造聚类器
estimator.fit(data)
label_pred = estimator.labels_  #获取聚类标签
print(label_pred)
# fn = open(OUTPUT_PATH,'w')
# for each in label_pred:
#     fn.write(str(each))
#     fn.write('\n')
# fn.close()


markers = ['*', 'o', '+', 's', 'v']
j = 0
for i in range(n_clusters):
    members = label_pred == i  # members是布尔数组
    plt.scatter(data[members, 4], data[members, 7], s=60, marker=markers[i], c='b', alpha=0.5)  # 画与menbers数组中匹配的点
    for a, b in zip(data[members, 4], data[members, 7]):
        plt.text(a, b + 0.0001, name[j], ha='center', va='bottom', fontsize=6)
        j+=1


plt.xlabel("X5_Education index")
plt.ylabel("X8_Gender inequality index")
plt.title('Country of Area')
plt.show()
