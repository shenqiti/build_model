import pandas as pd
from sklearn.decomposition import PCA

#参数初始化
# inputfile = './data_finall/day_data/feature_xg1_norm.csv'
# inputfile = './data_finall/day_data/hour2day_norm.csv'
inputfile = './data_finall/day_data/data_Scenarios_Analysis_norm.csv'
outputfile = './data_finall/day_data/data_Scenarios_Analysis_norm_PCA.csv' #降维后的数据

data = pd.read_csv(inputfile) #读入数据


pca = PCA()
pca.fit(data)
com=pca.components_ #返回具有最大方差的成分。
pca=pca.explained_variance_ratio_ #返回各个成分各自的方差百分比
print (com,"百分比：\n",pca)
# print ("百分比：\n",pca)

pca=PCA(6)
pca.fit(data)
low_d=pca.transform(data)#降维

pd.DataFrame(low_d).to_csv(outputfile)
pca.inverse_transform(low_d)
