
'''单样本z检验
常见用途：检测总体平均值是否等于某个常数
中心极限定理：大量互相独立的随机变量，其均值的分布以正态分布为极限，中心极限定理支持了可以用正态分布来进行均值的判断
单样本z检验案例：原假设：掷硬币朝上和朝下的概率都为50%。连续抛掷硬币1000次，观察到700次朝上，在中心极限定理的前提下，这种情况出现的概率远远小于预先设定的参数，所以选择拒绝原假设
'''

#零假设：复旦大学男生平均身高175cm


#备择假设：复旦大学男生平均身高不为175cm


#需要先构造一个平均值为175，标准差为5，服从正态分布的样本X，样本量为100（你也可以尝试构造平均值为180，标准差为2，服从正态分布的样本，看看使用Z检验的得分区别）

import numpy
X=numpy.random.normal(175,5,100).round(1)

#使用Z检验计算pval

import statsmodels.stats.weightstats
z,pval = statsmodels.stats.weightstats.ztest(X,value=175)

#直接返回的pval，即为P值可以用于判断零假设是否成立



'''单样本t检验
适用：适用于样本量较少的情况（n<30）
参数：t分布的参数为自由度，即样本数量减1（n-1）；当自由度接近于30的时候，t分布趋向于正态分布，这也就说明了为什么t分布适用于样本量较少的情况，因为样本量大于30时，往往可以用z检验
'''
 #零假设：复旦大学男生平均身高175cm
 #备择假设：复旦大学男生平均身高不为175cm
 #使用t检验计算pval
 t,pval=scipy.stats.ttest_1samp(X,popmean=175)#这里的X也需要事先构造
 #直接返回的pval，即为P值可以用于判断零假设是否成立
 
 
 '''双样本t检验
 适用：用于比较两组样本的平均值是否一致
 '''
  #零假设：复旦大学和上海交大男生平均身高一样
 #备择假设：复旦大学和上海交大男生平均身高不一样
 #双样本检验用到了scipy包的另一个函ttest_ind
 #注意，X1，X2需要事先自行创建
 t,pval=scipy.stats.ttest_ind(X1,X2)
 #直接返回的pval，即为P值可以用于判断零假设是否成立
 
 
import statsmodels.stats.weightstats 
import pandas as pd
import numpy as np
#括号里面直接指定了数据的来源，当然你也可以按照老师视频中所讲授的来操作
iris = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=False)
#对导入数据的每一列命名
iris.columns=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species']
#随机抽取10条记录
iris.sample(10)
#单样本z检验
z,pval = statsmodels.stats.weightstats.ztest(iris['SepalLengthCm'],value=175)
#pval=0.0,说明零假设发生的概率为0.0，拒绝零假设
#用数据集真实的均值进行z检验
np.mean(iris('SepalLengthCm'))
z,pval = statsmodels.stats.weightstats.ztest(iris['SepalLengthCm'],value=5.8)
#pval>设定的阈值，选择接受零假设
#尝试对ztest中value的值进行修改，即建立不同的零假设，看看pval有什么样的变化
#单样本t检验
import scipy
t,pval=scipy.stats.ttest_1samp(iris['SepalLengthCm'],popmean=5.8)
#pval>预先设定的阈值，选择接受零假设在小样本上，比较单样本t检验和单样本z检验的结果差异

#因为单样本t检验适用于样本量较小的情况（通常是样本量小于30），随机选取10行iris数据，分别进行t检验和z检验,零假设均为均值为7
iris_sample=iris.sample(10)
##z检验
z,pval_z = statsmodels.stats.weightstats.ztest(iris_sample['SepalLengthCm'],value=7)
##t检验
t,pval_t=scipy.stats.ttest_1samp(iris_sample['SepalLengthCm'],popmean=7)
#根据视频中实验结果，pval_z=0.0003，pval_t=0.006>>0.0003（>>是“远大于”的意思），证明了t检验出现的特别大的值和特别小的值的概率较z检验大一些，这也符合t分布的概率密度曲线尖峰厚尾的特征
##【注意！】这两个数字会随着sample函数抽取到的样本的不同而有变化


#双样本t检验
#对iris进行抽样，形成两个样本
iris_1=iris[iris.Species == 'Iris-setosa']
iris_2=iris[iris.Species == 'Iris-virginica']
t,pval=scipy.stats.ttest_ind(iris_1['SepalLengthCm'],iris_2['SepalLengthCm'])

##pval=6.89e-28
#来验证一下结论是否是正确的
np.mean(iris_1['SepalLengthCm'])
np.mean(iris_2['SepalLengthCm'])
##请手打一遍代码，对比这两个类别的数据均值，你就会理解为什么在t检验中会得到这么小的P值
