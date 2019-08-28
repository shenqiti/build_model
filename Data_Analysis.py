'''
探索型数据分析实践
'''
import pandas
#括号里面直接指定了数据的来源，当然你也可以按照老师视频中所讲授的来操作
iris = pandas.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
iris.columns=['sepal_length','sepal_width','petal_length','petal_width','species']

import seaborn
%matplotlib inline
seaborn.countplot(x="species",data=iris)
seaborn.barplot(x='species',y='petal_length',data=iris)
seaborn.boxplot(x='species',y='petal_length',data=iris)
#你是否也从上面的三种类型的绘图结果中，对这个数据集有一个初步的印象了呢？
seaborn.distplot(iris['petal_width'])

#Pandas库对类别进行选取，然后进行画图
iris_vir=iris[iris.species == 'Iris-virginica']
iris_s=iris[iris.species == 'Iris-setosa']
#参数赋值，加上label&图例&设置坐标轴范围，xlim设置x轴范围，ylim设置y轴范围
iris_ver=iris[iris.species =='Iris-versicolor']
seaborn.distplot(iris_vir['petal_width'],label='vir').set(ylim=(0,15))
seaborn.distplot(iris_s['petal_width'],label='s')
seaborn.distplot(iris_ver['petal_width'],label='ver').legend()

#FacetGrid 从数据集不同的侧面进行画图，hue指定用于分类的字段，使得代码会更加简洁

#尝试修改row/col参数，替代hue参数，row:按行展示，col：按列展示
g=seaborn.FacetGrid(iris,row='species')
g.map(seaborn.distplot,'petal_width').add_legend()

#画出线性回归的曲线
seaborn.regplot(x='petal_width',y='petal_length',data=iris)
#分类画线性回归
g = seaborn.FacetGrid(iris,hue='species')
#设置坐标轴范围
g.set (xlim=(0,2.5))
g.map(seaborn.regplot,'petal_width','petal_length').add_legend()

#不显示拟合曲线,用matplotlib画散点图
import matplotlib.pyplot as plt
g.map(plt.scatter,'petal_width','petal_length').add_legend()

'''
boxplot函数有这么五行的参数:

x,y :分别代表坐标轴的命名
hue:指定用于分类的字段
data:数据来源
order：分类数据的绘图顺序
color：元素的颜色
linewidth：线宽

seaborn.set_style('parameter'):

darkgrid 黑色网格（默认）
whitegrid 白色网格
dark 黑色背景无网格
white 白色背景无网格
ticks 白色背景无网格，坐标有刻度 
尝试使用这个指令，用不同的背景绘图
'''

