'''
Content：

格式转换
缺失数据
异常数据
数据标准化操作

'''

#读取数据
import pandas
users=pandas.read_csv("train_users_2.csv")

#首先需要进行的是对数据的基本查看
#第一行是属性的名称，index从0开始，NaN代表missing value
users.head()
#和head相反，tail给出了数据集末尾的值
users.tail()
users.describe()
users.shape
users.head()
uesers.loc[0:3,"age"]

#格式转换：转变为日期格式,可以实现时间的加减
users["date_account_created"] = pandas.to_datetime(users["date_account_created"])

users.loc[0,"date_account_created"]-users.loc[1,"date_account_created"]

#定义到时分秒的关系，设置format函数参数，针对一些非常规的数据
users["timestamp_first_active"] = pandas.to_datetime(users["timestamp_first_active"],format="%Y%m%d%H%M%S")

#缺失数据处理：查看去掉缺失值的数据
users["age"].dropna()

#画图,可以很直观的观察数据的异常值
import seaborn
seaborn.distplot(users["age"].dropna())
seaborn.boxplot(users["age"].dropna())

#异常数据处理：筛选age<90以及>10
users_with_true_age=users[users["age"]<90]
users_with_true_age=users_with_true_age[users_with_true_age["age"]>10]

'''
格式转换：使用pandas.to_datetime函数转化为标准的时间格式
缺失数据处理：dropna()函数进行缺失值处理
异常数据处理：设置条件筛选”age”字段<90以及>10

'''
