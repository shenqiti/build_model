'''
学生数据分析处理，包括：
1.数据的统计描述性分析
2.分数标签y设定（优秀与否80）
3.指标量化（例如课程）
By:shenqiti
2019/9/4

'''
# 导入包
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
myfont =matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simkai.ttf')
plt.rcParams['font.sans-serif']=['SimHei']#画图中文显示问题
import warnings
warnings.filterwarnings('ignore')#忽略wanrning
import os
os.chdir("..\\datas")#设置默认数据读取路径
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

def read_data(path):
    '''
    读取数据（csv文件）
    :param path: 数据读取路径
    :return: 读取的数据
    '''
    f_data = open(path, encoding='utf-8')
    data = pd.read_csv(f_data)
    return data

def trans_col(data):
    '''
    量化指标（课程）
    :param data:需量化的指标列(pd.series)
    :return:量化后的指标列
    '''
    OHE = OneHotEncoder()
    data_OHE = OHE.fit_transform(np.array(data).reshape(-1,1)).toarray()
    data_OHE_df = pd.DataFrame(data=data_OHE)
    return data_OHE_df,data_OHE.shape[1]

def set_label(data,boundary):
    '''
    根据成绩设置标签y,data为分数列（pd.series）
    :param data:分数列
    :param boundary:区分阈值，以优秀与否为例（boundary=80）
    :return:标签列y
    '''
    labels = []
    grade = list(data.astype('float32'))
    for i in grade:
        if i<boundary:
            labels.append(0)
        else:
            labels.append(1)
    return labels

def statistic_info(dataset):
    #数据的统计描述，分布

    # 描述性统计信息
    pd.set_option('precision', 3)
    print(dataset.describe())
    print(dataset.shape)

    # 数据的分类分布
    print(dataset.groupby('总分').size())
    # pd.plotting.scatter_matrix(dataset)
    # plt.show()

    # 直方图
    dataset_hist = pd.DataFrame(dataset.drop(columns='总分').values,index=dataset.index,columns=['Feature'+str(i) for i in range(1,dataset.shape[1])])
    print(dataset_hist.head())
    dataset_hist.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1)
    plt.show()

    # 关系矩阵图
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(dataset.corr(), vmin=-1, vmax=1, interpolation='none')
    fig.colorbar(cax)
    plt.show()

def prepare_data():
    '''
    准备数据：
    1.读取数据
    2.量化指标
    3.转换标签y
    :return:处理后的数据
    '''
    # 读取数据
    stu_data = read_data('stu_data_ustb.csv')
    grade = stu_data['总分']
    cc = set(stu_data['班课号'].tolist())

    # statistic_info(stu_data.drop(columns=['姓名', '学号','课程名','班课号']))

    # 转换课程、分数标签列
    trans_data_temp,n_OHE =  trans_col(data=stu_data['课程名'])
    stu_data_cla = pd.concat((stu_data,trans_data_temp),axis=1)
    stu_data_cla['总分'] = set_label(data=grade, boundary=80)
    stu_info = stu_data_cla[['姓名', '学号','课程名']]
    stu_data_cla.drop(columns=['姓名', '学号','课程名'],inplace=True)
    #归一化处理
    stu_data_out = pd.DataFrame(columns=stu_data_cla.columns).drop(columns=['班课号'])
    scaler = {}
    for c in cc:
        scaler[c] = MinMaxScaler()
        data_scaler = pd.DataFrame(data=scaler[c].fit_transform(stu_data_cla[stu_data_cla['班课号']==c].drop(columns=['班课号'])),
                                   columns=stu_data_out.columns,index=stu_data_cla[stu_data_cla['班课号']==c].index)
        stu_data_out = pd.concat((stu_data_out,data_scaler),axis=0)
    return stu_data_out,n_OHE,stu_info

if __name__ == '__main__':
    stu_data_cla,n_OHE,stu_info = prepare_data()
    #X的标签
    col_x = list(stu_data_cla.columns)
    col_x.remove('总分')
    #数据的统计信息
    statistic_info(stu_data_cla)
