'''
筛选数据：以“种瓜得瓜”的策略，筛选学生数据中明显导致分类错误的数据
By:shenqiti
2019/9/4
'''
from data_analysis import prepare_data,read_data
import pandas as pd
import numpy as np
import math
import os
os.chdir("..\\datas")
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score,recall_score,f1_score,roc_auc_score,classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import KFold,cross_val_score

def screen_threshold(screen_data,n_OHE):
    '''
    筛选分数：将数据减去平均值，计算数据中大于0，或者小于0的绝对值的和
    计算筛选分数，并根据筛选分数计算累积百分比
    :param screen_data: 需筛选的训练数据x
    :return: 训练数据x+筛选分数列
    '''
    col_drop = ['总分']+[i for i in range(n_OHE)]
    cols = screen_data.drop(columns=col_drop).columns
    screen_data['筛选分数'] = None
    for grade in [0,1]:
        data = screen_data[screen_data['总分']==grade][cols]
        ae = []
        for i in range(data.shape[0]):
            temp = (data.iloc[i]-data.mean())
            if grade==0:
                value = math.fabs(temp[temp>0].sum())
            if grade==1:
                value = math.fabs(temp[temp<0].sum())
            ae.append(value)
            screen_data.loc[data.index.tolist()[i],'筛选分数'] = value
        compute_frequence(ae)

    return screen_data

def compute_frequence(data,gap=0.5):
    # 计算累积百分比，以0.5为间隔
    scope = np.arange(min(data), max(data), gap)
    frequency = pd.cut(data, scope).value_counts()
    fr_df = pd.DataFrame(frequency, columns=['频数'])
    fr_df['频率f'] = fr_df / fr_df['频数'].sum()
    fr_df['频率%'] = fr_df['频率f'].map(lambda x: '%.2f%%' % (x * 100))
    fr_df['累计频率f'] = fr_df['频率f'].cumsum()
    fr_df['累计频率%'] = fr_df['累计频率f'].map(lambda x: '%.4f%%' % (x * 100))
    print('累积百分比：')
    print(fr_df)
    return fr_df

def screen(screen_data,col,radio=None,threshold=None):
    '''
    根据筛选阈值，分别对0，1类进行筛选
    :param screen_data: 需筛选的数据x
    :param threshold: 筛选阈值
    :return: 筛选后的数据
    '''
    if threshold !=None:
        data0 = screen_data[screen_data['总分'] == 0]
        data0_hold = data0[data0[col] <= threshold[0]]
        data0_remove = data0[data0[col] > threshold[0]]

        data1 = screen_data[screen_data['总分'] == 1]
        data1_hold = data1[data1[col] <= threshold[1]]
        data1_remove = data1[data1[col] > threshold[1]]

        data = pd.concat([data0_hold, data1_hold], axis=0)
        data_ad = pd.concat([data0_remove, data1_remove], axis=0)
        return data.drop(columns=col), data_ad.drop(columns=col)

    else:
        data0 = screen_data[screen_data['总分'] == 0].sort_values(col, axis=0, ascending=True)
        n0 = round(data0.shape[0]*radio[0])
        data0_hold = data0.iloc[[i for i in range(n0)]]
        data0_remove = data0.drop(index = data0_hold.index)

        data1 = screen_data[screen_data['总分'] == 1].sort_values(col, axis=0, ascending=True)
        n1 = round(data1.shape[0] * radio[1])
        data1_hold = data1.iloc[[i for i in range(n1)]]
        data1_remove = data1.drop(index=data1_hold.index)

        data = pd.concat([data0_hold,data1_hold],axis=0)
        data_ad = pd.concat([data0_remove,data1_remove],axis=0)
        return data.drop(columns=col),data_ad.drop(columns=col)

def cv_model(X,Y,model):
    num_folds = 10
    seed = 7
    scoring = 'accuracy'
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X , Y, cv=kfold, scoring=scoring)
    # print(cv_results)
    print('交叉验证准确率均值（方差）：%f (%f)' % (cv_results.mean(), cv_results.std()))
    return model.fit(X,Y),cv_results.mean(),cv_results.std()

if __name__ == '__main__':
    stu_data,n_OHE,stu_info = prepare_data()
    data0 = stu_data
    screen_data = screen_threshold(data0,n_OHE)
    models = [LogisticRegression(),KNeighborsClassifier(),DecisionTreeClassifier(),GaussianNB(),SVC()]
    for no,model in enumerate(models):
        report = pd.DataFrame(data=None,index=np.arange(0.50,0.99,0.01,dtype=float),columns=['accuracy','TP','TN','FP','FN','cv_acc_mean','cv_acc_std'])
        for i in report.index:
            scr_data,data_ad = screen(screen_data,'筛选分数',radio=[i,i])
            print(scr_data.shape)

            scr_data = scr_data.sample(scr_data.shape[0],random_state=5)
            train_x = scr_data.drop(columns = ['总分'])
            train_y = scr_data['总分']
            test_x = data_ad.drop(columns = ['总分'])
            test_y = data_ad['总分']
            model,cv_acc_mean,cv_acc_std = cv_model(train_x,train_y,model)
            predictions = model.predict(test_x)
            report.loc[i,'cv_acc_mean'] = cv_acc_mean
            report.loc[i, 'cv_acc_std'] = cv_acc_std
            report.loc[i,'accuracy'] = roc_auc_score(test_y, predictions)
            report.loc[i, 'recall'] = recall_score(test_y, predictions)
            report.loc[i, 'f1'] = f1_score(test_y, predictions)
            report.loc[i, 'TP'] = confusion_matrix(test_y, predictions)[0, 0]
            report.loc[i, 'TN'] = confusion_matrix(test_y, predictions)[0, 1]
            report.loc[i, 'FP'] = confusion_matrix(test_y, predictions)[1, 0]
            report.loc[i, 'FN'] = confusion_matrix(test_y, predictions)[1, 1]
            print('auc：', roc_auc_score(test_y, predictions))
            print('准确率：', accuracy_score(test_y, predictions))
            print('混淆矩阵：\n', confusion_matrix(test_y, predictions))
            print('分类报告：\n', classification_report(test_y, predictions))
            print(i,'-------------------------------------------------------------')

        report.to_csv('..\\results\\screen_radio_compare_%d.csv'%no, index=True, encoding='utf-8 sig')

    # stu_data,n_OHE,stu_info = prepare_data()
    # data0 = stu_data
    # screen_data = screen_threshold(data0,n_OHE)
    # scr_data, data_ad = screen(screen_data, '筛选分数', radio=[0.91,0.91])
    # print(scr_data.shape)
    #
    # pca = joblib.load('..\\datas\\pca.joblib')
    #
    # scr_data = scr_data.sample(scr_data.shape[0],random_state=5)
    # train_x = scr_data.drop(columns = ['总分'])
    # train_y = scr_data['总分']
    # test_x = data_ad.drop(columns = ['总分'])
    # test_y = data_ad['总分']
    # test_x = pca.transform(test_x)
    # path = '..\\models'
    # files = os.listdir(path)
    # models = []
    # for file in files:
    #     model_name = os.path.splitext(file)[0]
    #     model = joblib.load(filename=path+'\\'+file)
    #     predictions = model.predict(test_x)
    #     print(model_name,'准确率：', accuracy_score(test_y, predictions))


