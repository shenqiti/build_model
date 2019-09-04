'''
By:shenqiti
2019/9/4

'''

import data_filter
from data_analysis import prepare_data,read_data
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from math import ceil,pow
from sklearn.metrics import accuracy_score,recall_score,f1_score,roc_auc_score,classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold,cross_val_score


def OF(dataset,num_iterations,neighbor_size):
    #输入数据集dataset（pandas.DataFrame）、迭代次数num_iterations、邻居数neighbor_size
    rep = num_iterations
    epsilon = k_distance(dataset,neighbor_size)
    M = F(dataset.shape[0],rep)

    index = dataset.index
    dataset.reset_index(drop=True,inplace=True)

    delta = dict(zip(list(dataset.index),[0 for i in list(dataset.index)]))
    for r in range(1,rep,1):
        S = dataset.sample(n=M)
        Ni = []
        for i in S.index:
            Ni += find_Neibor(dataset=dataset, s=np.array(S.loc[i]).reshape(1,-1), epsilon=epsilon)
        Ni = set(Ni)
        for i in dataset.index:
            if i in Ni:
                delta[i] +=1
    dataset_op = dataset.copy()
    dataset_op['OF值'] = pd.Series(list(delta.values()))/rep
    # print(dataset_op.head())
    dataset.index = index
    return dataset_op

def k_distance(dataset,k):
    nbrs = NearestNeighbors(n_neighbors=k).fit(dataset)
    distances, indices = nbrs.kneighbors(dataset)
    return distances[:,k-1].mean()

def find_Neibor(dataset,s,epsilon):
    nbrs = NearestNeighbors(radius=epsilon).fit(dataset)
    distances, indices = nbrs.radius_neighbors(s)
    return indices[0].tolist()

def F(n,rep,confidence=0.99):
    M = ceil(n*(1-pow(1-confidence,1/rep)))
    return M

def cv_model(X,Y):
    num_folds = 10
    seed = 7
    scoring = 'accuracy'
    model = LogisticRegression()
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X , Y, cv=kfold, scoring=scoring)
    # print(cv_results)
    print('交叉验证准确率均值（方差）：%f (%f)' % (cv_results.mean(), cv_results.std()))
    return model.fit(X,Y),cv_results.mean(),cv_results.std()

if __name__ == '__main__':
    stu_data, n_OHE, stu_info = prepare_data()
    data0 = stu_data
    of_0 = OF(dataset=stu_data[stu_data['总分'] == 0], num_iterations=1000, neighbor_size=50)
    of_1 = OF(dataset=stu_data[stu_data['总分'] == 1], num_iterations=1000, neighbor_size=50)
    data_filter.compute_frequence((1 - of_0['OF值']).tolist(), gap=0.05)
    data_filter.compute_frequence((1 - of_1['OF值']).tolist(), gap=0.05)
    data_of = pd.concat([of_0, of_1], axis=0)
    data_of['OF值'] = 1 - data_of['OF值']

    report = pd.DataFrame(data=None, index=np.arange(0.50, 0.99, 0.01, dtype=float),
                          columns=['accuracy','recall','f1', 'TP', 'TN', 'FP', 'FN', 'cv_acc_mean', 'cv_acc_std'])
    for i in report.index:
        scr_data, data_ad = data_filter.screen(data_of, 'OF值', radio=[i, i])
        print(scr_data.shape)

        scr_data = scr_data.sample(scr_data.shape[0], random_state=5)
        train_x = scr_data.drop(columns=['总分'])
        train_y = scr_data['总分']
        test_x = data_ad.drop(columns=['总分'])
        test_y = data_ad['总分']
        model, cv_acc_mean, cv_acc_std = cv_model(train_x, train_y)
        predictions = model.predict(test_x)
        report.loc[i, 'cv_acc_mean'] = cv_acc_mean
        report.loc[i, 'cv_acc_std'] = cv_acc_std
        report.loc[i, 'accuracy'] = roc_auc_score(test_y, predictions)
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
        print(i, '-------------------------------------------------------------')

    report.to_csv('..\\results\\of_screen_radio_compare.csv', index=True, encoding='utf-8 sig')
