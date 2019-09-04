'''
By:shenqiti
2019/9/4

'''

import OF_Algorithm
import data_filter
from data_analysis import prepare_data
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import KFold,cross_val_score,train_test_split
from sklearn.decomposition import PCA
import pandas as pd

def model_compare(data):
    data = data.sample(data.shape[0],random_state=5)
    X = data.drop(columns = ['总分'])
    Y = data['总分']
    pca = PCA(n_components=0.95)
    X = pca.fit_transform(X, Y)
    num_folds = 10
    seed = 7
    scoring = 'accuracy'
    models = [LogisticRegression(), KNeighborsClassifier(), DecisionTreeClassifier(), GaussianNB(), SVC()]
    kfold = KFold(n_splits=num_folds, random_state=seed)
    for model in models:
        cv_results = cross_val_score(model, X , Y, cv=kfold, scoring=scoring)
        # print(cv_results)
        print('交叉验证准确率均值（方差）：%f (%f)' % (cv_results.mean(), cv_results.std()))

if __name__ == '__main__':
    stu_data,n_OHE,stu_info = prepare_data()
    model_compare(stu_data)
    #OF
    print('筛选算法-OF——————————————————————')
    of_0 = OF_Algorithm.OF(dataset=stu_data[stu_data['总分']==0], num_iterations=1000, neighbor_size=50)
    of_1 = OF_Algorithm.OF(dataset=stu_data[stu_data['总分']==1], num_iterations=1000, neighbor_size=50)
    data_filter.compute_frequence((1-of_0['OF值']).tolist(),gap=0.05)
    data_filter.compute_frequence((1-of_1['OF值']).tolist(),gap=0.05)
    data_of = pd.concat([of_0,of_1],axis=0)
    data_of['OF值'] = 1-data_of['OF值']
    threshold = [data_of[data_of['总分']==0]['OF值'].  mean()+2*data_of[data_of['总分']==0]['OF值'].std(),data_of[data_of['总分']==1]['OF值'].mean()+2*data_of[data_of['总分']==1]['OF值'].std()]
    scr_data_OF,data_remove_of = data_filter.screen(data_of,'OF值',threshold=threshold)
    print(scr_data_OF.shape)
    print(scr_data_OF.head())
    model_compare(scr_data_OF)
    #异常值筛选算法2
    print('筛选算法-AE——————————————————————')
    screen_data = data_filter.screen_threshold(stu_data,n_OHE)
    threshold = [screen_data[screen_data['总分']==0]['筛选分数'].mean()+2*screen_data[screen_data['总分']==0]['筛选分数'].std(),screen_data[screen_data['总分']==1]['筛选分数'].mean()+2*screen_data[screen_data['总分']==1]['筛选分数'].std()]
    # print(threshold)
    scr_data,data_remove = data_filter.screen(screen_data, '筛选分数',threshold=threshold)
    print(scr_data.shape)
    model_compare(scr_data)


