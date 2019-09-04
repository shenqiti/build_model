'''
训练模型，包括：
1.特征工程
2.模型调参
3.模型集成
By:shenqiti
2019/9/4
'''
from data_analysis import prepare_data,read_data
from data_filter import screen_threshold,screen
import pandas as pd
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib import pyplot as plt
myfont = FontProperties(fname='C:\Windows\Fonts\simkai.ttf')
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import os
os.chdir("..\\datas")#设置默认数据读取路径
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import accuracy_score,roc_auc_score,classification_report,confusion_matrix,precision_recall_curve,roc_curve,average_precision_score,f1_score
from sklearn.utils.fixes import signature
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier,RandomForestClassifier,ExtraTreesClassifier,VotingClassifier
from sklearn.externals import joblib
import itertools

def spilt_data(data):
    '''
    根据由分布筛选后的特征（feature_select.csv）来划分数据集
    :param data: 需分割的数据集
    :return: 分割后的数据train_x, test_x, train_y, test_y
    '''
    data = data.sample(data.shape[0],random_state=5)
    radio_impute = 0.1
    data_imputation = data.sample(round(stu_data.shape[0] * radio_impute), random_state=5)
    data = data.drop(index=data_imputation.index)
    seed = 7
    X = data.drop(columns='总分')
    Y = data['总分']


    pca = PCA(n_components=0.95)
    X = pca.fit_transform(X, Y)
    print(pca.explained_variance_ratio_)
    joblib.dump(pca, 'pca.joblib')

    imputation_x = pca.transform(data_imputation.drop(columns='总分'))
    imputation_y = data_imputation['总分']

    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=seed)
    return train_x, test_x, train_y, test_y, imputation_x, imputation_y, X,Y

def parameter_adjustment(model,param_grid,X_train, Y_train):
    '''
    模型调参
    :param model:需调参的模型
    :param param_grid: 调参的参数（字典）
    :param X_train: 训练数据x
    :param Y_train: 训练数据y
    :return: 最优模型
    '''
    num_folds = 10
    seed = 7
    scoring = 'accuracy'
    kfold = KFold(n_splits=num_folds, random_state=seed)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
    grid_result = grid.fit(X=X_train, y=Y_train)
    print('最优：%s 使用%s' % (grid_result.best_score_, grid_result.best_params_))
    cv_results = zip(grid_result.cv_results_['mean_test_score'],
                     grid_result.cv_results_['std_test_score'],
                     grid_result.cv_results_['params'])
    for mean, std, param in cv_results:
        print('%f (%f) with %r' % (mean, std, param))
    return grid_result.best_estimator_

def train_models(X_train, Y_train):
    '''
    训练模型，调参得到最好模型，并使用pickl保存
    :param X_train: 训练数据x
    :param Y_train: 训练数据y
    :return: None
    '''
    # models['LDA'] = LinearDiscriminantAnalysis()
    # models['CART'] = DecisionTreeClassifier()
    print('svc------------------------------')
    model = SVC()
    param_grid = {}
    param_grid['C'] = [i / 10 for i in range(3, 10)]
    param_grid['kernel'] = ['linear', 'poly', 'rbf', 'sigmoid']
    param_grid['gamma'] = [0.8, 0.7, 0.6, 0.5, 0.4]
    best_model = parameter_adjustment(model, param_grid, train_x, train_y)
    # joblib.dump(best_model, '..\\models\\SVC.joblib')
    print('lr------------------------------')
    model = LogisticRegression()
    param_grid = {}
    param_grid['C'] = [i / 40 for i in range(1, 20)]
    param_grid['penalty'] = ['l1', 'l2']
    param_grid['tol'] = [0.11, 0.10, 0.09,0.12,0.01]
    best_model = parameter_adjustment(model, param_grid, train_x, train_y)
    # joblib.dump(best_model, '..\\models\\LR.joblib')
    print('knn------------------------------')
    model = KNeighborsClassifier()
    param_grid = [
        {
            'weights':['uniform'],
            'n_neighbors': [i for i in range(3, 15)],
            'p': [i for i in range(1, 6)]
        },
        {
            'weights': ['distance'],
            'n_neighbors': [i for i in range(3, 15)],
            'p': [i for i in range(1, 6)]
        }
    ]
    best_model = parameter_adjustment(model, param_grid, train_x, train_y)
    # joblib.dump(best_model, '..\\models\\KNN.joblib')
    print('nb------------------------------')
    model = GaussianNB()
    param_grid = {}
    best_model = parameter_adjustment(model, param_grid, train_x, train_y)
    # joblib.dump(best_model, '..\\models\\model_nb.joblib')
    print('dt------------------------------')
    model = DecisionTreeClassifier()
    param_grid = {}
    param_grid['criterion'] = ['entropy', 'gini']
    param_grid['min_impurity_decrease'] = np.linspace(0, 1, 10)
    param_grid['max_depth'] = np.arange(3, 10, 1)
    param_grid['min_samples_split'] = np.arange(60, 120, 10)
    best_model = parameter_adjustment(model, param_grid, train_x, train_y)
    # joblib.dump(best_model, '..\\models\\DT.joblib')
    print('RF------------------------------')
    model = RandomForestClassifier(n_estimators=40)
    param_grid = {}
    param_grid['n_estimators'] = [55, 56, 58, 60]  # np.arange(60,80,1)
    param_grid['max_depth'] = np.arange(11, 15, 1)
    param_grid['min_samples_split'] = np.arange(38, 56, 2)
    best_model = parameter_adjustment(model, param_grid, train_x, train_y)
    # joblib.dump(best_model, '..\\models\\RF.joblib')
    print('GBDT------------------------------')
    model = GradientBoostingClassifier(n_estimators=14)
    param_grid = {}
    # param_grid['n_estimators'] = np.arange(1,20,1)
    param_grid['max_depth'] = np.arange(1, 15, 2)
    param_grid['min_samples_split'] = np.arange(80, 201, 20)
    best_model = parameter_adjustment(model, param_grid, train_x, train_y)
    # joblib.dump(best_model, '..\\models\\GBDT.joblib')
    print('AB_DT------------------------------')
    model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy',max_depth=7,min_samples_split=100), algorithm='SAMME')
    param_grid = {}
    param_grid['n_estimators'] = np.arange(20, 60, 10)
    param_grid["learning_rate"] = np.arange(0.06, 0.15, 0.1)
    best_model = parameter_adjustment(model, param_grid, train_x, train_y)
    # joblib.dump(best_model, '..\\models\\AB_DT.joblib')

def ad_parameter(X_train, Y_train):
    print('AB_DT------------------------------')
    model = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(criterion='entropy', max_depth=7, min_samples_split=100),
        algorithm='SAMME')
    param_grid = {}
    param_grid['n_estimators'] = np.arange(20, 60, 10)
    param_grid["learning_rate"] = np.arange(0.06, 0.15, 0.01)
    best_model = parameter_adjustment(model, param_grid, train_x, train_y)
    # joblib.dump(best_model, '..\\models\\AB_DT.joblib')

def voting_model(models,train_x, train_y,test_x,test_y,comb_n=5,voting='hard'):
    result_auc = {}
    result_acc = {}
    result_f1 = {}
    for model in itertools.combinations(models, comb_n):
        print(dict(model).keys())
        model_name = ','.join(list(dict(model).keys()))
        vote_model = VotingClassifier(estimators=model, voting=voting).fit(train_x, train_y)
        vote_result = vote_model.predict(test_x)
        result_auc[model_name] = roc_auc_score(test_y, vote_result)
        result_acc[model_name] = accuracy_score(test_y, vote_result)
        result_f1[model_name] = f1_score(test_y, vote_result)
        print( '投票预测结果:')
        print('auc：', roc_auc_score(test_y, vote_result))
        print('准确率：', accuracy_score(test_y, vote_result))
        print('混淆矩阵：\n', confusion_matrix(test_y, vote_result))
        print('分类报告：\n', classification_report(test_y, vote_result))
    positions = np.arange(1, len(result_acc.keys()) + 1)
    plt.bar(positions, result_acc.values(), width=0.3, label='Accuracy')
    plt.bar(positions + 0.3, result_auc.values(), width=0.3, label='AUC')
    plt.bar(positions + 0.6, result_f1.values(), width=0.3, label='F1')
    for a, b in zip(positions, result_acc.values()):
        plt.text(a, b + 0.005, '%.3f' % b, ha='center', va='bottom')
    for a, b in zip(positions + 0.3, result_auc.values()):
        plt.text(a, b + 0.005, '%.3f' % b, ha='center', va='bottom')
    for a, b in zip(positions + 0.6, result_f1.values()):
        plt.text(a, b + 0.005, '%.3f' % b, ha='center', va='bottom')
    plt.legend()
    plt.ylabel('values')
    plt.xlabel('models')
    plt.ylim((0.6, 0.85))
    plt.xticks(positions, result_acc.keys())
    plt.title("分类模型准确率、AUC值和F1值对比")  # 标题
    plt.show()
    pd.DataFrame(np.array([list(result_acc.values()), list(result_auc.values()),list(result_f1.values())]).T,index=list(result_f1.keys()),columns=['acc','auc','f1'])\
        .to_csv('..\\results\\vote_%s_%d.csv'%(voting,comb_n),index=True, encoding='utf-8 sig')

if __name__ == '__main__':
    #导入数据
    stu_data,n_OHE,stu_info = prepare_data()
    # 筛选数据
    data0 = stu_data
    screen_data = screen_threshold(data0,n_OHE)
    threshold = [
        screen_data[screen_data['总分'] == 0]['筛选分数'].mean() + 2 * screen_data[screen_data['总分'] == 0]['筛选分数'].std(),
        screen_data[screen_data['总分'] == 1]['筛选分数'].mean() + 2 * screen_data[screen_data['总分'] == 1]['筛选分数'].std()]
    stu_data,temp = screen(screen_data, '筛选分数',threshold=threshold)
    print('筛选后的数据个数：',stu_data.shape[0])
    print('筛选后的数据比例：',stu_data[stu_data['总分']==0].shape[0],':',stu_data[stu_data['总分']==1].shape[0])
    print('--------------------------------------------')
    #划分数据集为缺失值处理集、训练集、测试集
    train_x, test_x, train_y, test_y, imputation_x, imputation_y,X,Y = spilt_data(stu_data)
    joblib.dump(imputation_x, 'imputation_x.joblib')
    joblib.dump(imputation_y, 'imputation_y.joblib')
    joblib.dump(X, 'reference_x.joblib')
    joblib.dump(Y, 'reference_y.joblib')
    # 模型训练
    # ad_parameter(train_x,train_y)
    # train_models(train_x,train_y)

    # 评估模型
    print('--------------------------------------------')
    path = '..\\models'
    files = os.listdir(path)
    result_auc = {}
    result_acc = {}
    result_f1 = {}
    models = []
    plt.figure(figsize=(6, 6), dpi=80)
    for i,file in enumerate(files):
        model_name = os.path.splitext(file)[0]
        model = joblib.load(filename=path+'\\'+file)
        # print(model.get_params())
        if model_name in ['SVC','AB_DT','GBDT','LR']:
            models.append((model_name,model))
        predictions = model.predict(test_x)
        result_auc[model_name] = roc_auc_score(test_y, predictions)
        result_acc[model_name] = accuracy_score(test_y, predictions)
        result_f1[model_name] = f1_score(test_y, predictions)
        print(model_name,'预测结果:')
        print('auc：', roc_auc_score(test_y, predictions))
        print('准确率：', accuracy_score(test_y, predictions))
        print('f1：', f1_score(test_y, predictions))
        print('混淆矩阵：\n', confusion_matrix(test_y, predictions))
        print('分类报告：\n', classification_report(test_y, predictions))
    # voting_model(models,train_x,train_y,test_x,test_y)
    # vote_model = VotingClassifier(estimators=models,voting='hard').fit(train_x,train_y)
    # joblib.dump(vote_model, '..\\models\\vote_model_hard.joblib')
    # vote_result = vote_model.predict(test_x)
    # print( '投票预测结果:')
    # print('auc：', roc_auc_score(test_y, vote_result))
    # print('准确率：', accuracy_score(test_y, vote_result))
    # print('混淆矩阵：\n', confusion_matrix(test_y, vote_result))
    # print('分类报告：\n', classification_report(test_y, vote_result))
    # result_auc["集成模型"] = roc_auc_score(test_y, vote_result)
    # result_acc["集成模型"] = accuracy_score(test_y, vote_result)
    # result_f1["集成模型"] = f1_score(test_y, vote_result)

    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 15,
             }

    positions = np.arange(1,len(result_acc.keys())+1)
    plt.bar(positions,result_acc.values(),width=0.3,label='Accuracy')
    plt.bar(positions+0.3, result_auc.values(), width=0.3, label = 'AUC')
    plt.bar(positions + 0.6, result_f1.values(), width=0.3, label='F1')
    for a,b in zip(positions,result_acc.values()):
        plt.text(a, b+0.005, '%.3f' % b, ha='center', va= 'bottom',fontdict=font1)
    for a,b in zip(positions+0.3,result_auc.values()):
        plt.text(a, b+0.005, '%.3f' % b, ha='center', va= 'bottom',fontdict=font1)
    for a,b in zip(positions+0.6,result_f1.values()):
        plt.text(a, b+0.005, '%.3f' % b, ha='center', va= 'bottom',fontdict=font1)
    plt.legend()
    plt.ylabel('values')
    plt.xlabel('models')
    plt.ylim((0.6,0.85))
    plt.xticks(positions,result_acc.keys(),fontsize=15)
    plt.title("分类模型准确率、AUC值和F1值对比") #标题
    plt.show()





