'''
插补算法比较
By:shenqiti
2019/9/4
'''
from SOM_Algorithm import SOM
import warnings
warnings.filterwarnings("ignore")
import os
os.chdir("..\\datas")
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib import pyplot as plt
myfont = FontProperties(fname='C:\Windows\Fonts\simkai.ttf')
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from impyute.imputation.cs import mice,em
from impyute.dataset.corrupt import Corruptor
from impyute.util import find_null
import fancyimpute
from sklearn.externals import joblib
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.metrics import accuracy_score,mutual_info_score
from sklearn.mixture import GaussianMixture

def impute(data,method='knn',n=5):
    if method=='knn':
        data_impute = fancyimpute.KNN(k=n).fit_transform(data)
    if method == 'mice':
        data_impute_list=[]
        for i in range(11):
            imputer = fancyimpute.IterativeImputer(n_iter=13, sample_posterior=True, random_state=i)
            data_impute_list.append(imputer.fit_transform(data))
        data_impute = np.mean(data_impute_list,0)
        # data_impute = mice(data)
    if method == 'em':
        data_impute = em(data)
    if method == 'mean':
        data_impute = fancyimpute.simple_fill.SimpleFill(fill_method='mean').fit_transform(data)
    return data_impute

def impute_SOM(data,n):
    imputer = SimpleImputer(missing_values=np.nan,strategy='constant',fill_value=0)
    data_fill0 = imputer.fit_transform(data)
    data_miss = data.copy()

    som = SOM(data_fill0, (n, n), 3, 300)
    som.train()
    res = np.array(som.train_result()).reshape(-1,1)
    null_set = find_null(data)
    for i1,i2 in null_set:
        neighbor = som.getneighbor(res[i1][0], 1)
        activation_group = neighbor[0]
        for j in neighbor:
            activation_group = activation_group|j
        activation_group = np.array([som.W.T[p] for p in activation_group])
        data_miss[i1,i2] = activation_group.mean(axis=0)[i2]
    return data_miss

def impute_parameter_adjustment(method,param_grid,impute_radio,x_init, y_init, reference_x, reference_y):
    model = joblib.load('..\\models\\vote_model_hard.joblib')
    markers = ['o','*','1','s','2']
    I=20
    for radio,marker in zip(impute_radio,markers):
        acc_1 = {i:0 for i in param_grid}
        acc_2 = {i:0 for i in param_grid}
        for m in range(I):
            corruptor = Corruptor(x_init, radio)
            x_miss = getattr(corruptor, "mcar")()
            for n in param_grid:
                if method=='knn':
                    x_impute = fancyimpute.KNN(k=n).fit_transform(np.vstack((x_miss,reference_x)))[range(x_init.shape[0])]
                if method == 'mice':
                    data_impute_list=[]
                    for i in range(n):
                        imputer = fancyimpute.IterativeImputer(n_iter=13, sample_posterior=True, random_state=i)
                        data_impute_list.append(imputer.fit_transform(np.vstack((x_miss,reference_x)))[range(x_init.shape[0])])
                    x_impute = np.mean(data_impute_list,0)
                    print(radio,m,n)
                if method == 'em':
                    x_impute = em(np.vstack((x_miss,reference_x)),loops=n)[range(x_init.shape[0])]
                if method == 'som':
                    x_impute = impute_SOM(x_miss, n)[range(x_init.shape[0])]
                y_pred1 = model.predict(x_impute)
                y_pred2 = model.predict(x_init)
                acc_1[n] += 1-accuracy_score(y_pred1, y_pred2)
                acc_2[n] += 1-accuracy_score(y_pred1, y_init)
        acc_1 = {i: (j/I) for i,j in acc_1.items()}
        acc_2 = {i: (j / I) for i, j in acc_2.items()}
        plt.subplot(121)
        plt.plot(acc_1.keys(), acc_1.values(),marker=marker,label='%.1f%%'%(radio*100))
        plt.xlabel('K')
        plt.ylabel('CER between imputation and prediction')
        plt.subplot(122)
        plt.plot(acc_2.keys(), acc_2.values(),marker=marker,label='%.1f%%'%(radio*100))
        plt.xlabel('K')
        plt.ylabel('CER between imputation and real label')
        plt.legend(loc=0,bbox_to_anchor =(0.3,-0.05),ncol =5)
    plt.show()

def CER(x_impute,x_init,y_init,cer=1):
    model = joblib.load('..\\models\\vote_model_hard.joblib')
    y_pred1 = model.predict(x_impute)
    y_pred2 = model.predict(x_init)
    if cer==1:
        return 1 - accuracy_score(y_pred1, y_pred2)
    else:
        return 1-accuracy_score(y_pred1, y_init)

if __name__ == '__main__':
    x_init = np.array(joblib.load('imputation_x.joblib'))
    y_init = np.array(joblib.load('imputation_y.joblib'))
    reference_x = np.array(joblib.load('reference_x.joblib'))
    reference_y = np.array(joblib.load('reference_y.joblib'))
    # X=np.vstack((x_init,reference_x))
    # Y = np.hstack((y_init,reference_y))
    # MIC = []
    # for i in range(X.shape[1]):
    #     MIC.append(mutual_info_score(X[:,i],Y))
    # print(MIC.index(max(MIC)))
    # print(MIC)
    #
    # impute_parameter_adjustment('knn',np.arange(1,40,2),[0.05,0.1,0.2,0.3,0.4],x_init, y_init, reference_x, reference_y)
    # impute_parameter_adjustment('mice', np.arange(1, 15, 1), [0.05, 0.1, 0.2, 0.3, 0.4], x_init, y_init, reference_x,
    #                             reference_y)
    # impute_parameter_adjustment('em', np.arange(10, 100, 5), [0.05, 0.1, 0.2, 0.3, 0.4], x_init, y_init, reference_x,
    #                             reference_y)
    # impute_parameter_adjustment('som', np.arange(10, 20, 1), [0.05, 0.1, 0.2, 0.3, 0.4], x_init, y_init, reference_x,
    #                             reference_y)

    MAE_result = pd.DataFrame(data=None,columns=['MEAN','MICE','EM','KNN_3',"SOM"],index=[5,10,20,30,40])
    CER_result = pd.DataFrame(data=None, columns=['MEAN', 'MICE', 'EM','KNN_3', "SOM"],index=[5,10,20,30,40])
    CER_result_1 = pd.DataFrame(data=None, columns=['MEAN', 'MICE', 'EM', 'KNN_3', "SOM"], index=[5, 10, 20, 30, 40])

    for impute_radio in [0.05,0.1,0.2,0.3,0.4]:
        mae_df = pd.DataFrame(data=None,columns=['MEAN','MICE','EM','KNN_3',"SOM"],index=np.arange(0,20))
        cer_df = pd.DataFrame(data=None, columns=['MEAN', 'MICE','EM', 'KNN_3', "SOM"],index=np.arange(0,20))
        cer_df_1 = pd.DataFrame(data=None, columns=['MEAN', 'MICE', 'EM', 'KNN_3', "SOM"], index=np.arange(0, 20))
        for j in range(20):
            corruptor = Corruptor(x_init,impute_radio)
            x_miss = getattr(corruptor, "mcar")()
            print(len(list(x_miss[np.isnan(x_miss)])))
            x_miss = np.vstack((x_miss,reference_x))
            col_num = 0
            #mean、mice插补
            for i in ['mean','mice']:
                print(i,'插补---------------------------------')
                x_impute = impute(x_miss,i)[range(x_init.shape[0])]
                mae_df.iloc[j,col_num] = sum(sum(np.abs(x_impute-x_init)))
                cer_df.iloc[j,col_num] = CER(x_impute,x_init,y_init,1)
                cer_df_1.iloc[j, col_num] = CER(x_impute, x_init, y_init, 2)
                col_num +=1
            #KNN插补（n=3、5、10）
            for n in [11]:
                print('KNN-',n,'插补---------------------------------')
                x_impute = impute(x_miss, 'knn', n)[range(x_init.shape[0])]
                mae_df.iloc[j, col_num] = sum(sum(np.abs(x_impute - x_init)))
                cer_df.iloc[j, col_num] = CER(x_impute, x_init, y_init, 1)
                cer_df_1.iloc[j, col_num] = CER(x_impute, x_init, y_init, 2)
                col_num += 1
            #SOM插补
            print('SOM插补---------------------------------')
            x_impute = impute_SOM(x_miss,17)[range(x_init.shape[0])]
            mae_df.iloc[j, col_num] = sum(sum(np.abs(x_impute - x_init)))
            cer_df.iloc[j, col_num] = CER(x_impute, x_init, y_init, 1)
            cer_df_1.iloc[j, col_num] = CER(x_impute, x_init, y_init, 2)
            col_num += 1
            print(impute_radio,j)
        MAE_result.loc[impute_radio * 100] = ['%.3f±%.3f'%(m,s) for m,s in zip(list(mae_df.mean()),list(mae_df.std()))]
        CER_result.loc[impute_radio * 100] = ['%.3f±%.3f'%(m,s) for m,s in zip(list(cer_df.mean()),list(cer_df.std()))]
        CER_result_1.loc[impute_radio * 100] = ['%.3f±%.3f' % (m, s) for m, s in
                                              zip(list(cer_df_1.mean()), list(cer_df_1.std()))]
    # MAE_result.to_csv('..\\results\\impute_MAE.csv', index=True, encoding='utf-8 sig')
    # CER_result.to_csv('..\\results\\impute_CER.csv', index=True, encoding='utf-8 sig')
    CER_result_1.to_csv('..\\results\\impute_CER_1.csv', index=True, encoding='utf-8 sig')
