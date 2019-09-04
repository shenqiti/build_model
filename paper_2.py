'''
2019/9/4
By:shenqiti

筛选时变趋势
'''


import numpy as np
import pandas as pd
import time
import datetime
import matplotlib.pyplot as plt
import math
import statsmodels.api as sm
import statsmodels.stats.diagnostic




start = datetime.datetime.fromtimestamp(time.mktime(time.strptime(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),"%Y-%m-%d %H:%M:%S")))
path = 'D:/python_code/NEW_step/try_paper/mycode/results/Poly-22-Params-2019-04-23.csv'
oil_price_path ='D:/python_code/NEW_step/try_paper/data/Data/Brent-11-25.xlsm'

path = 'D:/python_code/NEW_step/try_paper/mycode/new/train_pro.csv'
oil_price_path ='D:/python_code/NEW_step/try_paper/mycode/new/Brent_pro.csv'

trend_df = pd.read_csv(path,header=0)
price_df = pd.read_csv(oil_price_path,header=None)



trenda = np.array(trend_df["MSEtrenda"])
trendb = np.array(trend_df["MSEtrendb"])  #88128
trendc = np.array(trend_df['MSEtrendc'])
trendd = np.array(trend_df["MSEtrendd"])
trende = np.array(trend_df["MSEtrende"])
AA = np.array(trend_df["a"])
BB = np.array(trend_df['b'])




def select_trend(t):
    '''
    选择第t个的最优趋势
    :param t: 第t个
    :return: 最优趋势的a,b，同时返回最优油价
    '''

    for m in range(0,16):
        oil_price = []
        a = AA[t]
        b = BB[t]
        X = a+b
        temp_result = trenda[t + m] * np.power(X, 4) + trendb[t + m] * np.power(X, 3) + trendc[t + m] * np.power(X, 2) + \
                      trendd[t + m] * (X) + trende[t + m]
        oil_price.append(temp_result)
    print(oil_price)

    true_price = np.array(price_df.loc[t:t+15, 1])
    regression_price = []
    evaluation = []
    evaluation.append(abs(true_price-regression_price))
    best_relative_index = evaluation.index(min(evaluation))
    print(min(evaluation))
    best_index = t-best_relative_index
    best_price = regression_price[evaluation.index(min(evaluation))]
    return best_index, best_relative_index, best_price





def main():
    best_trenda = np.zeros(len(trenda)//16)
    best_trendb = np.zeros(len(trendb)//16)
    best_trendc = np.zeros(len(trendc)//16)
    best_trendd = np.zeros(len(trendd)//16)
    best_trende = np.zeros(len(trende)//16)
    best_a = np.zeros(len(AA)//16)
    best_b = np.zeros(len(BB)//16)
    relative_index = []
    cnt = 0
    for i in range(0,len(trenda),16):
        best_index, best_relative_index, best_price = select_trend(i)
        best_trenda[cnt] = trenda[best_index]
        best_trendb[cnt] = trendb[best_index]
        best_trendc[cnt] = trendc[best_index]
        best_trendd[cnt] = trendd[best_index]
        best_trende[cnt] = trende[best_index]
        best_a[cnt] = AA[best_index]
        best_b[cnt] = BB[best_index]
        relative_index.append(best_relative_index)
        x = best_relative_index
        cnt += 1


    result = pd.DataFrame({
        'trenda': best_trenda,
        'trendb': best_trendb,
        'trendc': best_trendc,
        'trendd': best_trendd,
        'trende': best_trende,
        'relative_index': relative_index,
        'a':best_a,
        'b':best_b
    })
    return result




if __name__ == '__main__':
    test_df = main()
    selected_path = 'D:/python_code/NEW_step/try_paper/mycode/new/new2'
    test_df.to_csv(selected_path+'/window_best_trend-'+str(AA)+str(BB)+'-'+str(start)[0:10]+'(3).csv')
    oil_price = []
    print(test_df.head())
    for i in range(test_df.shape[0]):
        # if i <= 5:
        #     print(trend2oil(i))
        x = test_df['relative_index'][i]
        best_price = test_df['trenda'][i]*np.power(x, 4)+test_df['trendb'][i]*np.power(x, 3)\
                     + test_df['trendc'][i]*np.power(x, 2)+test_df['trendd'][i]*x+test_df['trende'][i]
        oil_price.append(best_price)
    price_comparison = pd.DataFrame({
        'Ture Price': price_df.loc[0:5507, 1],
        'Predict Price': oil_price
    })
    price_comparison.to_csv('D:/python_code/NEW_step/try_paper/mycode/new/new2/'+str(AA)+str(BB)+'-'+str(start)[0:10]+'(3).csv')
