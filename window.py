
'''
By：shenqiti
2019/7/9
滑动窗口  
'''

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.tsa.api as st

path = 'D:/python_code/NEW_step/try_paper/data/best_trend/window_best_trend-2019-04-22(3).csv'
oil_price_path = 'D:/python_code/NEW_step/try_paper/data/Data/Brent-11-25.xlsm'
df = pd.read_csv(path,header=0)
price_df = pd.read_excel(oil_price_path,header=None,sheet_name='Sheet1')

print(df.loc[1,'trenda'])

trenda = np.array(df.loc[:, 'trenda'].astype('float64'))
trendb = np.array(df.loc[:, 'trendb'].astype('float64'))
trendc = np.array(df.loc[:, 'trendc'].astype('float64'))
trendd = np.array(df.loc[:, 'trendd'].astype('float64'))
trende = np.array(df.loc[:, 'trende'].astype('float64'))
relative_index = np.array(df.loc[:, 'relative_index']).astype('float64')
trenda1 = [""]
trendb1 = [""]
trendc1 = [""]
trendd1 = [""]
trende1 = [""]
relative_index1 = ['']
trenda2 = ["", ""]
trendb2 = ["", ""]
trendc2 = ["", ""]
trendd2 = ["", ""]
trende2 = ["", ""]
relative_index2 = ['', '']
trenda3 = ["", "", ""]
trendb3 = ["", "", ""]
trendc3 = ["", "", ""]
trendd3 = ["", "", ""]
trende3 = ["", "", ""]
relative_index3 = ['', '', '']
trenda4 = ["", "", "", ""]
trendb4 = ["", "", "", ""]
trendc4 = ["", "", "", ""]
trendd4 = ["", "", "", ""]
trende4 = ["", "", "", ""]
relative_index4 = ['', '', '', '']
trenda5 = ["", "", "", "", ""]
trendb5 = ["", "", "", "", ""]
trendc5 = ["", "", "", "", ""]
trendd5 = ["", "", "", "", ""]
trende5 = ["", "", "", "", ""]
relative_index5 = ['', '', '', '', '']
'''
以下循环构造滞后各项的数组
'''
for i in range(len(trenda) - 1):
    trenda1.append(trenda[i])
    trendb1.append(trendb[i])
    trendc1.append(trendc[i])
    trendd1.append(trendd[i])
    trende1.append(trende[i])
    relative_index1.append(relative_index[i])
    if i <= len(trenda) - 3:
        trenda2.append(trenda[i])
        trendb2.append(trendb[i])
        trendc2.append(trendc[i])
        trendd2.append(trendd[i])
        trende2.append(trende[i])
        relative_index2.append(relative_index[i])
    if i <= len(trenda) - 4:
        trenda3.append(trenda[i])
        trendb3.append(trendb[i])
        trendc3.append(trendc[i])
        trendd3.append(trendd[i])
        trende3.append(trende[i])
        relative_index3.append(relative_index[i])
    if i <= len(trenda) - 5:
        trenda4.append(trenda[i])
        trendb4.append(trendb[i])
        trendc4.append(trendc[i])
        trendd4.append(trendd[i])
        trende4.append(trende[i])
        relative_index4.append(relative_index[i])
    if i <= len(trenda) - 6:
        trenda5.append(trenda[i])
        trendb5.append(trendb[i])
        trendc5.append(trendc[i])
        trendd5.append(trendd[i])
        trende5.append(trende[i])
        relative_index5.append(relative_index[i])


def getRegressResult(a, b, alpha):
    '''通过线性回归求趋势向量自回归系数，被注释的c和d是四维趋势向量用到的'''

    xa = []
    xb = []
    xc = []
    xd = []
    xe = []
    x_index = []
    ya = []
    yb = []
    yc = []
    ye = []
    yd = []
    y_index = []
    for i in range(a, b + 1):
        xa.append([trenda1[i], trenda2[i], trenda3[i], trenda4[i], trenda5[i]])
        xb.append([trendb1[i], trendb2[i], trendb3[i], trendb4[i], trendb5[i]])
        xc.append([trendc1[i], trendc2[i], trendc3[i], trendc4[i], trendc5[i]])
        xd.append([trendd1[i], trendd2[i], trendd3[i], trendd4[i], trendd5[i]])
        xe.append([trende1[i], trende2[i], trende3[i], trende4[i], trende5[i]])
        x_index.append([relative_index1[i], relative_index2[i], relative_index3[i]])
        ya.append(trenda[i])
        yb.append(trendb[i])
        yc.append(trendc[i])
        yd.append(trendd[i])
        ye.append(trende[i])
        y_index.append(relative_index[i])
    xa = np.array(xa)
    xb = np.array(xb)
    xc = np.array(xc)
    xd = np.array(xd)
    xe = np.array(xe)
    x_index = np.array(x_index)
    xa = sm.add_constant(xa)
    xb = sm.add_constant(xb)
    xc = sm.add_constant(xc)
    xd = sm.add_constant(xd)
    xe = sm.add_constant(xe)
    x_index = sm.add_constant(x_index)

    resulta = sm.OLS(ya, xa).fit()
    resultb = sm.OLS(yb, xb).fit()
    resultc = sm.OLS(yc, xc).fit()
    resultd = sm.OLS(yd, xd).fit()
    resulte = sm.OLS(ye, xe).fit()
    result_index = sm.OLS(y_index, x_index).fit()

    #   print(resulta.summary(),resultb.summary(),resultc.summary())
    '''
	这一段本来想按照显著性水平筛掉一些结果，发现没什么用，可删掉
    for i in range(len(resulta.pvalues)):
        if(resulta.pvalues[i]>alpha):
            resulta.params[i]=0
    for i in range(len(resultb.pvalues)):
        if(resultb.pvalues[i]>alpha):
            resultb.params[i]=0

    for i in range(len(resultc.pvalues)):
        if(resultc.pvalues[i]>alpha):
            resultc.params[i]=0

     '''
    # 获取回归系数数组
    ka = resulta.params
    kb = resultb.params
    kc = resultc.params
    kd = resultd.params
    ke = resulte.params
    k_index = result_index.params
    return ka, kb, kc, kd, ke, k_index


# 设定显著性水平
alpha = 0.1
# 窗口长度
step = 22


def getResultForPredict(a, b):
    '''
	根据向量自回归方程获取在区间(b+1,b+1+step)上的趋势向量分量，
	所以后面数组无论trenda还是trendb都在b处取值，
	因为想根据已知区间的终点b趋势预测未知区间的起点b+1趋势
	'''
    #   print(b,trenda[b],trendb[b])
    # 获取向量回归参数
    ka, kb, kc, kd, ke, k_index = getRegressResult(a, b, alpha)
    #  print(ka,kb)
    ya_ols = ka[0] + ka[1] * trenda1[b] + ka[2] * trenda2[b] + ka[3] * trenda3[b] + ka[4] * trenda4[b] + ka[5] * \
             trenda5[b]
    yb_ols = kb[0] + kb[1] * trendb1[b] + kb[2] * trendb2[b] + kb[3] * trendb3[b] + kb[4] * trendb4[b] + kb[5] * \
             trendb5[b]
    yc_ols = kc[0] + kc[1] * trendc1[b] + kc[2] * trendc2[b] + kc[3] * trendc3[b] + kc[4] * trendc4[b] + kc[5] * \
             trendc5[b]
    yd_ols = kd[0] + kd[1] * trendd1[b] + kd[2] * trendd2[b] + kd[3] * trendd3[b] + kd[4] * trendd4[b] + kd[5] * \
             trendd5[b]
    ye_ols = ke[0] + kd[1] * trende1[b] + kd[2] * trende2[b] + ke[3] * trende3[b] + ke[4] * trende4[b] + ke[5] * \
             trende5[b]
    y_index_ols = k_index[0] + k_index[1] * relative_index1[b] + k_index[2] * relative_index2[b] + k_index[3] * \
                  relative_index3[b]
    return ya_ols, yb_ols, yc_ols, yd_ols, ye_ols, y_index_ols


def main():
    # 设置回归区间
    a = 5;
    # b=df.shape[0]-1-1
    avError = 0
    mape = 0
    sse = 0
    # 滚动向前一步预测，思路是每次向前一步，将结果取平均
    # 比如 第一次用(a,b)预测(b+1,step+b+1)的趋势
    # 第二次用(a,b+1)预测(b+2,step+b+2)的趋势，以此类推
    oil_price_list = []
    for i in range(df.shape[0] - 6, df.shape[0] - 1):
        ya_ols, yb_ols, yc_ols, yd_ols, ye_ols, index_ols = getResultForPredict(a, i)
        print('ya_ols', ya_ols, 'yb_ols', yb_ols, 'yc_ols', yc_ols, 'yd_ols', yd_ols, 'ye_ols', ye_ols)
        print('df.loc[i-1,"trenda"]', df.loc[i - 1, 'trenda'], 'df.loc[i-1,"trendb"]', df.loc[i - 1, 'trendb'],
              'pre_trende', df.loc[i - 1, 'trende'])
        # TODO x还可以从经验分布里随机抽取
        x = 0
        # print('index_ols', index_ols)
        #         prefit_x = df.loc[i-1,'relative_index']
        #         pre_fitting_price = df.loc[i-1,'trenda']*np.power(prefit_x,4)+df.loc[i-1,'trendb']*np.power(prefit_x,3)+df.loc[i-1, 'trendc']*np.power(prefit_x,2)\
        #                         +df.loc[i-1, 'trendd']*prefit_x+df.loc[i-1, 'trende']
        print('pre_price', price_df.loc[i - 1, 1])

        fit_x = df.loc[i, 'relative_index']
        oil_price = ya_ols * (np.power(x, 4)) + yb_ols * (np.power(x, 3)) + yc_ols * (
            np.power(x, 2)) + yd_ols * x + ye_ols
        fitting_price = df.loc[i, 'trenda'] * np.power(fit_x, 4) + df.loc[i, 'trendb'] * np.power(fit_x, 3) + df.loc[
            i, 'trendc'] * np.power(fit_x, 2) \
                        + df.loc[i, 'trendd'] * fit_x + df.loc[i, 'trende']
        oil_price_list.append(oil_price)
        oil_error = abs(oil_price - price_df.loc[i, 1])

        if i == df.shape[0] - 6:
            i = df.shape[0] - 7

        error = (0.5) * (np.square(ya_ols - trenda[i + 1]) + np.square(yb_ols - trendb[i + 1]) + np.square(
            yc_ols - trendc[i + 1])
                         + np.square(yd_ols - trendd[i + 1]) + np.square(ye_ols - trende[i + 1]))
        dmape = np.average(
            [np.abs((ya_ols - trenda[i + 1]) / trenda[i + 1]), np.abs((yb_ols - trendb[i + 1]) / trendb[i + 1])
                , np.abs((yc_ols - trendc[i + 1]) / trendc[i + 1]), np.abs((yd_ols - trendd[i + 1]) / trendd[i + 1]),
             np.abs((ye_ols - trende[i + 1]) / trende[i + 1])])
        mape += dmape
        avError += error
        sse += error * 2

        print('预测第', i + 1, '个点的油价', ',预测值为', oil_price, '真实值为', price_df.loc[i, 1], ',误差为', oil_error, '拟合值为',
              fitting_price)
        print('预测第', i + 1, '个点', ',平方损失为:', error, ',mape=', dmape, ',sse=', error * 2)
        # print('预测第',i+1,'个点,第一个参数：',ya_ols, ',真值:',trenda[i+1], '第二个参数:',yb_ols,',真值：',trendb[i+1],'\n')
        # print('预测第',i+1,'个点,第一个参数误差',trenda[i+1]-ya_ols,'，第二个参数误差:',trendb[i+1]-yb_ols,',平方损失:',np.square(trendb[i+1]-yb_ols)+np.square(trenda[i+1]-ya_ols),'\n')
    print('loss=', avError / 5, 'mape=', mape / 5, ',sse=', sse / 5)



main()
