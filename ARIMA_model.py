
'''
By:shenqiti
2019/7/9
'''
import pandas as pd
import pandas_datareader
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import style
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import statsmodels.api as sm

style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

stockFile = 'D:/python_code/NEW_step/try_paper/data/oil_price.csv'
stock = pd.read_csv(stockFile,index_col=0,parse_dates=[0])

stock_week = stock['Price'].resample('W-MON').mean()
stock_train = stock_week['2016':'2018']
stock_train.plot(figsize=(12,8))
plt.legend(bbox_to_anchor=(1.25,0.5))
plt.title('Stock Price')
sns.despine()
# plt.show()
#差分操作
stock_diff = stock_train.diff()
stock_diff = stock_diff.dropna()  #删除NAN值

plt.figure()
plt.plot(stock_diff)
plt.title('一阶差分')
# plt.show()
check = sm.tsa.stattools.adfuller(stock_diff)    #平稳性检验
print (check)

'''
result : (-8.940749717155187, 9.267799032663645e-15, 0, 149, {'1%': -3.4750180242954167, '5%': -2.8811408028842043, '10%': -2.577221358046935}, 561.3312278939167)
1%、%5、%10不同程度拒绝原假设的统计值和ADF Test result的比较，ADF Test result同时小于1%、5%、10%即说明非常好地拒绝该假设;
P-value是否非常接近0

故通过平稳性检验；
'''

acf = plot_acf(stock_diff,lags=20)
plt.title('ACF')
# acf.show()

pacf = plot_pacf(stock_diff,lags=20)
plt.title('PACF')
# pacf.show()

model = ARIMA(stock_train,order=(1,1,1),freq='W-MON')
result = model.fit()
#print(result.summary)

pred = result.predict('20160829','20181203',dynamic=True,typ='levels')
plt.figure(figsize=(6,6))
plt.xticks(rotation=45)
plt.plot(pred)
plt.plot(stock_train)
plt.show()
