'''
By:shenqiti
2019/9/4

常用的回归模型
ELM和BELM见相应的文件


1.SVR
2.BPNN
3.LSTM
4.ELM/DELM
5.KRR
'''
#SVR
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
import time
import matplotlib.pyplot as plt
import math

# oil_price_path = 'D:/python_code/NEW_step/try_paper/mycode/new/media_oil_price.xlsx'
# data=np.array(pd.read_excel(oil_price_path,sheet_name='Sheet1',header=None))
# cnt = 20


# for n in range(9,-1,-1):
#     train_x = []
#     train_y = []
#     test_x = []
#     test_y = []
#     for i in range(len(data)-32-500*n-cnt-500,len(data)-32-500*n-cnt):
#     # for i in range(0,len(data)-32-500*n-cnt):
#         train_x.append(data[i][0])
#         train_y.append(data[i][1])
#
#     train_x = np.array(train_x).reshape(-1,1)
#     train_y = np.array(train_y).reshape(-1,1)
#
#     for i in range(len(data)-cnt-500*n-32,len(data)-500*n-32):
#         test_x.append(data[i][0])
#         test_y.append(data[i][1])
#     train_x = np.array(train_x).reshape(-1,1)
#     train_y = np.array(train_y).reshape(-1,1)
#     test_x = np.array(test_x).reshape(-1,1)
#     test_y = np.array(test_y)
#
# # # #############################################################################
#
#
#     # 训练SVR模型
#
#     # 初始化SVR
#     svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
#                        param_grid={"C": [1e0, 1e1, 1e2, 1e3],
#                                    "gamma": np.logspace(-2, 2, 5)})
#     # 记录训练时间
#     t0 = time.time()
#     # 训练
#     svr.fit(train_x,train_y.ravel())
#     svr_fit = time.time() - t0
#
#     t0 = time.time()
#     # 测试
#     y_svr = svr.predict(test_x)
#     svr_predict = time.time() - t0
#     error1 = abs(test_y-y_svr)
#     error2 = [each*each for each in error1]
#     MSE = sum(error2)/cnt
#     MAE = sum(error1)/cnt
#     MAPE = sum(error1/test_y)/cnt
#     SSE = sum(error2)
#     RMSE = math.sqrt(MSE)
#     print("训练数据点个数为:",len(train_x))
#     print("SVR的" + "MSE=", MSE, "MAE=", MAE, 'MAPE=', MAPE, 'SSE=', SSE, 'RMSE=', RMSE)
# #############################################################################


#BPNN
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn import datasets,preprocessing
# from sklearn.model_selection import train_test_split
# from keras.models import Sequential
# from keras.layers import Dense
# from sklearn.metrics import mean_squared_error
#
# import warnings
# warnings.filterwarnings("ignore")
#
#
# oil_price_path = 'D:/python_code/NEW_step/try_paper/mycode/new/media_oil_price.xlsx'
# data=np.array(pd.read_excel(oil_price_path,sheet_name='Sheet1',header=None))
# cnt = 20
# #读取数据
#
# #Y [24.  21.6 34.7 33.]  X [[[4.7410e-02 0.0000e+00 1.1930e+01 ... 2.1000e+01 3.9690e+02 7.8800e+00]],[]]
# X = np.array(data[:,0]).reshape(-1,1)
# Y = data[:,1]
#
#
#
# train_x = []
# train_y = []
# test_x = []
# test_y = []
# # for n in range(10,11):
# for n in range(9,-1,-1):
#     train_x = []
#     train_y = []
#     test_x = []
#     test_y = []
#     for i in range(len(data)-32-500*n-cnt-500,len(data)-32-500*n-cnt):
#     # for i in range(0,len(data)-32-500*n-cnt):
#         train_x.append(data[i][0])
#         train_y.append(data[i][1])
#
#     train_x = np.array(train_x).reshape(-1,1)
#     train_y = np.array(train_y).reshape(-1,1)
#
#     for i in range(len(data)-cnt-500*n-32,len(data)-500*n-32):
#         test_x.append(data[i][0])
#         test_y.append(data[i][1])
#     train_x = np.array(train_x).reshape(-1,1)
#     train_y = np.array(train_y).reshape(-1,1)
#     test_x = np.array(test_x).reshape(-1,1)
#     test_y = np.array(test_y).reshape(-1,1)
#
#     dim = 1
#
#     print('n=',n)
#     model = Sequential()
#     model.add(Dense(28,activation='relu',input_dim=dim))
#     model.add(Dense(28,activation='relu'))
#     model.add(Dense(1,activation='linear'))
#     model.compile(optimizer='adam',loss='mse')
#     model.fit(train_x,train_y,epochs=500,batch_size=20,verbose=0)
#     predict =model.predict(test_x,batch_size=20)
#
#     error1 = abs(test_y-predict)
#     error2 = [each*each for each in error1]
#     MSE = sum(error2)/cnt
#     MAE = sum(error1)/cnt
#     MAPE = sum(error1/test_y)/cnt
#     SSE = sum(error2)
#     RMSE = math.sqrt(MSE)
#     print("训练数据点个数为:",len(train_x))
#     print("BPNN的" + "MSE=", MSE, "MAE=", MAE, 'MAPE=', MAPE, 'SSE=', SSE, 'RMSE=', RMSE)



# # #LSTM
#
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn import preprocessing
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
# from sklearn.metrics import mean_squared_error
#
# import warnings
# warnings.filterwarnings('ignore')
#
# #读取数据
# oil_price_path = 'D:/python_code/NEW_step/try_paper/mycode/new/media_oil_price.xlsx'
# data=np.array(pd.read_excel(oil_price_path,sheet_name='Sheet1',header=None))
# cnt = 20
# train_x = []
# train_y = []
# test_x = []
# test_y = []
# for n in range(10,11):
#     train_x = []
#     train_y = []
#     test_x = []
#     test_y = []
#     # for i in range(len(data)-32-500*n-cnt-500,len(data)-32-500*n-cnt):
#     for i in range(0,len(data)-32-500*n-cnt):
#          train_x.append(data[i][0])
#          train_y.append(data[i][1])
#
#     train_x = np.array(train_x).reshape(-1,1)
#     train_y = np.array(train_y).reshape(-1,1)
#
#     for i in range(len(data)-cnt-500*n-32,len(data)-500*n-32):
#         test_x.append(data[i][0])
#         test_y.append(data[i][1])
#     train_x = np.array(train_x).reshape(456,1,1)
#     train_y = np.array(train_y).reshape(-1,1)
#     test_x = np.array(test_x).reshape(20,1,1)
#     test_y = np.array(test_y).reshape(-1,1)
#
#
#     dim = 1
#     #建立网络(adam)
#     hiddennum = 12
#     batch_size = 20
#     model = Sequential()
#     model.add(LSTM(hiddennum,return_sequences=True,input_dim = dim))
#     model.add(LSTM(hiddennum))
#     model.add(Dense(1,activation='linear'))
#     model.compile(optimizer='adam',loss='mse')
#
#     model.fit(train_x,train_y,batch_size=batch_size,epochs=500,verbose=0)
#     predict =model.predict(test_x,batch_size=20)
#
#     error1 = abs(test_y-predict)
#     error2 = [each*each for each in error1]
#     MSE = sum(error2)/cnt
#     MAE = sum(error1)/cnt
#     MAPE = sum(error1/test_y)/cnt
#     SSE = sum(error2)
#     RMSE = math.sqrt(MSE)
#     print("训练数据点个数为:",len(train_x))
#     print("LSTM的" + "MSE=", MSE, "MAE=", MAE, 'MAPE=', MAPE, 'SSE=', SSE, 'RMSE=', RMSE)


#ELM 和DELM


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ELM_myself import HiddenLayer  #此行和下一行不能同时出现 用哪个取消注释哪个
# from DELM_myself import HiddenLayer
import math
import warnings
warnings.filterwarnings('ignore')

#读取数据
oil_price_path = 'D:/python_code/NEW_step/try_paper/mycode/new/media_oil_price.xlsx'
data=np.array(pd.read_excel(oil_price_path,sheet_name='Sheet1',header=None))
cnt = 20
train_x = []
train_y = []
test_x = []
test_y = []

for n in range(9,-1,-1):
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for i in range(len(data)-32-500*n-cnt-500,len(data)-32-500*n-cnt):
    # for i in range(0,len(data)-32-500*n-cnt):
         train_x.append(data[i][0])
         train_y.append(data[i][1])

    train_x = np.array(train_x).reshape(-1,1)
    train_y = np.array(train_y).reshape(-1,1)

    for i in range(len(data)-cnt-500*n-32,len(data)-500*n-32):
        test_x.append(data[i][0])
        test_y.append(data[i][1])
    train_x = np.array(train_x).reshape(-1,1)
    train_y = np.array(train_y).reshape(-1,1)
    test_x = np.array(test_x).reshape(-1,1)
    test_y = np.array(test_y).reshape(-1,1)
    #TODO
    # print(1.0 / (1 + np.exp(-test_x))) #算出来都是1  这个问题要解决一下  换成ReLu结果就好了


    my_EML = HiddenLayer(train_x, 5)
    my_EML.regressor_train(train_y)

    predict = my_EML.regressor_test(test_x)



    # my_DEML = HiddenLayer(train_x,10)
    # my_DEML.regressor_train(train_y)
    # predict = my_DEML.regressor_test(test_x)
    # print(predict)
    # print(test_y)

    error1 = abs(test_y-predict)
    error2 = [each*each for each in error1]
    MSE = sum(error2)/cnt
    MAE = sum(error1)/cnt
    MAPE = sum(error1/test_y)/cnt
    SSE = sum(error2)
    RMSE = math.sqrt(MSE)
    print("训练数据点个数为:",len(train_x))
    print("DELM的" + "MSE=", MSE, "MAE=", MAE, 'MAPE=', MAPE, 'SSE=', SSE, 'RMSE=', RMSE)









#KRR
#
# import time
# import numpy as np
# from sklearn.kernel_ridge import KernelRidge
#
#
# oil_price_path = 'D:/python_code/NEW_step/try_paper/mycode/new/media_oil_price.xlsx'
# data=np.array(pd.read_excel(oil_price_path,sheet_name='Sheet1',header=None))
# cnt = 20
#
#
# for n in range(9,-1,-1):
#     train_x = []
#     train_y = []
#     test_x = []
#     test_y = []
#     for i in range(len(data)-32-500*n-cnt-500,len(data)-32-500*n-cnt):
#     # for i in range(0,len(data)-32-500*n-cnt):
#         train_x.append(data[i][0])
#         train_y.append(data[i][1])
#
#     train_x = np.array(train_x).reshape(-1,1)
#     train_y = np.array(train_y).reshape(-1,1)
#
#     for i in range(len(data)-cnt-500*n-32,len(data)-500*n-32):
#         test_x.append(data[i][0])
#         test_y.append(data[i][1])
#     train_x = np.array(train_x).reshape(-1,1)
#     train_y = np.array(train_y).reshape(-1,1)
#     test_x = np.array(test_x).reshape(-1,1)
#     test_y = np.array(test_y)
#
# # # #############################################################################
#
#
#     # 训练KRR模型
#
#     # 初始化KRR
#     kr = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=5,
#                       param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3], "gamma": np.logspace(-2, 2, 5)})
#
#     # 记录训练时间
#     t0 = time.time()
#     # 训练
#     kr.fit(train_x,train_y.ravel())
#
#     t0 = time.time()
#     # 测试
#     y_kr = kr.predict(test_x)
#     error1 = abs(test_y-y_kr)
#     error2 = [each*each for each in error1]
#     MSE = sum(error2)/cnt
#     MAE = sum(error1)/cnt
#     MAPE = sum(error1/test_y)/cnt
#     SSE = sum(error2)
#     RMSE = math.sqrt(MSE)
#     print("训练数据点个数为:",len(train_x))
#     print("KRR的" + "MSE=", MSE, "MAE=", MAE, 'MAPE=', MAPE, 'SSE=', SSE, 'RMSE=', RMSE)
