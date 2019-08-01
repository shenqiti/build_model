
'''
将数据进行标准化，映射到[0,1]区间
以便后续进行主成分分析

'''

import pandas as pd
import math
import numpy as np


# INPUT_DATA = './data_finall/day_data/hour2day.csv'
INPUT_DATA= './data_finall/day_data/data_Scenarios_Analysis.csv'
OUPUT_PATH = './data_finall/day_data/data_Scenarios_Analysis_norm.csv'
data = pd.read_csv(INPUT_DATA)
DEWP = data['DEWP']
TEMP = data['TEMP']
PRES = data['PRES']
Iws = data['Iws']
Is = data['Is']
Ir = data['Ir']
# SO2 = data['SO2']
# NO2 = data['NO2']
# Co = data['Co']
# O3 = data['O3']
# maxtemp = data['maxtemp']
# mintemp = data['mintemp']
# windspeed = data['windspeed']
#
# pm25 = data['pm2.5']

DEWP = np.array(DEWP)
TEMP = np.array(TEMP)
PRES = np.array(PRES)
Iws = np.array(Iws)
Is = np.array(Is)
Ir = np.array(Ir)
# SO2 = np.array(SO2)
# NO2 = np.array(NO2)
# Co = np.array(Co)
# O3 = np.array(O3)
# maxtemp = np.array(maxtemp)
# mintemp = np.array(mintemp)
# windspeed = np.array(windspeed)
# pm25 = np.array(pm25)


DEWP_norm = []
TEMP_norm = []
PRES_norm = []
Iws_norm = []
Is_norm = []
Ir_norm = []
SO2_norm = []
No2_norm = []
Co_norm = []
O3_norm = []
maxtemp_norm = []
mintemp_norm = []
windspeed_norm = []

pm25_norm = []


DEWP_mean = np.mean(DEWP)
DEWP_var = np.var(DEWP)
TEMP_mean = np.mean(TEMP)
TEMP_var = np.var(TEMP)
PRES_mean = np.mean(PRES)
PRES_var = np.var(PRES)
Iws_mean = np.mean(Iws)
Iws_var = np.var(Iws)
Is_mean = np.mean(Is)
Is_var = np.var(Is)
Ir_mean = np.mean(Ir)
Ir_var = np.var(Ir)
# SO2_mean = np.mean(SO2)
# SO2_var = np.var(SO2)
# NO2_mean = np.mean(NO2)
# NO2_var = np.var(NO2)
# Co_mean = np.mean(Co)
# Co_var = np.var(Co)
# O3_mean = np.mean(O3)
# O3_var = np.var(O3)
# maxtemp_mean = np.mean(maxtemp)
# maxtemp_var = np.var(maxtemp)
# mintemp_mean = np.mean(mintemp)
# mintemp_var = np.var(mintemp)
# windspeed_mean = np.mean(windspeed)
# windspeed_var = np.var(windspeed)
# pm25_mean = np.mean(pm25)
# pm25_var = np.var(pm25)

for i in range(0,len(data)):
    DEWP_norm.append((DEWP[i]-DEWP_mean)/math.sqrt(DEWP_var))    #中心化处理
    TEMP_norm.append((TEMP[i]-TEMP_mean)/math.sqrt(TEMP_var))
    PRES_norm.append((PRES[i]-PRES_mean)/math.sqrt(PRES_var))
    Iws_norm.append((Iws[i]-Iws_mean)/math.sqrt(Iws_var))
    Is_norm.append((Is[i]-Is_mean)/math.sqrt(Is_var))
    Ir_norm.append((Ir[i]-Ir_mean)/math.sqrt(Ir_var))
    # SO2_norm.append((SO2[i]-SO2_mean)/math.sqrt(SO2_var))
    # No2_norm.append((NO2[i]-NO2_mean)/math.sqrt(NO2_var))
    # Co_norm.append((Co[i]-Co_mean)/math.sqrt(Co_var))
    # O3_norm.append((O3[i]-O3_mean)/math.sqrt(O3_var))
    # maxtemp_norm.append((maxtemp[i]-maxtemp_mean)/math.sqrt(maxtemp_var))
    # mintemp_norm.append((mintemp[i]-mintemp_mean)/math.sqrt(mintemp_var))
    # windspeed_norm.append((windspeed[i]-windspeed_mean)/math.sqrt(windspeed_var))
    # pm25_norm.append((pm25[i]-pm25_mean)/math.sqrt(pm25_var))


fn = open(OUPUT_PATH,'w')
for i in range(0,len(data)):
    fn.write(str(DEWP_norm[i]))
    fn.write(',')
    fn.write(str(TEMP_norm[i]))
    fn.write(',')
    fn.write(str(PRES_norm[i]))
    fn.write(',')
    fn.write(str(Iws_norm[i]))
    fn.write(',')
    fn.write(str(Is_norm[i]))
    fn.write(',')
    fn.write(str(Ir_norm[i]))
    # fn.write(',')
    # fn.write(str(SO2_norm[i]))
    # fn.write(',')
    # fn.write(str(No2_norm[i]))
    # fn.write(',')
    # fn.write(str(Co_norm[i]))
    # fn.write(',')
    # fn.write(str(O3_norm[i]))
    # fn.write(',')
    # fn.write(str(maxtemp_norm[i]))
    # fn.write(',')
    # fn.write(str(mintemp_norm[i]))
    # fn.write(',')
    # fn.write(str(windspeed_norm[i]))
    # # fn.write(',')
    # fn.write(str(pm25_norm[i]))
    fn.write('\n')
fn.close()



print('ok!!!!')
