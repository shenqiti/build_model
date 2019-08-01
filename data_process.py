'''
数据处理：将一天24小时的各项数据求平均算在一天内；
By:wsg
2019/5/18

'''

import pandas as pd

INPUT_PATH = './data_pro/data_pro.csv'
OUT_PUT = './data_pro/6666.csv'

data = pd.read_csv(INPUT_PATH)
date = data["date"]
hour = data["hour"]
pm25 = data["pm2.5"]
DEWP = data["DEWP"]
TEMP = data["TEMP"]
PRES = data["PRES"]
Iws = data["Iws"]
Is = data["Is"]
Ir = data["Ir"]
pm25_ave = []
DEWP_ave = []
TEMP_ave = []
PRES_ave = []
Iws_ave = []
Is_ave = []
Ir_ave = []
day = []
date_copy = []

for each in date:
    if each not in date_copy:
        date_copy.append(each)


j = 0
cnt = 0
for i in range(0,len(date)):
    if date[i] == date_copy[j]:
        cnt = cnt + 1
    else:
        day.append(cnt)
        j = j + 1
        cnt = 1
day.append(24)

sum_pm25 = 0
sum_DEWP = 0
sum_TEMP = 0
sum_PRES = 0
sum_Iws = 0
sum_Is = 0
sum_Ir = 0

j = 0
for i in range(0,len(date)):

    if date[i] == date_copy[j]:
        sum_pm25 = sum_pm25 + pm25[i]
        sum_DEWP = sum_DEWP + DEWP[i]
        sum_TEMP = sum_TEMP + TEMP[i]
        sum_PRES = sum_PRES + PRES[i]
        sum_Iws = sum_Iws + Iws[i]
        sum_Is = sum_Is + Is[i]
        sum_Ir = sum_Ir + Ir[i]
    else:

        pm25_ave.append(sum_pm25 / day[j])
        DEWP_ave.append(sum_DEWP / day[j])
        TEMP_ave.append(sum_TEMP / day[j])
        PRES_ave.append(sum_PRES / day[j])
        Iws_ave.append(sum_Iws / day[j])
        Is_ave.append(sum_Is / day[j])
        Ir_ave.append(sum_Ir / day[j])
        j = j + 1
        sum_pm25 = pm25[i]
        sum_DEWP = DEWP[i]
        sum_TEMP = TEMP[i]
        sum_PRES = PRES[i]
        sum_Iws = Iws[i]
        sum_Is = Is[i]
        sum_Ir = Ir[i]

fn = open(OUT_PUT,'w')
for i in range(0,len(pm25_ave)):
    fn.write(str(date_copy[i]))
    fn.write(',')
    fn.write(str(pm25_ave[i]))
    fn.write(',')
    fn.write(str(DEWP_ave[i]))
    fn.write(',')
    fn.write(str(TEMP_ave[i]))
    fn.write(',')
    fn.write(str(PRES_ave[i]))
    fn.write(',')
    fn.write(str(Iws_ave[i]))
    fn.write(',')
    fn.write(str(Is_ave[i]))
    fn.write(',')
    fn.write(str(Ir_ave[i]))
    fn.write('\n')

fn.close()

print("OK")
