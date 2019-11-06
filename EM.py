'''
EM 算法对含缺失值的样本进行均值，协方差矩阵估计
By:shenqiti
2019/10/9
'''
import numpy as np

# X = [[-10000,0.0000001,3],[7,2,6],[5,1,2],[-10000,-10000,5]]  #-100000代表缺失值   原矩阵
# Y = [[-10000,0.0000001,3],[7,2,6],[5,1,2],[-10000,-10000,5]]  #  操作矩阵

X = [[3,6,0.0000001],[4,4,3],[-100000,8,3],[5,-100000,-100000]]  #-100000代表缺失值   原矩阵
Y = [[3,6,0.0000001],[4,4,3],[-100000,8,3],[5,-100000,-100000]]  #  操作矩阵

'''
Prediction step： 为方便查找，缺失值用-100000替代，0的位置用0.0000001替代 
'''
sample_num = len(X)   #4
feature_num = len(X[0])   #3
X_plus = []  # 存放缺失值
mu = []   #保存特征均值
sigma = []   #协方差矩阵
T2_temp = [] #存放X_2 这种新的“相关系数”矩阵
for i in range(0,feature_num):
    temp = 0
    cnt = 0
    for j in range(0,sample_num):
        if X[j][i] != -100000:
            temp += X[j][i]
            cnt += 1
    mu.append(temp/cnt)

for i in range(0,sample_num):
    for j in range(0,feature_num):
        if Y[i][j] == -100000:
            Y[i][j] = mu[j]
sigma = (np.cov(np.transpose(Y)))*(sample_num-1)/sample_num

for i in range(0,sample_num):
    cntt = i
    mu_miss = []
    mu_not_miss = []
    x_res = []
    sign_j = []
    sigma11 = []
    sigma12 = []
    sigma22 = []
    result = []
    temp = []

    for j in range(0,feature_num):
        if X[i][j] == -100000:
            mu_miss.append(mu[j])
            sign_j.append(j)
        else:
            mu_not_miss.append(mu[j])
            x_res.append(X[i][j])
    for m in range(0,len(sign_j)):
        for n in range(0,len(sign_j)):
            sigma11.append(sigma[m][n])

    if len(sigma11)>0:
        sigma11 = np.array(sigma11).reshape(len(sign_j),len(sign_j))
    for i in range(0,len(sigma11)):
        sigma12.append(sigma[i][len(sigma11[i]):len(sigma)])
    sigma12 = np.array(sigma12).reshape(len(sign_j),len(sigma)-len(sign_j))
    for i in range(len(sign_j)-1,len(sigma)-1):
        for j in range(len(sign_j)-1,len(sigma) - 1):
            sigma22.append(sigma[i+1][j+1])
    size = (len(sigma)-len(sign_j))
    sigma22 = np.array(sigma22).reshape(size,size)
    sigma21 = np.transpose(sigma12)
    if len(sigma11)>0:
        for i in range(0,len(x_res)):
            result.append(x_res[i]-mu_not_miss[i])
        result = np.transpose(result)
        X11 = mu_miss + np.array(np.array(sigma12).dot(np.linalg.inv(sigma22))).dot(result)
        for w in range(0,len(X11)):
            for e in range(0,len(X11)):
                temp.append(X11[w]*X11[e])
        temp = np.array(temp).reshape(len(X11),len(X11))
        X11_2 = sigma11-np.array(np.array(sigma12).dot(np.linalg.inv(sigma22))).dot(sigma21)+temp
        for each in X11:
            X_plus.append(each)
        # print(X11)
        # print(X11_2)
    else:continue

    if len(sign_j)>0:
        ttt = [0,1,2]
        for each in sign_j:
            if each in ttt:
                ttt.remove(each)
        begin = ttt[0]
        end = ttt[-1]
        cross = list(X11*X[cntt][begin:end+1])   #存放互相关

        for i in range(0,len(X11_2)):
            each = []
            each = list(X11_2[i])
            if len(each) == 1:
                for E in cross:
                    each.append(E)
            else:
                each.append(cross[i])
            T2_temp.append(each)

T2_temp_pro = [] #存放初始X的[x11x11 x11x12 x11x13...]
T2_plus = []   #存放修改后的[x11x11 x11x12 x11x13 x12x12 x12x13 x13x13 ......]

for i in range(0,sample_num):
    temp = []
    for k in range(0, feature_num):
        for j in range(k,feature_num):
            temp.append(X[i][0+k]*X[i][j])

    T2_temp_pro.append(temp)

T2_temp = np.array(T2_temp).reshape(-1,1)   #存放待替换的[x11x11 x11x12...]

cnt = 0

for i in range(0,feature_num+1):
    for j in range(0,len(T2_temp_pro[0])):
        if T2_temp_pro[i][j] >100000 or T2_temp_pro[i][j]<0:
            if T2_temp[cnt][0] in T2_plus:
                cnt += 1
                T2_plus.append(T2_temp[cnt][0])
            else:
                T2_plus.append(T2_temp[cnt][0])
                cnt += 1
        else:T2_plus.append(T2_temp_pro[i][j])

T2 = []   #下三角

for i in range(0,6):
    temp = 0
    for j in range(0,sample_num):
        temp += T2_plus[i+6*j]
    T2.append(temp)

T2_reg = []
# 下三角形对称填充

temp = []
temp.append(T2[0])
temp.append(T2[1])
temp.append(T2[2])
T2_reg.append(temp)

temp = []
temp.append(T2[1])
temp.append(T2[3])
temp.append(T2[4])
T2_reg.append(temp)

temp = []
temp.append(T2[2])
temp.append(T2[4])
temp.append(T2[5])
T2_reg.append(temp)

T1 = []
t = 0
for i in range(0,feature_num):
    temp = 0
    for j in range(0,sample_num):
        if X[j][i] == -100000:
            temp += X_plus[t]
            t += 1
        else:
            temp += X[j][i]
    T1.append(temp)
print("________________________________EM估计结果_____________________:")
print("T1="+ str(T1))
print("T2="+ str(T2_reg))


'''
Estimation step
'''
temp = []
TEMP = []
mu_est = [each/sample_num for each in T1]

for i in range(0,len(T2_reg)):
    temp = []
    for each in T2_reg[i]:
        temp.append(each/sample_num)
    TEMP.append(temp)

mu_est_T = np.array(mu_est).reshape(3,1)
mu_est_T2 = np.array(mu_est).reshape(1,3)
result =np.array(mu_est_T).dot(mu_est_T2)
sigma_est = TEMP-result

print("mu_est = " + str(mu_est))
print("sigma_est = "+ str(sigma_est))











print("_______________________部分中间变量____________________:")
print("X11 = "+ str(X11))
print("X_plus = "+ str(X_plus))
print("mu = "+ str(mu))
print("sigma = "+ str(sigma))
print("T2_temp = "+ str(T2_temp))
print("mu_miss = " + str(mu_miss))
print("mu_not_miss = " + str(mu_not_miss))
print("sign_j = " + str(sign_j))
print("sigma11 = " + str(sigma11))
print("sigma12 = " + str(sigma12))
print("sigma22 = " + str(sigma22))
