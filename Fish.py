'''
Estimated Minimum TPM Rule for Equal-Covariance Normal Populations

2019/11/6
By:shenqiti

'''

import numpy as np
import math
INPUT_PATH = 'D:/python_code/NEW_step/Fisher-Discriminant/iris.txt'
fn = open(INPUT_PATH,'r')
dataset = []   #分三类，每50行一类

for line in fn:
        line = line.strip().split(' ')
        dataset.append(line)
dataset = dataset[0:len(dataset)-1]

# x0 = [5.1,3.5,1.4,0.2]
# x0 = [6.9,3.1,4.9,1.5]
x0 = [6.5,3.0,5.2,2.0]   #输入待判别特征

X1 = np.array(dataset[0:30]).astype(float)   #1
X2 = np.array(dataset[0+50:30+50]).astype(float)  #2
X3 = np.array(dataset[0+100:30+100]).astype(float)   #分割数据 每类前30个   #3

S1 = np.cov(X1.transpose())
S2 = np.cov(X2.transpose())
S3 = np.cov(X3.transpose())   #方差协方差矩阵

X1_mu = X1.mean(axis=0)
X2_mu = X2.mean(axis=0)
X3_mu = X3.mean(axis=0)  #按列求均值

# Given p1=p2=0.33  p3=0.33
p1 = 0.33
p2 = 0.33
p3 = 0.33
g = 3  #3类

S_pooled = (1/(len(X1)+len(X2)+len(X3)-g))*((len(X1)-1)*S1+(len(X2)-1)*S2+(len(X3)-1)*S3)
S_pooled_T = np.linalg.inv(S_pooled)

DD_1 = np.matmul(np.matmul(X1_mu,S_pooled_T),X1_mu)
DD_2 = np.matmul(np.matmul(X2_mu,S_pooled_T),X2_mu)
DD_3 = np.matmul(np.matmul(X3_mu,S_pooled_T),X3_mu)

DD_X1 =  np.matmul(np.matmul(X1_mu,S_pooled_T),x0)
DD_X2 =  np.matmul(np.matmul(X2_mu,S_pooled_T),x0)
DD_X3 =  np.matmul(np.matmul(X3_mu,S_pooled_T),x0)

d1 = math.log(p1,math.e) + DD_X1 - 0.5*DD_1
d2 = math.log(p2,math.e) + DD_X2 - 0.5*DD_2
d3 = math.log(p3,math.e) + DD_X3 - 0.5*DD_3   #找最大得分

d = [d1,d2,d3]

print("d1= ",d1,"d2= ",d2,"d3= ",d3)
print("The largest discriminant score is:",max(d))
print("该样本属于：第",d.index(max(d))+1,"类")


