import numpy as np
import torch.nn as nn

import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import  GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.metrics import mean_squared_error
import glob
device = 'cuda' if torch.cuda.is_available() else 'cpu'
MEAN = 0.9838866
STD = 0.010740371




def normalize(num):
    return (num - MEAN) / STD
def denormalize(num):
    return num * STD + MEAN

def visualize_scatterplot(predict, target, score,scale=100.,method="linear"):
    def _scatter(predict, target, subplot, mse,score,threshold=None):
        plt.subplot(subplot)
        plt.grid(linestyle="--")
        plt.xlabel("Validation Accuracy")
        plt.ylabel("Prediction")
        plt.scatter(target, predict, s=1)
        ax=plt.gca()
        ax.plot(ax.get_xlim(), ax.get_ylim(), ls='-',color='gray' )
        # plt.text(98.62,99.1,'mse='+str(mse))
        # plt.text(98.62, 99.05,'score='+str(score)) #0.1
        # plt.text(98.0,99.0,'mse='+str(mse))
        # plt.text(98.0, 98.7,'score='+str(score))
        # if threshold:
        #     ax = plt.gca()
        #     ax.set_xlim(threshold, 95)
        #     ax.set_ylim(threshold, 95)
    # mse=mean_squared_error(target,pred)
    # print('mse:',mse)
    predict = denormalize(predict) * scale
    target = denormalize(target) * scale
    mse=mean_squared_error(target/100,predict/100)
    plt.figure(figsize=(12, 6))
    _scatter(predict, target, 111,mse,score)

    # _scatter(predict, target, 122, threshold=90)
    plt.savefig("figs/scatterplot_"+method+".png", bbox_inches="tight")
    plt.close()




def convert_gpu_tensor(x):
    x=torch.from_numpy(x).float()
    x=x.to(device)
    return x


acc_list=[]
vector_list=[]
acc_arr=np.load("acc_" + str(0) + ".npy")
vector_arr=np.load("weight_bit_" + str(0) + ".npy")
num=len(glob.glob('acc_*.npy'))
for i in range(1,num):

    acc=np.load("acc_"+str(i)+".npy")
    acc_arr=np.hstack((acc_arr, acc))
    vector_arr=np.vstack((vector_arr, np.load("weight_bit_" + str(i) + ".npy")))

acc_arr=acc_arr.reshape(-1, 1)
mean_value=np.mean(acc_arr[:])
std_value=np.std(acc_arr[:])

X_train,X_test, y_train, y_test =\
    train_test_split(vector_arr, acc_arr, test_size=0.2, random_state=1)
y_train=normalize(y_train)
y_test=normalize(y_test)
y_test=y_test.ravel()
y_train=y_train.ravel()

#linear regression
reg = LinearRegression().fit(X_train, y_train)
score=reg.score(X_test, y_test)
print(score)
score=reg.score(X_test,y_test)
pred=reg.predict(X_test)
print(mean_squared_error(y_test,pred))
visualize_scatterplot(pred,y_test,score,method='linear')


#MLP
regr = MLPRegressor(random_state=1,max_iter=10000).fit(X_train, y_train)
pred=regr.predict(X_test)
score=regr.score(X_test,y_test)
print(mean_squared_error(y_test,pred))
print(score)
visualize_scatterplot(pred,y_test,score,method="MLP")


#Gaussian Process
kernel = DotProduct() + WhiteKernel()
gpr = GaussianProcessRegressor(kernel=kernel,
         random_state=0).fit(X_train, y_train)

pred=gpr.predict(X_test)
score=gpr.score(X_test, y_test)
print(mean_squared_error(y_test,pred))
print(score)
visualize_scatterplot(pred,y_test,score,method="Gaussian")



