#!/usr/bin/env python
# -*- coding:utf-8 -*-
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg



def set_seed(seed=None):
    if seed is None:
        import time
        seed = int((time.time() * 10 ** 6) % 4294967295)
    try:
        np.random.seed(seed)
    except Exception as e:
        print("!!! WARNING !!!: Seed was not set correctly.")
        print("!!! Seed that we tried to use: " + str(seed))
        print("!!! Error message: " + str(e))
        seed = None
    return seed

def normalize(ori_data, flag='01'):
    data = ori_data.copy()
    if len(data.shape) > 1:  # 2-D
        N, K = data.shape
        minV = np.zeros(shape=K)
        maxV = np.zeros(shape=K)
        for i in range(N):
            minV[i] = np.min(data[i, :])
            maxV[i] = np.max(data[i, :])
            if np.abs(maxV[i] - minV[i]) > 0.00001:
                if flag == '01':  # normalize to [0, 1]
                    data[i, :] = (data[i, :] - minV[i]) / (maxV[i] - minV[i])
                else:
                    data[i, :] = 2 * (data[i, :] - minV[i]) / (maxV[i] - minV[i]) - 1
        return data, maxV, minV
    else:  # 1D
        minV = np.min(data)
        maxV = np.max(data)
        if np.abs(maxV - minV) > 0.00001:
            if flag == '01':  # normalize to [0, 1]
                data = (data - minV) / (maxV - minV)
            else:
                data = 2 * (data - minV) / (maxV - minV) - 1
        return data, maxV, minV


def re_normalize(ori_data, maxV, minV, flag='01'):
    data = ori_data.copy()
    if len(data.shape) > 1:  # 2-D
        Nc, K = data.shape
        for i in range(Nc):
            if np.abs(maxV[i] - minV[i]) > 0.00001:
                if flag == '01':  # normalize to [0, 1]
                    data[i, :] = data[i, :] * (maxV[i] - minV[i]) + minV[i]
                else:
                    data[i, :] = (data[i, :] + 1) * (maxV[i] - minV[i]) / 2 + minV[i]
    else:  # 1-D
        if np.abs(maxV - minV) > 0.00001:
            if flag == '01':  # normalize to [0, 1]
                data = data * (maxV - minV) + minV
            else:
                data = (data + 1) * (maxV - minV) / 2 + minV
    return data



seed = 42  # None

initLen = 6
trainLen = 2100
testLen = 1031



import pandas as pd


# dataset 1:MG
# import scipy.io as sio
# data = sio.loadmat('./dataset/MG_chaos.mat')['dataset'].flatten()
# # only use data from t=124 : t=1123  (all data previous are not in the same pattern!)
# data0 = data[123:1123]

# dataset 2 :sp500
# sp500_src = "./dataset/sp500.csv"
# dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
# sp500 = pd.read_csv(sp500_src, delimiter=',', parse_dates=[0], date_parser=dateparse).values
# data0 = np.array(sp500[:, 1], dtype=np.float)

# dataset 3:monthly-closings-of-the-dowjones.csv  #todo:  good
# dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m')  # %Y-%m-%d
# dowjones = pd.read_csv(r'./dataset/monthly-closings-of-the-dowjones.csv', delimiter=',', parse_dates=[0],date_parser=dateparse).values
# data0 = dowjones[:, 1].astype(np.float)

# dataset 4: monthly-critical-radio-frequenci  #todo:  good
# dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m')  # %Y-%m-%d
# sunspot = pd.read_csv(r'./dataset/monthly-critical-radio-frequenci.csv', delimiter=',', parse_dates=[0],date_parser=dateparse).values
# data0 = sunspot[:, 1].astype(np.float)

# dataset 5:  co2-ppm-mauna-loa-19651980.csv  #todo:  good
# dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m')  # %Y-%m-%d
# co2 = pd.read_csv(r'./dataset/co2-ppm-mauna-loa-19651980.csv', delimiter=',', parse_dates=[0],date_parser=dateparse).values
# data0 = co2[:, 1].astype(np.float)

# dataset 6: monthly-lake-erie-levels-1921-19.csv #todo:  good
# dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m')  # %Y-%m-%d
# lake = pd.read_csv(r'./dataset/monthly-lake-erie-levels-1921-19.csv', delimiter=',', parse_dates=[0],date_parser=dateparse).values
# data0 = lake[:, 1].astype(np.float)

# dataset 7: BrentOilPrice.csv #todo:  good
# BrentOilPrice = pd.read_csv(r'./dataset/BrentOilPrices.csv', delimiter=',').values
# data0 = BrentOilPrice[:, 1].astype(np.float)


# dataset 8: sunspot3132.csv #todo:  good
BrentOilPrice = pd.read_csv(r'./dataset/sunspot3132.csv', delimiter=';', header=None).values
data0 = BrentOilPrice[:, 3].astype(np.float)







data, maxV, minV = normalize(data0,'-01')

dataset_name = 't_last0'

# generate the ESNP reservoir
inSize = outSize = 1 # input/output dimension
resSize = 62
α = 0.1
spectral_radius = 0.98
sigma = 0.2
input_scaling = 1
reg = 1e-6  # regularization coefficient
β = 1


mode = 'prediction'

import math
min_rmse= math.inf
for mm in np.arange(0,1):
    RMSE_V = 0
    MAE_V = 0
    MSE_V = 0
    RMSE_std=[]
    MSE_std=[]
    MAE_std=[]
    for i in range(30):
        set_seed(seed)
        Win = (np.random.rand(resSize, 1 + inSize) - 0.5) * input_scaling
        W = (np.random.rand(resSize, resSize) - 0.5) * input_scaling
        Z = np.random.binomial(1, 1 - sigma, (resSize, resSize))  # Zero-one matrix with density 1-alpha (sparsity alpha)
        W = np.multiply(W, Z)  # Element-wise multiply to get desired sparsity
        rhoW = max(abs(linalg.eig(W)[0]))
        W *= spectral_radius / rhoW

        U = np.zeros((1 + inSize + resSize, trainLen - initLen))
        Yt = data[None, initLen+1:trainLen+1]

        u = np.zeros((resSize, 1))
        for t in range(trainLen):
            x = data[t]
            # ESNP update equation
            u = α * u + np.dot(W, np.tanh(np.dot(Win, np.vstack((1, x))) + np.dot(β, u)))
            if t >= initLen:
                U[:, t - initLen] = np.vstack((1, x, u))[:, 0]

        # train the output
        U_T = U.T
        if reg is not None:
            # use ridge regression
            Wout = np.dot(np.dot(Yt, U_T), linalg.inv(np.dot(U, U_T) + \
                                                      reg * np.eye(1 + inSize + resSize)))
        else:
            # use pseudo inverse
            Wout = np.dot(Yt, linalg.pinv(U))

        Y = np.zeros((outSize, testLen))
        x = data[trainLen]
        for t in range(testLen):
            u = α * u + np.dot(W, np.tanh(np.dot(Win, np.vstack((1, x))) + np.dot(β, u)))
            y = np.dot(Wout, np.vstack((1, x, u)))
            Y[:, t] = y
            if mode == 'generative':
                x = y
            elif mode == 'prediction':
                x = data[trainLen + t + 1]

        # compute MSE,RMSE,MAE for the first errorLen time steps
        errorLen = testLen
        np.savetxt('./predict/' + dataset_name + str(i) + '.csv', Y[0, 0:errorLen], delimiter=',')
        # Y = re_normalize(Y.reshape((Y.shape[1])), maxV, minV, '-01')
        # mse = sum(np.square(data0[trainLen+1:trainLen + errorLen+1] - Y[0:errorLen])) / errorLen
        mse = sum(np.square(data[0:trainLen + 1:trainLen + errorLen + 1] - Y[0, 0:errorLen])) / errorLen
        MSE_std.append(mse)
        from sklearn.metrics import mean_squared_error
        from math import sqrt
        rmse = sqrt(mean_squared_error(data[trainLen+1:trainLen+errorLen+1],Y[0,0:errorLen]))
        # rmse = sqrt(mean_squared_error(data0[trainLen+1:trainLen + errorLen+1], Y[0:errorLen]))
        RMSE_std.append(rmse)

        from sklearn.metrics import median_absolute_error
        mae = median_absolute_error(data[trainLen+1:trainLen+errorLen+1],Y[0,0:errorLen])
        # mae = median_absolute_error(data0[trainLen+1:trainLen + errorLen+1], Y[0:errorLen])
        MAE_std.append(mae)
        MAE_V += mae
        MSE_V += mse
        RMSE_V += rmse
    print('******************************************')
    print('resSize='+str(resSize))
    print('RMSE=' + str(RMSE_V / 30)+'±'+str(np.std(RMSE_std)))
    print('MAE=' + str(MAE_V / 30)+'±'+str(np.std(MAE_std)))
    print('MSE=' + str(MSE_V / 30)+'±'+str(np.std(MSE_std)))
    if (RMSE_V / 30)<min_rmse:
        min_rmse = RMSE_V / 30
        resSize_N = resSize
print('resSize_N='+str(resSize_N))

Y = re_normalize(Y.reshape((Y.shape[1])),maxV,minV,'-01')

fig4 = plt.figure()
ax41 = fig4.add_subplot(111)
time = range(testLen)
ax41.plot(time, data0[trainLen+1:trainLen + testLen+1], 'r-', label='the original data')
# ax41.plot(time, Y[0,0:errorLen], 'g--', label='the predicted data')
ax41.plot(time, Y[0:errorLen], 'g--', label='the predicted data')
ax41.set_ylabel("Magnitude")
ax41.set_xlabel('Time')
ax41.set_title('sunspot')
fig5 = plt.figure()
ax5 = fig5.add_subplot(111)
# ax5.plot(time, np.abs(data0[trainLen+1:trainLen+testLen+1]-Y[0,0:errorLen]), 'r')
ax5.plot(time, np.abs(data0[trainLen+1:trainLen + testLen+1] - Y[0:errorLen]), 'r')
ax5.set_ylabel("Magnitude")
ax5.set_xlabel('Time')
ax5.set_title('sunspot')

ax41.legend()
plt.tight_layout()
plt.show()

