import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import fmin_bfgs

import skewGARCH_GJR

np.random.seed(1234)


'''Parameters sets'''
omegas = np.arange(0, 1, 0.1)
betas = np.arange(0, 1, 0.1)
alphas = np.arange(0, 1, 0.1)
gammas = np.arange(0, 1, 0.1)
etas = np.arange(2, 100, 1)
lamds = np.arange(-1, 0, 0.01)
parms_sets_n = list(itertools.product(omegas, betas, alphas))
parms_sets_n = np.array(parms_sets_n)
parms_sets_n = parms_sets_n[~((parms_sets_n[:, 1]+parms_sets_n[:, 2] >= 1) | (parms_sets_n[:, 0] <= 0))]
parms_sets_skt = list(itertools.product(omegas, betas, alphas, etas, lamds))
parms_sets_skt = np.array(parms_sets_skt)
parms_sets_skt = parms_sets_skt[~((parms_sets_skt[:, 3] <= 2) | (parms_sets_skt[:, 1]+parms_sets_skt[:, 2] >= 1) | (np.abs(parms_sets_skt[:, 4]) >= 1))]
parms_sets_gjr = list(itertools.product(omegas, betas, alphas, gammas))
parms_sets_gjr = np.array(parms_sets_gjr)
parms_sets_gjr = parms_sets_gjr[~((parms_sets_gjr[:, 1]+parms_sets_gjr[:, 2] == 1) | (parms_sets_gjr[:, 0] == 0))]


def cbf_n(params):
    global count
    count += 1
    _, logL = skewGARCH_GJR.garch_n(e, params)
    print(f"{count}:{logL}")
    if logL < 10**10:
        plt.scatter(count, logL, color='black')
    else:
        pass


def cbf_skt(params):
    global count
    count += 1
    _, logL = skewGARCH_GJR.garch_skt(e, params)
    print(f"{count}:{logL}")
    if logL < 10**10:
        plt.scatter(count, logL, color='black')
    else:
        pass


def cbf_gjr(params):
    global count
    count += 1
    _, logL = skewGARCH_GJR.gjr_n(e, params)
    print(f"{count}:{logL}")
    if logL < 10**10:
        plt.scatter(count, logL, color='black')
    else:
        pass


def obj_ngarch(params):
    _, logL = skewGARCH_GJR.garch_n(e, params)
    return logL


def obj_sktgarch(params):
    _, logL = skewGARCH_GJR.garch_skt(e, params)
    return logL


def obj_gjr(params):
    _, logL = skewGARCH_GJR.gjr_n(e, params)
    return logL


'''Frist step estimation'''

df = pd.read_csv("../data/midterm.csv", index_col=0, parse_dates=True)
df = df.loc["2010-03-04":]
dtindex = df.index[2:]
dfns = df.shape[0]
tindex = df.index[1:]
logdata = np.log(np.array(df))
dfii_logdata = logdata[1:dfns] - logdata[0:(dfns-1)]
ndata = dfii_logdata*100

ns = len(ndata)
yt = ndata[1:ns]
Lyt = ndata[0:ns-1]
c = np.ones(ns-1).reshape((ns-1, 1))
x = np.hstack((c, Lyt))
beta = np.linalg.inv(x.T @ x) @ (x.T @ yt)
e = yt - x @ beta

'''Second step estimation'''
'''Optimization'''
s = np.random.randint(0, len(parms_sets_n)-1, 1000)
ms = parms_sets_n[s]
count = 0
mle_n = np.zeros((5, parms_sets_n.shape[1]))
logL_n = np.arange(0, 5, 1)+(10**10)
for p in ms:
    tmp = fmin_bfgs(obj_ngarch, p, callback=cbf_n, disp=False, full_output=True, retall=True)
    print('\n')
    if (count <= 4) and (tmp[1] < 10**10):
        mle_n[count] = tmp[0]
        logL_n[count] = tmp[1]
    elif (np.any(tmp[0] < np.max(logL_n))):
        argmax_logL = np.where(logL_n == np.max(logL_n))
        mle_n[argmax_logL] = tmp[0]
        logL_n[argmax_logL] = tmp[1]
    else:
        pass
mle_min_n = mle_n[np.where(logL_n == np.min(logL_n))[0][0]]
vsig2_n, logL_n = skewGARCH_GJR.garch_n(e, mle_min_n)

s = np.random.randint(0, len(parms_sets_skt)-1, 1000)
ms = parms_sets_skt[s]
count = 0
mle_skt = np.zeros((5, parms_sets_skt.shape[1]))
logL_skt = np.arange(0, 5, 1)+(10**10)
for p in ms:
    tmp2 = fmin_bfgs(obj_sktgarch, p, callback=cbf_skt, disp=False, full_output=True, retall=True)
    print('\n')
    if (count <= 4) and (tmp2[1] < 10**10):
        mle_skt[count] = tmp2[0]
        logL_skt[count] = tmp2[1]
    elif (np.any(tmp2[0] < np.max(logL_skt))):
        argmax_logL = np.where(logL_skt == np.max(logL_skt))
        mle_skt[argmax_logL] = tmp2[0]
        logL_skt[argmax_logL] = tmp2[1]
    else:
        pass
mle_min_skt = mle_skt[np.where(logL_skt == np.min(logL_skt))[0][0]]
vsig2_skt, logL_skt = skewGARCH_GJR.garch_skt(e, mle_min_skt)

s = np.random.randint(0, len(parms_sets_gjr)-1, 1000)
ms = parms_sets_gjr[s]
count = 0
mle_gjr = np.zeros((5, parms_sets_gjr.shape[1]))
logL_gjr = np.arange(0, 5, 1)+(10**10)
for p in ms:
    tmp3 = fmin_bfgs(obj_gjr, p, callback=cbf_gjr, disp=False, full_output=True, retall=True)
    print('\n')
    if (count <= 4) and (tmp3[1] < 10**10):
        mle_gjr[count] = tmp3[0]
        logL_gjr[count] = tmp3[1]
    elif (np.any(tmp3[0] < np.max(logL_gjr))):
        argmax_logL = np.where(logL_gjr == np.max(logL_gjr))
        mle_gjr[argmax_logL] = tmp3[0]
        logL_gjr[argmax_logL] = tmp3[1]
    else:
        pass
mle_min_gjr = mle_gjr[np.where(logL_gjr == np.min(logL_gjr))[0][0]]
vsig2_gjr, logL_gjr = skewGARCH_GJR.gjr_n(e, mle_min_gjr)

print('\nnGARCH')
print(f"ML:{logL_n}")
print(f"MLE:{mle_min_n}")

print('\nsktGARCH')
print(f"ML:{logL_skt}")
print(f"MLE:{mle_min_skt}")

print('\nGJR')
print(f"ML:{logL_gjr}")
print(f"MLE:{mle_min_gjr}")

predict = pd.DataFrame(np.concatenate([e**2, vsig2_n, vsig2_skt, vsig2_gjr], 1), columns=['e2', 'vsig2_n', 'vsig2_skt', 'vsig2_gjr'], index=dtindex)
predict.to_csv('predict.csv')

'''Plot'''
fig, ax = plt.subplots(1, 1, figsize=(16, 9))
ax.plot(dtindex, e**2, label='data', alpha=0.3)
ax.plot(dtindex, vsig2_skt, label='sktGARCH', alpha=0.7)
ax.plot(dtindex, vsig2_n, label='nGARCH', alpha=0.7)
ax.plot(dtindex, vsig2_gjr, label='GJR', alpha=0.4)
ax.legend()
plt.savefig('../fig/all.png')
