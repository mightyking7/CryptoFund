#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 18:44:40 2020

@author: trentc
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

def get_data():
    df = pd.read_csv("trentOutput/bitcoin.csv")
    N = df.shape[1]
    true_price = np.zeros((N+1,6))
    mu = np.zeros((N,6))
    sigma = np.zeros((N,6))
    n = 0
    for name in ["bitcoin","dash","ethereum","litecoin","monero","ripple"]:
        df = pd.read_csv("input_12mo/"+name+".csv")
        true_price[:,n] = df["Close"][-N-1:]
        
        df = pd.read_csv("trentOutput/"+name+".csv")
        mu[:,n] = np.mean(df, axis=0)
        sigma[:,n] = np.std(df, axis=0)
        n+=1
    
    true_chg = true_price[1:] / true_price[:-1,:] - 1
    est_chg = mu / true_price[:-1,:] - 1
    est_sig = sigma / true_price[:-1,:]
    return true_price, true_chg, est_chg, est_sig
########################################################

def calc_dist(u, s):
    
    maxNdx = np.argmax(u)
    mu_delta = u - u[maxNdx]
    sigma_delta = np.sqrt(s**2 + s[maxNdx]**2)
    
    P_best = stats.norm(0,1).cdf(mu_delta/sigma_delta) # Pr{largest return}
    if len(u)>1: P_best[maxNdx] = 1-np.sort(P_best)[-2]
    
    weighted_P_largest = P_best * u
    W = weighted_P_largest / sum(weighted_P_largest)
    return W
######################################################

true_price, true_chg, est_chg, est_sig = get_data()

P_lose = stats.norm(est_chg, est_sig).cdf(0)

risk_thold = 0.25 # if P_loose > this, sell it
gain_thold = 0.005 # if est gain is < this, sell it
acct_balance = true_price[0,0] + np.zeros(len(true_price))
day_chg = np.zeros(len(est_chg))

for day in range(len(est_chg)):
    W = np.zeros((6,1))
    idx = (P_lose[day,:] < risk_thold) & (est_chg[day,:] > gain_thold)
    if any(idx):
        W[idx,0] = calc_dist(est_chg[day,idx], est_sig[day,idx])
        day_chg[day] = np.matmul(W.T, true_chg[day,:]) + 1
        acct_balance[day+1] = acct_balance[day] * day_chg[day]
    else:
        # sell all investments and hold for 0% change
        acct_balance[day+1] = acct_balance[day]
    

plt.plot(true_price[:,0],'.-',label="BTC");
plt.plot(acct_balance,'.-',label="JKIT"); plt.grid()
plt.legend()

"""
x = np.linspace(0,0.1,500)
for k in range(6):
    plt.plot(x*100 , stats.norm(est_chg[day,k],est_sig[day,k]).pdf(x))
"""
#########################################################
"""
x[k-1] = [x1 , x2 ; x3 ; x4]
x[k] = [x2 , x3 , x4 , x4-x3+x4]
F = 0 1  0 0
	0 0  1 0
	0 0  0 1
	0 0 -1 2
z = y_pred

Y_pred = np.array([[-0.75365889, -0.65518671, -0.67831546, -0.69850153],
       [-0.7120828 , -0.63446337, -0.65418303, -0.64578646],
       [-0.62045223, -0.57979047, -0.59247988, -0.58455998],
       [-0.50804281, -0.49607569, -0.49490774, -0.43317083]])

F = np.array([[0,1,0,0],[0,0,1,0],[0,0,0,1],[0,0,-1,2]])
Q = np.eye(4) * 0.05**2 # prediction
R = np.diag([0.01 , 0.02 , 0.03 , 0.04])**2
P = np.eye(4)
I = np.eye(4)

xold = y_pred[0,:,0,0]

#xhat = np.matmul(F, xold)
P = np.matmul(np.matmul(F,P), F.T) + Q

#z = y_pred[1,:,0,0]
#yhat = z - xhat
S = P + R
K = np.linalg.solve(S, P)
#x = xhat + np.matmul(K, yhat)
Pnew = (I-K) + P
    
#########################################
d = np.zeros((Ntest+3,Ntest)); s = d.copy()
yNdx = np.arange(4)
for k in range(0,Ntest):
    d[k+yNdx,k] = np.mean(y_pred[k,:,:,0],axis=1)
    s[k+yNdx,k] = np.std(y_pred[k,:,:,0],axis=1)

g = np.zeros((len(d), 4-1))
gs = np.zeros((len(d), 4-1))
for k in range(len(d)):
    start = max(0,k-3)
    col = np.arange(start,min(len(d),min(k+1,Ntest)))
    if len(col)>1:
        g[k,0:len(col)-1] = np.diff(d[k,col])
        for kn in range(len(col)-1):
            gs[k,kn] = np.sqrt(s[k,col[kn]]**2 + s[k,col[kn+1]]**2)
    else:
        g[k,0] = 0; gs[k,0] = 0
plt.subplot(2,1,2)
plt.plot(g,'.-')

w = np.array([0.2 , 0.3 , 0.5])
plt.plot(np.matmul(g,w),'*-'); plt.grid()
"""
    