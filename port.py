import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import blas, solvers
import cvxopt as opt
import time 

from cvxopt.solvers import qp

start_time = time.time()
r = pd.read_excel('Returns.xlsx')
s = pd.read_excel('covariance1.xlsx')
returns = r['Return']
#print(returns)
sigma = s.loc[:,s.columns!='INDEX'];
sigma = sigma*100*100;
returns = returns*100;

n= returns.size


print(returns.to_numpy())
print(sigma.to_numpy())



def optimal_portfolio(returns,S):
    n = 25
    returns = np.asmatrix(returns)
    # Convert to cvxopt matrices
    #S = opt.matrix(np.cov(returns.T))
    pbar = opt.matrix(np.mean(returns, axis=0), (1, n), 'd')

    pbar = pbar.T #returns matrix 
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1, (1, n))
    b = opt.matrix(1)
    
    N=500

    mus = [ 10**(5.0*t/N-1.0) for t in range(N) ]
    la = opt.matrix(S.to_numpy());

    portfolios = [ qp(mu*la, -pbar, G, h, A, b)['x'] for mu in mus ]

    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, la*x)) for x in portfolios]
   
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE 
    m1 = np.polyfit(returns, risks, 2)
    print(m1)
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = qp(opt.matrix(x1 * la), -pbar, G, h, A, b)['x']

    return np.asarray(wt), returns, risks
    

#weights , returns , risks = optimal_portfolio(returns,sigma)
end_time = time.time()

#print("weights",weights)
#print("returns",returns)
#print("risks",risks)
#print(end_time-start_time);

#print( solution['x'])