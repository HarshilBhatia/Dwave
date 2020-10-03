import numpy as np
import pandas as pd
#import pyomo 


# getting values of Returns and Sigma 

r = pd.read_excel('Returns.xlsx')
s = pd.read_excel('covariance1.xlsx')

returns = r['Return']
#print(returns)
sigma = s.loc[:,s.columns!='INDEX'];
#sigma = sigma*100*100;
#eturns = returns*100;
sigma = sigma.to_numpy();

"""
for i in range(25):
    for j in range(25):
        sigma[i][j]=100000000;

sigma[0][0]=1;
sigma[1][1]=1;
"""


import localsolver

with localsolver.LocalSolver() as ls:
    # Declares the optimization model
    model = ls.model    

    N = 25;
    n =12;
        
    x = [model.bool() for i in range(N)]

    # weight constraint
    
    min_sigx = 0;

    for i in range(N):
        for j in range(N):
            min_sigx += x[i]*x[j]*sigma[i][j];
        


    #summation xi
    xi =0 ;
    for i in x:
        xi+=i;
    model.constraint(xi == n)

    R =  450;
    # Initilize Expected Return to 0 for now . 

    # myu * x 
    ret = 0; 
    for i in range(N):
        ret += returns[i]*x[i];

    model.constraint(ret>=R)
    # maximize value

    
    model.minimize(min_sigx);
    model.close()
    #
    # Parameterizes the solver
    #
    ls.param.time_limit = 10

    ls.solve()
   # ls.solution.get_value(x)
    #print(min_sigx)

    print(ls.solution.get_value(min_sigx))
    print(ls.solution.get_value(ret))
    
    #print(ls.solution.get_value(xi))
    print()
    s = 0;
    
    for i in range(N):
        if(x[i].value==1):
            print(i)
    