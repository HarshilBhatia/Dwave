import numpy as np
import pandas as pd
# getting values of Returns and Sigma 

r = pd.read_excel('ret.xlsx')
s = pd.read_excel('corr.xlsx')

returns = r['return']
#print(returns)
sigma = s.loc[:,s.columns!='STOCK'];
sigma = sigma.to_numpy();
returns = returns

sigma = ((sigma*1000).astype(int));
returns = ((returns*100).astype(int));

N = 100;
n = 50;
R = 2500;

print("N = {} n= {} R = {}".format(N,n,R));

import localsolver


with localsolver.LocalSolver() as ls:
    # Declares the optimization model
    model = ls.model    

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
    
    print("x'Sx = ", ls.solution.get_value(min_sigx),"Returns = ",ls.solution.get_value(ret))
       