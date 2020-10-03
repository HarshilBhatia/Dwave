import gurobipy as gp
from gurobipy import GRB

import numpy as np
import pandas as pd
#import pyomo 


# getting values of Returns and Sigma 

r = pd.read_excel('Returns.xlsx')
s = pd.read_excel('covariance1.xlsx')
#print(r)
returns = r['Return']
#print(returns)
sigma = s.loc[:,s.columns!='INDEX'];

sigma = sigma.to_numpy();

# Create a new model
m = gp.Model("qp")


N=25;
n=12;

# Create variables
x = m.addMVar(N , vtype=gp.GRB.BINARY, name="x")

#print(x[:])
#y = m.addMVar((3,4), vtype=GRB.BINARY) # add a 3x4 2-D array of binary variables
#print(y[:,1:3])


# Set objective: x^2 + x*y + y^2 + y*z + z^2 + 2 x
min_sigx = 0;

for i in range(N):
    for j in range(N):
        min_sigx += x[i]@x[j]*sigma[i][j];

m.setObjective(min_sigx)
xi =0 ;
for i in x:
    xi+=i;
m.addConstr(xi == n, "c0")

R =  450;
    # Initilize Expected Return to 0 for now . 

    # myu * x 
ret = 0; 
for i in range(N):
    ret += returns[i]*x[i];

m.addConstr(ret >= R, "c1")

m.optimize()

for v in m.getVars():
    print('%s %g' % (v.varName, v.x))

print('Obj: %g' % min_sigx.getValue())

