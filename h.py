import neal
import numpy
import pandas as pd
from pyqubo import Array, Placeholder, solve_qubo, Constraint
from pyqubo import Sum, Model, Mul
import math

# getting values of Returns and Sigma 
r = pd.read_excel('returnsall.xlsx')
s = pd.read_excel('covariances.xlsx')

returns = r['Return']
#print(returns)
sigma = s.loc[:,s.columns!='INDEX'];
sigma = sigma.to_numpy();
returns = abs(returns)
sigma = (sigma*1).astype(int);
returns =(returns*1).astype(int);


N = 25;
n = 5;
R = 00;

sum=0;
for i in range(N):
    sum += returns[i];

K = math.log2(abs(sum));
K = math.floor(K);
K+=1;
x = Array.create('arr', N+K, 'BINARY')

# Constraints in our model
#x' Sigmax
min_sigx = 0;

H =0;#temp hamiltonian for the Constraint

for i in range(N):
    for j in range(N):
        H += x[i]*x[j]*sigma[i][j];

min_sigx += Constraint(H,label = "min_portfolio");

H =0 ;#temp hamiltonian for the Constraint
for i in range(N):
    H+=x[i];
H-=n;
H = H**2;

select_n = Constraint(H , label = "select_n_projects")

H = 0; #temp hamiltonian for the Constraint
for i in range(N):
    H += returns[i]*x[i];
H -= R
for i in range(0,K):
    H-= (2**i)*(x[i+N]);
expR = Constraint(H**2, label = "min_expected_return");
mx =0 ;
for i in range(N):
    s=0;
    for j in range(N):
            s+=2*sigma[i][j];
            
    mx = max(mx,s-sigma[i][i])


lambda1 = mx;
lambda2 = mx;


H = min_sigx  + lambda1*select_n + expR;


model = H.compile();
qubo, offset = model.to_qubo()

from hybrid.reference.kerberos import KerberosSampler
import dimod
import hybrid


bqm = dimod.BinaryQuadraticModel.from_qubo(qubo);

iteration = hybrid.RacingBranches(
hybrid.InterruptableTabuSampler(),
hybrid.EnergyImpactDecomposer(size=15, rolling_history=0.20) |
hybrid.QPUSubproblemAutoEmbeddingSampler() | hybrid.SplatComposer()
) | hybrid.ArgMin()

workflow = hybrid.LoopUntilNoImprovement(iteration, convergence=3)

init_state = hybrid.State.from_problem(bqm)
final_state = workflow.run(init_state).result()



x=final_state.samples.first
mn=1e9;

# solution = KerberosSampler().sample(bqm, max_iter=10, convergence=3)

for sample in final_state.samples:   
    final_sigx=0; 
    n_temp=0;
    for i in range(N):
        n_temp += sample['arr[{}]'.format(i)]
    for i in range(N):
        for j in range(N):
            final_sigx += sample['arr[{}]'.format(i)]*sample['arr[{}]'.format(j)]*sigma[i][j];
    ret = 0;
    for i in range(N):
        ret += sample['arr[{}]'.format(i)] *returns[i];
    #print(final_sigx,n_temp)

    if(final_sigx<mn and ret>=R and n_temp==n):
        mn=final_sigx;
        x = sample;
        final_ret = ret;
for i in range(N):
    if(x['arr[{}]'.format(i)]==1):
        print(i)

#slack variables
sl=0;
for i in range(N,N+K):
    if(x['arr[{}]'.format(i)] == 1):
        sl += (2**(i-N))


print("slack = ",sl)

print("risk =" ,mn)
print("return =",final_ret)
