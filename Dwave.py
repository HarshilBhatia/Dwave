import neal
import numpy
import pandas as pd
from pyqubo import Array, Placeholder, solve_qubo, Constraint
from pyqubo import Sum, Model, Mul
import math

# getting values of Returns and Sigma 
r = pd.read_excel('Returns.xlsx')
s = pd.read_excel('covariance1.xlsx')

returns = r['Return']
#print(returns)
sigma = s.loc[:,s.columns!='INDEX'];
sigma = sigma.to_numpy();



N = 25;
n = 1;
mx = 0;
for i in returns:
    mx=max(i,mx)

K = math.log2(mx-1);
K = math.floor(K);
K+=1;
#print(returns)
x = Array.create('arr', N+K, 'BINARY')
#print(x)

# Constraints in our model
print(K)
#x' Sigmax
min_sigx = 0;

H =0;#temp hamiltonian for the Constraint
for i in range(N):
    H+=x[i]*sigma[i][i]

for i in range(N):
    for j in range(N):
        H += x[i]*x[j]*sigma[i][j];

min_sigx += Constraint(H,label = "min_portfolio");

H =0 ;#temp hamiltonian for the Constraint
for i in range(N):
    H+=x[i];
H-=n;
H = H**2;

#print(x)
select_n = Constraint(H , label = "select_n_projects")
#print(select_n)
ExpectedReturn = 0 ; # Initilize Expected Return to 0 for now . 

H = 0; #temp hamiltonian for the Constraint
for i in range(N):
    H += returns[i]*x[i];

H -= ExpectedReturn

for i in range(0,K-N):
    H-= (2**i)*(x[i+N]);

H = H**2;

expR = Constraint(H, label = "min_expected_return");

lambda1 = 100000;
lambda2 = 1;

H = min_sigx + lambda1*select_n + lambda2*expR;

model = H.compile();
qubo, offset = model.to_qubo()

#print(qubo)

useQPU = False;
Cs = 420;

# This is exactly same as your code , I haven't yet changed anything in this regards

if useQPU:
    sampler = EmbeddingComposite(DWaveSampler())
    response = sampler.sample_qubo(qubo, chain_strength=Cs, num_reads =1000) #solver=DWaveSampler()) #, num_reads=50)
else:
    sampler = neal.SimulatedAnnealingSampler()
    response = sampler.sample_qubo(qubo, num_sweeps=10000, num_reads=50)

sample = response.first.sample

final_sigx=0;

for i in range(N):
    for j in range(N):
        final_sigx += sample['arr[{}]'.format(i)]*sample['arr[{}]'.format(j)]*sigma[i][j];
final_return=0;

for i in range(N):
    final_return += returns[i]*sample['arr[{}]'.format(i)];

#print obj and final return 
print(final_sigx,final_return)


# print selected values
for i in range (N):
    if(sample['arr[{}]'.format(i)]==1):
        print(i)


#Sum(0, K, lambda i: var[N + j*K +i]*(2**(i))) - v[j])**2