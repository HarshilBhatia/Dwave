import neal
import numpy
import pandas as pd
from pyqubo import Array, Placeholder, solve_qubo, Constraint
from pyqubo import Sum, Model, Mul


# getting values of Returns and Sigma 

r = pd.read_excel('Returns.xlsx')
s = pd.read_excel('covariance1.xlsx')

returns = r['Return']
#print(returns)
sigma = s.loc[:,s.columns!='INDEX'];
sigma = sigma*100*100;
returns = returns*100;
sigma = sigma.to_numpy();



N = 25;
n = 12;

x = Array.create('vector', N, 'BINARY')
#print(x)

# Constraints in our model

#x' Sigmax
min_sigx = 0;

H =0;#temp hamiltonian for the Constraint
for i in range(N):
    H+=x[i]*sigma[i][i]

for i in range(N):
    for j in range(i+1,N):
        H += x[i]*x[j]*sigma[i][j];

min_sigx += Constraint(H,label = "min_portfolio");

H =0 ;#temp hamiltonian for the Constraint
for i in x:
    H+=i;
H-=n;
H = H**2;

select_n = Constraint(H , label = "select_n_projects")

ExpectedReturn = 0 ; # Initilize Expected Return to 0 for now . 

H = 0; #temp hamiltonian for the Constraint
for i in range(N):
    H += returns[i]*x[i];

H -= ExpectedReturn

H = H**2;

expR = Constraint(H, label = "min_expected_return");

lambda1 = 10000;
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

print(sample)