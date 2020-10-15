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


N = 15;
n = 7;
R = 300 ; # Initilize Expected Return to 0 for now . 

sum=0;
for i in range(N):
    sum += returns[i];

K = math.log2(sum-1);
K = math.floor(K);
K+=1;
#print(returns)
x = Array.create('arr', N+K, 'BINARY')
#print(x)

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

#print(x)
select_n = Constraint(H , label = "select_n_projects")
#print(select_n)

H = 0; #temp hamiltonian for the Constraint
for i in range(N):
    H += returns[i]*x[i];

H -= R

for i in range(0,K):
    H-= (2**i)*(x[i+N]);

H = H**2;

expR = Constraint(H, label = "min_expected_return");
numreads= 10000;
mx=0;
mn=1e9;
for i in range(N):
    mx = max(mx,returns[i]);
    mn = min(mn,returns[i]);

for i in range (1,2):
    sl = 0;
    lambda1 = (mx-mn);
    lambda1 *= 10;
    lambda1 = int(lambda1)
    #print(lambda1)
    lambda2 = 10;
    #mul = 20;
    H = min_sigx + lambda1*select_n + lambda2*expR;
    print(R,N,n,numreads,lambda1,lambda2);
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
        response = sampler.sample_qubo(qubo, num_sweeps=10000, num_reads=numreads)

    #sample = response.first.sample
    
    
    """
    final_sigx=0;

    for i in range(N):
        for j in range(N):
            final_sigx += sample['arr[{}]'.format(i)]*sample['arr[{}]'.format(j)]*sigma[i][j];
    final_return=0;

    for i in range(N):
        final_return += returns[i]*sample['arr[{}]'.format(i)];


    print(final_sigx,final_return)
    """

    x=response.first.sample
    mn=1e9;

    for sample in response.samples():   
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
        print(final_sigx,n_temp)

        if(final_sigx<mn and n_temp == n and ret>=R):
            mn=final_sigx;
            x = sample;
            final_ret = ret;
        
        ''' print(mn)
        #print(x)
        print(final_ret)
        '''
        
   # print(x)
    """
    for i in range(N):
        if(x['arr[{}]'.format(i)]==1):
            print(i)

    print(mn)
    print(final_ret)
    """
        #Sum(0, K, lambda i: var[N + j*K +i]*(2**(i))) - v[j])**2 