import numpy as np

def my_func(x,y):
    return np.exp(5*abs(x-5) + 5*abs(y-5))

N = 1000
fX = np.random.rand(1,N)
fY = np.random.rand(1,N)
X = my_func(fX,fY)
print('Mean is:', str(np.mean(X)))

# stratified sampling
K = 20
XSb = np.zeros((K,K))
SS = np.zeros_like(XSb)
Nij = N/np.power(K,2)

for i in range(0,K):
    for j in range(0,K):
        XS = my_func((i+np.random.rand(1,int(Nij)))/K,(j+np.random.rand(1,int(Nij)))/K)
        XSb[i][j] = np.mean(XS)
        SS[i][j] = np.var(XS)

SST = np.mean((SS/N))
SSM = np.mean((XSb))
print('Mean with stratified sampling is:', str(SSM))


# importance sampling
N_is = 10000
U = np.random.rand(2,N_is)
X_is = np.log(1+(np.exp(1)-1)*U)
T = np.power((np.exp(1)-1),2)*np.exp((np.power(np.sum(X_is,axis=0),2)) - np.sum(X_is,axis=0))
print('Mean is:',str(np.mean(T)))
print(2*np.std(T)/np.sqrt(N))