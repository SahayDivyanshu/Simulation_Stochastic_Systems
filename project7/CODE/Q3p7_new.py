import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import sem
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal

P_expect_list_D1 =[]
P_expect_list_D2 =[]
print('mean for 1st dist.')
mu1 = np.array([0,4])
print(mu1)
print('covariance matrix for distriution 1')
sigma1 = np.array([[3, 0],[0,3]])
print(sigma1)
print('mean for 2nd dist.')
mu2 = np.array([0,150])
print(mu2)
print('covariance matrix for distriution 1')
sigma2 = np.array([[3, 0], [0, 3]])
print(sigma2)
np.random.seed(1) #For reproducibility
r1 = np.random.multivariate_normal(mu1, sigma1,300)
r2 = np.random.multivariate_normal(mu2, sigma2,300)
X = np.vstack((r1,r2)) #Cascade data points
# print("Shape of array:\n", np.shape(X))   
# print(r1)
print(X)
# print(X)
# print("Covarinace matrix of x:\n", np.cov(X)) 
np.random.shuffle(X)
plt.figure(figsize=(9, 9))
plt.scatter(X[:,0], X[:,1])
plt.scatter(mu1[0],mu1[1], marker='^', c='red')
plt.scatter(mu2[0],mu2[1], marker='^', c='red')
plt.title('Data Points without Labels')
plt.show()
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
plt.figure(figsize=(9, 9))
plt.scatter(X[:,0], X[:,1], c=kmeans.labels_.astype(float))
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], marker='^', c='cyan')
print('-- KMEANS ESTIMATE OF MEANS---')
print(kmeans.cluster_centers_[1,:])
# uncomment below to plot the mean values from each distribution
plt.scatter(mu1[0],mu1[1], marker='^', c='red')
plt.scatter(mu2[0],mu2[1], marker='^', c='red')
plt.title('Data Points with Labels by K-means Clustering')
plt.show()
# initializing the mu by the one obtained in from Kmeans
mu_init_1 = kmeans.cluster_centers_[0,:]    #for first normal distribution
mu_init_2 = kmeans.cluster_centers_[1,:]    #for second normal distribution
print('INITIAL PARAMETERS--')
print(mu_init_1)
print(mu_init_2)
#intitializing the covariance matrix for 1st normal distributiuon by Identity
print('---INITIAL SIGMA---')
sigma_init_1 = np.array([[1,0],[0,1]])
sigma_init_2 = np.array([[1,0],[0,1]])
print(sigma_init_1)
print(sigma_init_2)

weight_1 = 0.5
weight_2 = 0.5

P_expect_D1_list = []
P_expect_D2_list = []

for j in range (100):    
    P_expect_D1_list = []
    P_expect_D2_list = []
    #EXPECTATION STEP:
    for i in range(int(len(X))):
        elem1 = weight_1*multivariate_normal.pdf(X[i,:], mu_init_1 ,sigma_init_1) 
        elem2 = weight_2*multivariate_normal.pdf(X[i,:], mu_init_2 ,sigma_init_2)
        P_expect_D1 = elem1/(elem1 + elem2) 
        P_expect_D1_list.append(P_expect_D1)
        P_expect_D2 = elem2/(elem1 + elem2)
        P_expect_D2_list.append(P_expect_D2)
    mu_init_1 = np.array([0,0])
    mu_init_2 = np.array([0,0])
    sigma_init_1 =  np.array([[0,0],[0,0]]) 
    sigma_init_2 =  np.array([[0,0],[0,0]])
    #MAXIMIZATION STEP:
    for k in range(int(len(X))):
        value1_mu = (X[k,:]*P_expect_D1_list[k])/sum(P_expect_D1_list)
        mu_init_1 = mu_init_1 + value1_mu #UPDATING MEAN FOR 1ST DISTRIBUTION
        value2_mu = (X[k,:]*P_expect_D2_list[k])/sum(P_expect_D2_list)
        mu_init_2 = mu_init_2 + value2_mu #UPDATING MEAN FOR 1ST DISTRIBUTION
    for k in range(len(X)):
        val_1 = np.outer((X[k,:]-mu_init_1),(X[k,:]-mu_init_1))
        val_2 = np.outer((X[k,:]-mu_init_2),(X[k,:]-mu_init_2))
        value1_sigma = P_expect_D1_list[k]*val_1/sum(P_expect_D1_list)
        sigma_init_1 = sigma_init_1 + value1_sigma #UPDATING SIGMA FOR 1ST DISTRIBUTION
        value2_sigma = P_expect_D2_list[k]*val_2/sum(P_expect_D2_list)
        sigma_init_2 = sigma_init_2 + value2_sigma #UPDATING SIGMA FOR 2ND DISTRIBUTION
    weight_1 = sum(P_expect_D1_list)/(len(X)) #UPDATING WEIGHTS
    weight_2 = sum(P_expect_D2_list)/(len(X))
print('mu1')    
print(mu_init_1)
print('mu2')
print(mu_init_2)
print('sigma1')
print(sigma_init_1)
print('sigma2')
print(sigma_init_2)