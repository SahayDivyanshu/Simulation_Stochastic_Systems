import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
wait = []
dura = []
with open('D:\\SHIVAM\\USC_STUDY\\EE511\\project7\\code\\q4\\DATA.dat') as f:
    wait = [float(line.split()[1]) for line in f]
with open('D:\\SHIVAM\\USC_STUDY\\EE511\\project7\\code\\q4\\DATA.dat') as f:  
    dura = [float(line.split()[2]) for line in f]   
print(wait)
print(dura)
mu_wait = sum(wait)/len(wait)
mu_dura = sum(dura)/len(dura)
print(mu_dura)
print(mu_wait)
kmeans_plot = np.array(list(zip(wait, dura)))
plt.figure(figsize=(9, 9))
plt.xlabel('DURATION')
plt.ylabel('WAITING TIME')
plt.title('2D-SCATTER PLOT OF THE DATA')
plt.scatter(wait,dura,c='blue')
plt.show()
#K-MEANS CLUSTERING
kmeans = KMeans(n_clusters=2, random_state=0).fit(kmeans_plot)
# plt.figure(figsize=(9, 9))
plt.scatter(kmeans_plot[:,0], kmeans_plot[:,1], c=kmeans.labels_.astype(float))
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], marker='^', c='red')
plt.title('Data Points with Labels by K-means Clustering')
plt.xlabel('DURATION')
plt.ylabel('WAITING TIME')
plt.show()
weight_1 = 0.5
weight_2 = 0.5
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
P_expect_D1_list = []
P_expect_D2_list = []
for j in range (5):    
    P_expect_D1_list = []
    P_expect_D2_list = []
    #EXPECTATION STEP:
    for i in range(int(len(kmeans_plot))):
        elem1 = weight_1*multivariate_normal.pdf(kmeans_plot[i,:], mu_init_1 ,sigma_init_1) 
        elem2 = weight_2*multivariate_normal.pdf(kmeans_plot[i,:], mu_init_2 ,sigma_init_2)
        P_expect_D1 = elem1/(elem1 + elem2) 
        P_expect_D1_list.append(P_expect_D1)
        P_expect_D2 = elem2/(elem1 + elem2)
        P_expect_D2_list.append(P_expect_D2)
    mu_init_1 = np.array([0,0])
    mu_init_2 = np.array([0,0])
    sigma_init_1 =  np.array([[0,0],[0,0]]) 
    sigma_init_2 =  np.array([[0,0],[0,0]])
    #MAXIMIZATION STEP:
    for k in range(int(len(kmeans_plot))):
        value1_mu = (kmeans_plot[k,:]*P_expect_D1_list[k])/sum(P_expect_D1_list)
        mu_init_1 = mu_init_1 + value1_mu #UPDATING MEAN FOR 1ST DISTRIBUTION
        value2_mu = (kmeans_plot[k,:]*P_expect_D2_list[k])/sum(P_expect_D2_list)
        mu_init_2 = mu_init_2 + value2_mu #UPDATING MEAN FOR 1ST DISTRIBUTION
    for k in range(len(kmeans_plot)):
        val_1 = np.outer((kmeans_plot[k,:]-mu_init_1),(kmeans_plot[k,:]-mu_init_1))
        val_2 = np.outer((kmeans_plot[k,:]-mu_init_2),(kmeans_plot[k,:]-mu_init_2))
        value1_sigma = P_expect_D1_list[k]*val_1/sum(P_expect_D1_list)
        sigma_init_1 = sigma_init_1 + value1_sigma #UPDATING SIGMA FOR 1ST DISTRIBUTION
        value2_sigma = P_expect_D2_list[k]*val_2/sum(P_expect_D2_list)
        sigma_init_2 = sigma_init_2 + value2_sigma #UPDATING SIGMA FOR 2ND DISTRIBUTION
    weight_1 = sum(P_expect_D1_list)/(len(kmeans_plot)) #UPDATING WEIGHTS
    weight_2 = sum(P_expect_D2_list)/(len(kmeans_plot))
print('mu1')    
print(mu_init_1)
print('mu2')
print(mu_init_2)
print('sigma1')
print(sigma_init_1)
print('sigma2')
print(sigma_init_2)
r1 = np.random.multivariate_normal(mu_init_1, sigma_init_1,300) #GENERATING GAUSSIAN MIXTURE FOR THE ESTIMATION
r2 = np.random.multivariate_normal(mu_init_2, sigma_init_2,300)
X = np.vstack((r1,r2))
np.random.shuffle(X)
plt.figure(figsize=(8,8))
x, y = np.mgrid[0:5:.01, 30:100:.01]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x; pos[:, :, 1] = y
rv1 = multivariate_normal(mu_init_1,sigma_init_1)
plt.contour(x, y, rv1.pdf(pos)) #CONTOUR PLOT
rv2 = multivariate_normal(mu_init_2,sigma_init_2)
plt.contour(x, y, rv2.pdf(pos))
plt.scatter(wait,dura,c='blue')
plt.xlabel('DURATION')
plt.ylabel('WAITING TIME')
plt.title('CONTOUR OVERLAYED WITH SCATTER')
plt.show()

#cluster prediction by GMM
plt.figure(figsize=(8,8))
plt.scatter(X[:,0], X[:,1]) 
plt.scatter(mu_init_1[0],mu_init_1[1], marker='^', c='red')
plt.scatter(mu_init_2[0],mu_init_2[1], marker='^', c='red')
plt.xlabel('DURATION')
plt.ylabel('WAITING TIME')
plt.title('CLUSTER PREDICTION BY GMM')
plt.show()

