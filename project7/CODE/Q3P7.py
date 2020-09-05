import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans
P_expect_list_D1 =[]
P_expect_list_D2 =[]
mu1 = np.array([5,8])
sigma1 = np.array([[1, 1.5],[1.5, 3]])
mu2 = np.array([-3, 3])
sigma2 = np.array([[3, -1.5], [-1.5, 1.1]])
np.random.seed(1) #For reproducibility
r1 = np.random.multivariate_normal(mu1, sigma1,300)
r2 = np.random.multivariate_normal(mu2, sigma2,300)
X = np.vstack((r1,r2)) #Cascade data points
print("Shape of array:\n", np.shape(X))   
# print(len(X))
# print("Covarinace matrix of x:\n", np.cov(X)) 
np.random.shuffle(X)
# plt.figure(figsize=(9, 9))
# plt.scatter(X[:,0], X[:,1])
# plt.scatter(mu1[0],mu1[1], marker='^', c='red')
# plt.scatter(mu2[0],mu2[1], marker='^', c='red')
# plt.title('Data Points without Labels')
# plt.show()
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
# plt.figure(figsize=(9, 9))
# plt.scatter(X[:,0], X[:,1], c=kmeans.labels_.astype(float))
# plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], marker='^', c='cyan')
print('-- KMEANS ESTIMATE OF MEANS---')
# print(kmeans.cluster_centers_[1,:])
# # uncomment below to plot the mean values from each distribution
# plt.scatter(mu1[0],mu1[1], marker='^', c='red')
# plt.scatter(mu2[0],mu2[1], marker='^', c='red')
# plt.title('Data Points with Labels by K-means Clustering')
# plt.show()

print('-Expectation maximization--')
# Initializing

#initializing the mu by the one obtained in from Kmeans
mu_init_1 = kmeans.cluster_centers_[0,:]    #for first normal distribution
mu_init_2 = kmeans.cluster_centers_[1,:]    #for second normal distribution
print(mu_init_1)

#intitializing the covariance matrix for 1st normal distributiuon by Identity
sigma_init_1 = [[1,0],[0,1]] 

sigma_x_1 = sigma_init_1[0][0] #sigma X for 1st normal distribution
sigma_y_1 = sigma_init_1[1][1] #sigma Y for 1st normal distribution

#intitializiing the covariance matrix for 2nd normal distributiuon by Identity
sigma_init_2 = [[1,0],[0,1]]

sigma_x_2 = sigma_init_2[0][0] #sigma X for 2nd normal distribution
sigma_y_2 = sigma_init_2[1][1] #sigma Y for 2nd normal distribution

mu_x_1 = mu_init_1[0] #mu X for 1st normal distribution
mu_y_1 = mu_init_1[1] #mu Y for 1st normal distribution

mu_x_2 = mu_init_2[0] #mu X for 2nd normal distribution
mu_y_2 = mu_init_2[1] #mu Y for 2nd normal distribution


rho_xy_1 = sigma_init_1[0][1]/sigma_x_1*sigma_y_1 #correlation coefficient for the first normal distribution

rho_xy_2 = sigma_init_2[0][1]/sigma_x_2*sigma_y_2 #correlation coefficient for the second normal distribution

mu_est_X_1_final = 0
mu_est_Y_1_final = 0
mu_est_X_2_final = 0
mu_est_Y_2_final = 0


for j in range (3):
    for i in range(len(X)):
        #intermediate values used in the first pdf
        value1 = (1 - rho_xy_1*rho_xy_1) 
        const_1 = 1/(2*np.pi*sigma_x_1*sigma_y_1*np.sqrt(value1)) 
        elem1 = ((X[i][0]-mu_x_1)/sigma_x_1)*((X[i][0]-mu_x_1)/sigma_x_1)
        elem2 = ((X[i][1]-mu_y_1)/sigma_y_1)*((X[i][1]-mu_y_1)/sigma_y_1)
        elem3_1 = (X[i][0]-mu_x_1)/sigma_x_1
        elem3_2 = (X[i][1]-mu_x_1)/sigma_x_1
        elem3 = 2*rho_xy_1*elem3_1*elem3_2
        weight_1 = 0.5
        f_xy_1 =  const_1 * np.exp(-(elem1 + elem2 - elem3)/(2*value1))

               
        #intermediate values used in the first pdf
        value2 = (1 - rho_xy_2*rho_xy_2)
        const_2 = 1/(2*np.pi*sigma_x_2*sigma_y_2*np.sqrt(value2))
        elem1_2 = ((X[i][0]-mu_x_2)/sigma_x_2)**2
        elem2_2 = ((X[i][1]-mu_y_2)/sigma_y_2)**2
        elem3_1_2 = (X[i][0]- mu_x_2)/sigma_x_2
        elem3_2_2 = (X[i][1]- mu_x_2)/sigma_x_2
        elem3_2 = 2*rho_xy_1*elem3_1*elem3_2
        weight_2 = 0.5
        f_xy_2 =  const_2 * np.exp(-(elem1_2 + elem2_2 - elem3_2)/(2*value2))

        P_expect_D1 = (weight_1*f_xy_1) / (weight_1*f_xy_1 + weight_2*f_xy_2) 
        P_expect_list_D1.append(P_expect_D1)
        P_expect_D2 = (weight_2*f_xy_2) / (weight_1*f_xy_1 + weight_2*f_xy_2) 
        P_expect_list_D2.append(P_expect_D2)
    
    #maximization step:

    for k in range(len(X)):
        mu_est_X_1 = X[k][0]*P_expect_list_D1[k]
        mu_est_X_1_final = mu_est_X_1_final + mu_est_X_1

        mu_est_Y_1 = X[k][1]*P_expect_list_D1[k]
        mu_est_Y_1_final = mu_est_Y_1_final + mu_est_Y_1

        mu_est_X_2 = X[k][0]*P_expect_list_D2[k]
        mu_est_X_2_final = mu_est_X_2_final + mu_est_X_2

        mu_est_Y_2 = X[k][1]*P_expect_list_D2[k]
        mu_est_Y_2_final = mu_est_Y_2_final + mu_est_Y_2

print(mu_est_X_1_final)
print(mu_est_Y_1_final)
print(mu_est_X_2_final)
print(mu_est_Y_2_final)
