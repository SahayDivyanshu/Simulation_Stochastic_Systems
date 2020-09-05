import random
import numpy as np
from scipy.stats import sem


with open('DATA.dat') as f:
    li = [line.split()[2] for line in f]

sample_stat = []
for i in range(15):
    sample_stat.append(int(li[i]))

print(sample_stat)

print('------BOOTSTRAP SAMPLING------')
mean_bootstrap_list = []
for i in range(100):
    bootstrap_sample = np.random.choice(sample_stat, 100)
    mean_bootstrap = (sum(bootstrap_sample)/len(bootstrap_sample))
    mean_bootstrap_list.append(mean_bootstrap)

# print(mean_bootstrap_list)
lower_bound_bootstrap_percentile = np.percentile(mean_bootstrap_list, 2.5)
upper_bound_bootstrap_percentile = np.percentile(mean_bootstrap_list, 97.5)
print(lower_bound_bootstrap_percentile)
print(upper_bound_bootstrap_percentile)

print('------STATISTICAL SAMPLING------')
mean_stat = (sum(sample_stat)/len(sample_stat))
mean_stat_95ci = 1.96 * sem(sample_stat)
lower_bound_stat_percentile = mean_stat - mean_stat_95ci
upper_bound_stat_percentile = mean_stat + mean_stat_95ci
print(lower_bound_stat_percentile)
print(upper_bound_stat_percentile)


