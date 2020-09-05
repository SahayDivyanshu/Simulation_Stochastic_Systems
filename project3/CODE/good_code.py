from scipy.stats import rv_discrete
numbers = (1,2,3)
distribution = (1./6, 2./6, 3./6)
random_variable = rv_discrete(values=(numbers,distribution))
random_variable.rvs(size=10)