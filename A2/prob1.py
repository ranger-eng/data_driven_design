import numpy as np
import matplotlib as plt

# COnsider the system
#
# t = w0 + w1*x1 + e
# where w0 = -.3 and w1 = .5. The random variable e is drawn from the
# normal distribution with mean 0, and variance .2^2. The random variable
# x1 is uniformly distributed between (-1,1). Generate the observation t for 40 trials
 
# setup
N = 400
w0_truth = -.3
w1_truth = .5

x = np.random.normal(loc=0.0, scale=.2, size=(1,N))
t = w0_truth + w1_truth*x + np.random.uniform(low=-1.0, high=1.0, size=(1,N))

# part a.) 
mean_estimate = t.sum()/N
variance_estimate = np.var(t)

# part b.)
j = N
t_sub = t[:,0:j].T
temp = np.squeeze(x[:,0:j])
x_sub2 = np.array([np.ones(j),temp]).T

w_guess = np.linalg.solve(np.dot(x_sub2.T,x_sub2), np.dot(x_sub2.T,t_sub))
print(w_guess)
