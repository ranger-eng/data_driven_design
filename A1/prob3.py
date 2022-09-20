import numpy as np
import matplotlib.pyplot as plt

length = 4

mu1 = 1
mu2 = 2
sigma1 = np.sqrt(1)
sigma2 = np.sqrt(2)
r = 3/4

E_x = np.array([mu1,mu2])
covariance = np.array([[sigma1*sigma1, r*sigma1*sigma2],[r*sigma1*sigma2, sigma2*sigma2]])

L = np.linalg.cholesky(covariance)

mu = np.squeeze(np.array([[mu1*np.array(np.ones([1,length]))],[mu2*np.array(np.ones([1,length]))]]))

# generate random variables loop
y1 = np.transpose(np.random.normal(0, 1, length))
y2 = np.transpose(np.random.normal(0, 1, length))

y = np.array([y1,y2])

x = mu + np.dot(L,y)

print(np.shape(y))
print(np.shape(L))
print(np.shape(x))

# plot data
plt.hexbin(x[0,:],x[1,:])
plt.show()
