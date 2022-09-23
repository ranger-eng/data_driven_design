import numpy as np
import matplotlib.pyplot as plt
import math

def gauss(x,mu,sigma):
    x = float(x)
    mu = float(mu)
    sigma = float(sigma)
    y = 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-mu)*(x-mu)/(2*sigma*sigma))
    return y

def gauss2d(x1,mu1,x2,mu2,covar):
    mean_array = np.array([x1-mu1, x2-mu2])
    exponent = -np.dot(np.dot(mean_array,covar), mean_array.T)/2
    exponent = math.exp(exponent) 
    y = 1/(2*np.pi * np.sqrt(np.linalg.det(covar)))*exponent
    return y

# setup
length = 101 

mu1 = 1
mu2 = 2
mu = np.squeeze(np.array([[mu1*np.array(np.ones([1,length]))],[mu2*np.array(np.ones([1,length]))]]))

sigma1 = np.sqrt(1)
sigma2 = np.sqrt(2)
r = .75
E_x = np.array([mu1,mu2])
covariance = np.array([[sigma1*sigma1, r*sigma1*sigma2],[r*sigma1*sigma2, sigma2*sigma2]])
L = np.linalg.cholesky(covariance)


# generate random variables
y1 = np.transpose(np.random.normal(0, 1, length))
y2 = np.transpose(np.random.normal(0, 1, length))
y = np.array([y1,y2])
x = mu + np.dot(L,y)

# generate ground truth
truth1 = np.zeros(length)
truth2 = np.zeros(length)
truthx = np.zeros(length)
for i in range(0, length):
    temp = float(i)*10.0/float(length) - 5.0
    truthx[i] = temp
    truth1[i] = gauss(temp, mu1, sigma1)
    truth2[i] = gauss(temp, mu2, sigma2)

truthx_, truthy_ = np.meshgrid(truthx,truthx)
truthz_ = np.zeros([length,length])
for i in range(0, length):
    for ii in range(0, length):
        truthz_[i,ii] = gauss2d(truthx_[i,ii],mu1,truthy_[i,ii],mu2,covariance)

# plot the marginal vs actual
f, (ax1, ax2) = plt.subplots(1, 2)
ax1.hist(x[0,:],bins=40, normed= True)
ax1.plot(truthx,truth1)
ax1.set_xlim(-8,8)
ax1.set_ylim(0,.45)
ax2.hist(x[1,:],bins=40, normed= True)
ax2.plot(truthx,truth2)
ax2.set_xlim(-8,8)
ax2.set_ylim(0,.45)
plt.show()

# plot contour plot and actual ground truth 2-d
f, (ax1, ax2) = plt.subplots(1,2)
ax1.contourf(truthy_ , truthx_, truthz_.T)
ax1.set_xlim(-5,5)
ax1.set_ylim(-5,5)
ax2.hist2d(x[0,:],x[1,:])
ax2.set_xlim(-5,5)
ax2.set_ylim(-5,5)
plt.show()


