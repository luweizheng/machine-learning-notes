import numpy as np
import time
from linreg import LinearRegression

# Load the data 
# The data is copied from the linear regression exercise of machine learning course from Andrew Ng
# The file ex1data2.txt contains a training set of housing prices in Portland, Oregon. The first column is the 
# size of the house (in square feet), the second column is the number of bedrooms, 
# and the third column is the price of the house, which we want to predict.
file_name = 'dataset/ex1data2.txt'
with open(file_name, 'r') as f:
    house_data = np.loadtxt(file_name, delimiter=',')
    
num_sample = house_data.shape[0] # number of all the samples
X = house_data[:, :2]
y = house_data[:, 2].reshape((-1,1))

# Add intercept term or bias to X
print('X shape: ', X.shape)
print('y shape: ', y.shape)
print('First 10 examples from the dataset')
print(house_data[0:10, :])

# Normalize
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
# Add bias dimension
X = np.hstack((X, np.ones((num_sample, 1))))

lr_bgd = LinearRegression()
tic = time.time()
losses_bgd = lr_bgd.train(X, y, method='sgd', learning_rate=1e-2, num_iters=1000, verbose=True)
toc = time.time()
print('Traning time for BGD with vectorized version is %f \n' % (toc - tic))
print(lr_bgd.W)