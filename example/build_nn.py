"""
example of training nn with one output
"""

import casadi as cas
import numpy as np
from caslearn import NN

# define the training dataset
X = np.random.uniform(-1, 1, [300, 2])
y = X[:, 0] ** 2 + np.sin(X[:, 1])

# define the neural network structure
# nin: number of input
# nout: number of output
# nhidden: number of hidden layer
# nhDList: List of dimension of the hidden layer
nn = NN(nin=2, nout=1, nhidden=2, nhDList=[3, 3])

nn.fit([X[:200, :]], [y[:200]])

ypred = nn.predict(X[200:, :])

import matplotlib.pyplot as plt

fig = plt.figure(dpi=200, figsize=(2, 2))
ax = fig.gca()
ax.set_aspect('equal')
ax.plot(y[200:], ypred, 'ro')
ax.plot([-1, 2], [-1, 2], 'k:')
ax.set_xlim([-1, 2])
ax.set_ylim([-1, 2])
ax.tick_params(which='both', direction='in', labelsize=6)
plt.show()

