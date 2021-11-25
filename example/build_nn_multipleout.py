"""
example of training nn with multiple outputs
"""

import casadi as cas
import numpy as np
from caslearn import NN

# define the training dataset
X = np.random.uniform(-1, 1, [300, 2])
y1 = X[:, 0] ** 2 + np.sin(X[:, 1])
y2 = np.cosh(X[:, 0]) + 3 * X[:, 1]

Xs = [X[:200, :], X[:100, :]]
ys = [y1[:200], y2[:100]]

# define the neural network structure
# nin: number of input
# nout: number of output
# nhidden: number of hidden layer
# nhDList: List of dimension of the hidden layer

nn = NN(nin=2, nout=2, nhidden=2, nhDList=[3, 3])


# train the model
nn.fit(Xs, ys)


# visualization

# here we compare the prediction of the second output
# nout: the index of output
ypred = nn.predict(X[200:, :], nout=1)

# ypred = nn.predict(X[200:, :], nout=0)

import matplotlib.pyplot as plt

fig = plt.figure(dpi=200, figsize=(2, 2))
ax = fig.gca()
ax.set_aspect('equal')
ax.plot(y2[200:], ypred, 'ro')
ax.plot([-1, 2], [-1, 2], 'k:')
ax.set_xlim([-1, 2])
ax.set_ylim([-1, 2])
ax.tick_params(which='both', direction='in', labelsize=6)
plt.show()

