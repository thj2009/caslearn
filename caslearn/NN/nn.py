import numpy as np
import json
import casadi as cas
from .activation import *

import sys
if sys.version_info.major == 3:
    py = 3
else:
    py = 2

class Layer:
    def __init__(self, nin, nout, actiF=None):
        self.nin = nin
        self.nout = nout
        self.weight = cas.SX.sym('weight', nout, nin + 1)
        self.params = None
        self.actiF = actiF

    def forward(self, xin):
        # print(xin)
        if py == 3:
            xx = cas.vertcat(1, xin)        # append bias
            wx = cas.mtimes(self.weight, xx)
            y = [self.actiF(wx[i]) for i in range(self.nout)]
            return cas.vertcat(*y)
        else:
            xx =cas.vertcat([1, xin])
            wx = cas.mul(self.weight, xx)
            y = [self.actiF([wx[i]])[0] for i in range(self.nout)]
            print(len(y))
            return cas.vertcat(y)


class NN:
    """
    Neural Network
    """
    def __init__(self, nin, nout=1, nhidden=3, nhDList=[1, 1, 1], actiF=sigmoid()):
        self.nin = nin
        self.nout = nout
        self.nhidden = nhidden
        self.nhDList = nhDList
        assert nhidden == len(nhDList)
        self.actiF = actiF
        self.hidden_layer = None

        self.nweights = 0
        self.weights = None
        self.nn_func = None
        self.params = None

        self.construct()
        self.link()

        self.constraint = []
        self.lbG = []
        self.ubG = []
    def construct(self):

        # construct multiple hidden layer
        layers = []
        for i in range(self.nhidden + 1):
            if i == 0:                       # first hidden layer
                l = Layer(self.nin, self.nhDList[i], self.actiF)
            elif i == self.nhidden:      # last hidden layer
                l = Layer(self.nhDList[i - 1], self.nout, const())
            else:                            # middle layer
                l = Layer(self.nhDList[i - 1], self.nhDList[i], self.actiF)

            layers.append(l)
        self.hidden_layer = layers

        # construct weight variable
        ww = []
        for l in layers:
            ww.append(cas.reshape(l.weight, l.weight.numel(), 1))
        if py == 3:
            self.weights = cas.vertcat(*ww)
        else:
            self.weights = cas.vertcat(ww)
        self.nweights = self.weights.shape[0]

    def link(self):
        input = cas.SX.sym('input', self.nin)
        for i in range(self.nhidden + 1):
            if i == 0:                       # first hidden layer
                output = self.hidden_layer[i].forward(input)
            else:
                output = self.hidden_layer[i].forward(output)
        output = cas.horzsplit(output)
        
        if py == 3:
            nn = cas.Function('nn', [input, self.weights], output)
        else:
            nn = cas.SXFunction('nn', [input, self.weights], output)
        self.nn_func = nn
    
    def add_constraint(self, inlet, bound=[], idx=0):
        
        if py == 3:
            constraint = self.nn_func(inlet, self.weights)[idx]
        else:
            constraint = self.nn_func([inlet, self.weights])[idx]
        
        self.constraint.append(constraint)
        self.lbG.append(bound[0])
        self.ubG.append(bound[1])

    def fit(self, Xs, ys):
        loss = 0
        for i, (X, y) in enumerate(zip(Xs, ys)):
            for _x, _y in zip(X, y):
                # SSE
                if py == 3:
                    err = self.nn_func(_x, self.weights)[i] - _y
                else:
                    err = self.nn_func([_x, self.weights])[i] - _y
                loss += err ** 2
        if self.constraint == []:
            nlp = {'x': self.weights, 'f': loss}
        else:
            nlp = {'x': self.weights, 'f': loss, 'g': cas.vertcat(*self.constraint)}
        if py == 3:
            # cal casadi solver
            nlpopt = {}
            nlpopt["ipopt.max_iter"] = 20000
            nlpopt["ipopt.tol"] = 1e-6
            prob = cas.nlpsol('prob', 'ipopt', nlp, nlpopt)
        else:
            # cal casadi solver
            nlpopt = {}
            nlpopt["max_iter"] = 50000
            nlpopt["tol"] = 1e-6
            prob = cas.NlpSolver('prob', 'ipopt', nlp, nlpopt)
        # if self.params is None:
        coef0 = np.random.uniform(-3, 3, self.nweights)
        # else:
        #     coef0 = np.copy(self.params)
        if self.constraint == []:
            solution = prob(x0=coef0)
        else:
            solution = prob(x0=coef0, lbg=self.lbG, ubg=self.ubG)
        
        sol = solution['x'].full().T[0]
        self.params = sol

    def predict(self, x, nout=0):
        y = []
        for _x in x:
            if py == 3:
                y.append(self.nn_func(_x, self.params)[nout].full()[0][0])
            else:
                y.append(self.nn_func([_x, self.params])[nout].full()[0][0])
        return np.array(y).reshape(-1, 1)

    def initialize(self, x):
        if py == 3:
            return self.nn_func(x, self.params)
        else:
            return self.nn_func([x, self.params])

    def save_model(self, mname='nn.json'):
        model = dict()
        model['nin'] = self.nin
        model['nout'] = self.nout
        model['nhidden'] = self.nhidden
        model['nhDlist'] = self.nhDList
        model['params'] = self.params.tolist()


        with open(mname, 'w') as fp:
            json.dump(model, fp)

def load_nn(mname='nn.json'):
    with open(mname, 'r') as fp:
        model = json.load(fp)
    nin = model['nin']
    nout = model['nout']
    nhidden = model['nhidden']
    nhDlist = model['nhDlist']
    params = model['params']

    nn = NN(nin=nin, nout=nout, nhidden=nhidden, nhDList=nhDlist)
    nn.params = np.array(params)
    return nn


if __name__ == "__main__":
    x = cas.SX.sym('x', 1)
    actiF = cas.Function('f', [x], [1. / (1 + cas.exp(-x))])
    l = Layer(3, 4, actiF)

    xin = cas.SX.sym('xin', 3)
    print(l.forward(xin))


    X = np.random.uniform(-1, 1, [300, 2])
    y = X[:, 0] ** 2 + np.sin(X[:, 1])

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