'''
Basis Polynomial Chaos Expansion Modeling

Ref: Sudret, Bruno. "Global sensitivity analysis using polynomial chaos expansions."
Reliability engineering & system safety 93.7 (2008): 964-979.
https://www.sciencedirect.com/science/article/pii/S0951832007001329
'''

import numpy as np
import itertools
import casadi as cas
from collections import Counter
# from sklearn import linear_model
# from sklearn.model_selection import KFold
import json
from .order_scheme import total_combination
from .poly import Hermite, Plain, Legendre
from .train_construct import expand_x

import sys
if sys.version_info.major == 2:
    py = 2
else:
    py = 3

class PCE:
    '''
    Class: polynomial chaos expansion
    nvar: number of variable
    nord: maximum order of multivariate polynomial
    polytype: multivariate type, default: 'Herm'
              available options: 'Herm', 'Legd', 'Plain'
    withdev: whether include the derivative information
    copy_xy: whether store the expanded X and Y for further checking, default False
    regr: regression scheme, default 'OLS'
          available options: 'OLS', 'Lasso', 'Ridge', 'LassoCV', 'RidgeCV'
    jtmax: maximum interaction term, default np.inf
    qnorm: normal of interaction, default 1
    order_list: prededined order list for expansion, default None
    '''
    def __init__(self, nvar, nord=1, polytype='Herm', 
                 jtmax=np.inf, qnorm=1, order_list=None):
        self.nvar = nvar
        self.nord = nord
        self.jtmax = jtmax
        self.qnorm = qnorm
        self.polytype = polytype

        if order_list is None:
            self.order_list = self.generate_order()
        else:
            self.order_list = order_list

        # initialize casadi coef
        self.coef = cas.SX.sym('coef_cas', len(self.order_list))
        self.params = None


    def generate_order(self):
        order_list = total_combination(self.nvar, self.nord, self.jtmax, self.qnorm)
        return order_list

    def initialize(self, x):
        xx = expand_x(self.order_list, x, self.polytype)
        if py == 2:
            xx = cas.vertcat(xx)
            if self.params is None:
                return cas.mul(self.coef.T, xx)
            else:
                params = self.params.reshape(1, -1)
                return cas.mul(params, xx)
        elif py == 3:
            xx = cas.vertcat(*xx)
            if self.params is None:
                return cas.dot(self.coef, xx)
            else:
                return cas.dot(self.params, xx)

    def build_loss(self, x, y):
        if py == 2:
            loss = cas.sum_square(y - self.initialize(x))
        elif py == 3:
            loss = cas.sumsqr(y - self.initialize(x))
        return loss
    
    def fit(self, x=[], y=[], verbose=True):
        loss = 0
        for _x, _y in zip(x, y):
            loss += self.build_loss(_x, _y)
        
        # call casadi optimizer
        nlp = {'x': self.coef, 'f': loss}
        if py == 2:
            prob = cas.NlpSolver('prob', 'ipopt', nlp)
        elif py == 3:
            prob = cas.nlpsol('prob', 'ipopt', nlp)
        coef0 = np.random.uniform(-3, 3, len(self.order_list))

        if not verbose:
            import sys
            import tempfile
            old_stdout = sys.stdout
            sys.stdout = tempfile.TemporaryFile()
        
        # solve the problem
        sol = prob(x0=coef0)['x'].full().T[0]

        if not verbose:
            sys.stdout = old_stdout
        self.params = sol
        return sol
    
    def predict(self, x=[]):

        # define the evaluation function
        xc = cas.SX.sym('x', self.nvar)
        if py == 2:
            yf = cas.SXFunction('yf', [xc, self.coef], [self.initialize(xc)])
        elif py == 3:
            yf = cas.Function('yf', [xc, self.coef], [self.initialize(xc)])
        
        y = []
        for _x in x:
            if py == 2:
                y.append(yf([_x, self.params]))
            elif py == 3:
                y.append(yf(_x, self.params))
        return np.array(y).reshape(-1, 1)

    def save_model(self, mname='pce.json'):
        model = dict()
        model['nvar'] = self.nvar
        model['order_list'] = self.order_list
        model['params'] = self.params.tolist()
        model['polytype'] = self.polytype

        with open(mname, 'w') as fp:
            json.dump(model, fp)

def load_pce(mname='pce.json'):
    with open(mname, 'r') as fp:
        model = json.load(fp)
    nvar = model['nvar']
    order_list = model['order_list']
    params = model['params']
    polytype = model['polytype']

    pce = PCE(nvar=nvar, order_list=order_list, polytype=polytype)
    pce.params = np.array(params)
    return pce
