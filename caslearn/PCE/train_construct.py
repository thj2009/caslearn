"""
Build Training dataset give orderlist and polynomial type
"""

import numpy as np

from .poly import Hermite, Plain, Legendre

# def build_xy(order_list, poly, x, y):
#     '''
#     build large X, Y for linear regression
#     '''
#     X = expand_x(order_list, x, poly)
#     Y = np.array(y).flatten()
#     return X, Y



def expand_x(order_list, x, polytype):
    '''
    expand x according to orderlist
    '''
    if polytype == 'Herm':
        Poly = Hermite
    elif polytype == 'Plain':
        Poly = Plain
    elif polytype == 'Legd':
        Poly = Legendre
    else:
        raise ValueError("%s is not available" % polytype)

    nvar = x.shape[0]
    norder = len(order_list)

    X = []      # initialize the input matrix X

    for i in range(norder):
        order = order_list[i]
        xx = 1
        for j in range(nvar):
            o = order[j]
            xx *= Poly(order=o).evaluate(x[j])
        X.append(xx)
    return X

