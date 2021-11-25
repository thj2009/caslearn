"""
Collection of activation function
"""

import sys
import casadi as cas

if sys.version_info.major == 3:
    py = 3
else:
    py = 2

def sigmoid():
    x = cas.SX.sym('x', 1)
    if py == 3:
        return cas.Function('sigmoid', [x], [1. / (1 + cas.exp(-x))])
    else:
        return cas.SXFunction('sigmoid', [x], [1. / (1 + cas.exp(-x))])

def const():
    x = cas.SX.sym('x', 1)
    if py == 3:
        return cas.Function('sigmoid', [x], [x])
    else:
        return cas.SXFunction('sigmoid', [x], [x])

def tanh():
    x = cas.SX.sym('x', 1)
    if py == 3:
        return cas.Function('tanh', [x], [(cas.exp(x) - cas.exp(-x)) / (cas.exp(x) + cas.exp(-x))])
    else:
        return cas.SXFunction('tanh', [x], [(cas.exp(x) - cas.exp(-x)) / (cas.exp(x) + cas.exp(-x))])