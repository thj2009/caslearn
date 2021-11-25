# test initialization of PCE

import casadi as cas
import numpy as np
from caslearn import PCE

p = PCE(nvar=2, nord=3)

x = np.random.uniform(-2, 2, [10, 2])
y = 3 * x[:, 0] + x[:, 1] ** 2

sol = p.fit(x, y)
# print(sol)
# print(p.params)

# x = cas.SX.sym('x', 2)
# print(p.initialize(x, sol))
# print(x)
p.predict(x)