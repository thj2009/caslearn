import casadi as cas
import numpy as np


import sys
if sys.version_info.major == 3:
    py = 3
else:
    py = 2

class RBF:
    '''
    Radial Basis Function
    '''
    def __init__(self, nele=1, actdims=[]):
        self.nele = nele
        self.actdims = actdims
        self.nP = 1 + len(actdims)
        # log_lengthscale = cas.SX.sym('log_lengthscale', self.nele)
        self.var = cas.SX.sym('var', 1)
        self.lengthscale = cas.SX.sym('lengthscale', self.nele)
        if py == 3:
            self.hyperparam = cas.vertcat(self.var, self.lengthscale)
        else:
            self.hyperparam = cas.vertcat([self.var, self.lengthscale])
        expand_leng = []
        for i in range(len(self.actdims)):
            expand_leng += [self.lengthscale[i]] * len(actdims[i])
        
        if py == 3:
            self.expand_leng = cas.vertcat(*expand_leng)
        else:
            self.expand_leng = cas.vertcat(expand_leng)

    def forward(self, x1, x2):
        d = np.array(x1) - np.array(x2)
        scale_d = d / self.expand_leng
        # return self.var * cas.exp(-cas.norm_2(scale_d)/ 2.)
        return self.var * cas.exp(-cas.sumsqr(scale_d) / 2.)


class GP:
    def __init__(self, kernel, X, y):
        self.kernel = kernel
        self.noise_var = cas.SX.sym('noise_var', 1)
        if py == 3:
            self.hypers = cas.vertcat(self.kernel.hyperparam, self.noise_var)
        else:
            self.hypers = cas.vertcat([self.kernel.hyperparam, self.noise_var])
        self.nP = self.kernel.nP + 1
        self.X = X
        self.y = y
        self.param_array = None
        # casadi object for GP model
        self.alpha = None
        self.K = None   # kernel matrix
        self.log_maginal_likelihood = None
        self.L = None

    def forward_Kf(self, X1, X2=None):
        if X2 is None:
            nd1 = X1.shape[0]
            nd2 = nd1
        else:
            nd1, nd2 = X1.shape[0], X2.shape[0]
        K = cas.SX.sym('K', nd1, nd2)
        if X2 is None:
            for i in range(nd1):
                for j in range(i, nd2):
                        K[i, j] = self.kernel.forward(X1[i, :], X1[j, :])
                        K[j, i] = K[i, j]
        else:
            for i in range(nd1):
                for j in range(nd2):
                    K[i, j] = self.kernel.forward(X1[i, :], X2[j, :])
        return K

    def init_model(self):
        """
        calculate the logarithm maginal likelihood function
        reference: https://groups.google.com/forum/#!topic/casadi-users/3obioBCEQL0
                   https://math.stackexchange.com/questions/109329/can-qr-decomposition-be-used-for-matrix-inversion
        :param X: Input
        :param y: output
        :return: casadi SX function
        """
        X = self.X
        y = self.y
        nd, nf = X.shape
        K = self.forward_Kf(X) +  self.noise_var * cas.SX_eye(nd)
        # Cholesky decomposition
        L = cas.chol(K).T

        alpha = cas.solve(L.T, cas.solve(L, y))
        log_detK = 2 * cas.sum1(cas.log(cas.diag(L)))    # logarithm determinat
        yKy = cas.mtimes(y.T , alpha)                      # deviation
        logP = - 1 / 2. * yKy \
               - 1 / 2. * log_detK \
               - nd / 2. * np.log(2 * np.pi)
        # save the casadi SX object
        self.K = K
        self.alpha = alpha
        self.L = L
        self.log_maginal_likelihood = logP

    def train(self, restart=1, verbose=True):
        self.init_model()       # initialize model
        logP = self.log_maginal_likelihood
        # build casadi nonlinear problem
        nlp = dict(f=-logP, x=self.hypers)
        nlpopts = {}
        nlpopts['ipopt.max_iter'] = 500
        nlpopts['ipopt.tol'] = 1e-4
        #nlpopts['acceptable_tol'] = 1e-6
        #nlpopts['jac_d_constant'] = 'yes'
        #nlpopts['expect_infeasible_problem'] = 'yes'
        # nlpopts['hessian_approximation'] = 'limited-memory'
        nlpopts['ipopt.hessian_approximation'] = 'exact'
        solver = cas.nlpsol('solver', 'ipopt', nlp, nlpopts)
        if not verbose:
            import sys
            import tempfile
            old_stdout = sys.stdout
            sys.stdout = tempfile.TemporaryFile()

        obj_list, sol_list = [], []
        for _ in range(restart):
            # randomize
            x0 = 10 ** (6 * np.random.uniform(0, 1, self.nP) - 3)
            # x0 = [1] + [1] * (self.nP - 2) + [1]
            lbx = [1e-5] + [1e-5] * (self.nP - 2) + [0]
            ubx = [1e5] + [1e5] * (self.nP - 2) + [1e5]

            solution = solver(x0=x0, lbx=lbx, ubx=ubx)
            opt_sol = solution['x'].full().T[0].tolist()
            obj = solution['f'].full()[0][0]
            # print(obj, opt_sol)
            obj_list.append(obj)
            sol_list.append(opt_sol)
        if not verbose:
            sys.stdout = old_stdout
        # get lowest objective
        idxmin = np.argmin(np.array(obj_list))
        self.param_array = sol_list[idxmin]
        print('Best Objective = %.2f' %obj_list[idxmin])
        print('Parameter:')
        print(sol_list[idxmin])

    def predict_func(self, xstar):
        if self.alpha is None:
            self.init_model()
        ks = self.forward_Kf(self.X, xstar)
        # print(ks.shape)
        # print(self.alpha.shape)
        mean = cas.mtimes(ks.T, self.alpha)
        v = cas.solve(self.L, ks)
        kss = self.forward_Kf(xstar, xstar)
        cov = kss - cas.dot(v, v)
        return mean, cov

    def funcname(self, parameter_list):
        pass

    def initialize(self, xstar):
        mean, cov = self.predict_func(xstar)
        mean_cov_func = cas.Function('mean_cov_func', [self.hypers, xstar], [mean, cov])
        mean, cov = mean_cov_func(self.param_array, xstar)
        return mean, cov

    def predict(self, xstar):
        # create SX function
        mean, cov = self.predict_func(xstar)
        mean_cov_func = cas.Function('mean_cov_func', [self.hypers], [mean, cov])
        # eval mean and cov function
        mean = mean_cov_func(self.param_array)[0].full().T[0]
        cov = mean_cov_func(self.param_array)[1].full()
        return mean, cov

class MTGP:
    pass

#%%
if __name__ == "__main__":
    import numpy as np
    x = np.random.uniform(0, 10, (200, 2))
    y = np.sin(x[:, 0]) + 1/3. * x[:, 1]**2
    # y += np.random.normal(0, 0.5, x.shape[0])

    # x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    k = RBF(1, [[0, 1]])
    model = GP(k, x, y)
    model.train()
    print(model.param_array)
    xx = np.sort(np.random.uniform(0, 5, (30, 2)))
    yy = np.sin(xx[:, 0]) + 1/3. * xx[:, 1] ** 2
    ypred, cov_pred = model.predict(xx)
    std_pred = [np.sqrt(c) for c in np.diagonal(cov_pred)]
    import matplotlib.pyplot as plt
    plt.figure()
    # plt.plot(xx, yy, 'ro:')
    max_, min_ = np.max(np.concatenate([yy, ypred])), np.min(np.concatenate([yy, ypred]))
    d = max_ - min_
    r = [min_-0.1*d, max_+0.1*d]
    plt.errorbar(yy, ypred, yerr=std_pred, fmt='ro')
    plt.xlim(r)
    plt.ylim(r)
    plt.plot(r, r, 'k:')
    plt.show()

