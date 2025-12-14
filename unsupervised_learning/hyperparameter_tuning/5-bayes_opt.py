#!/usr/bin/env python3
"""Hyperparameter Tuning"""
import numpy as np

GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Bayesian Optimization Class"""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """Init the class"""
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.X_s = np.linspace(
            bounds[0],
            bounds[1],
            ac_samples
        ).reshape((-1, 1))
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """Acquisition"""
        mu, sigma = self.gp.predict(self.X_s)
        if self.minimize is False:
            mu_sample_opt = np.amax(self.gp.Y)
            imp = mu - mu_sample_opt - self.xsi
        else:
            mu_sample_opt = np.amin(self.gp.Y)
            imp = mu_sample_opt - mu - self.xsi

        with np.errstate(divide='warn'):
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)

        X_next = self.X_s[np.argmax(ei)]
        return X_next, ei.reshape(-1)

    def optimize(self, iterations=100):
        """Optimise the function by finding the max or the
        min (Specified in the init) of the black block function"""
        obs = set()
        for i in range(iterations):
            self.gp.k = self.gp.kernel(self.gp.X, self.gp.Y)
            X_next, _ = self.acquisition()
            X_next_value = X_next[0]
            Y_next = self.f(X_next)
            if X_next_value in obs:
                break

            self.gp.update(X_next, Y_next)
            obs.add(X_next_value)

        idx_optimum = np.argmin(self.gp.Y) if self.minimize\
            else np.argmax(self.gp.Y)
        self.gp.X = self.gp.X[:-1, :]
        return self.gp.X[idx_optimum], self.gp.Y[idx_optimum]
