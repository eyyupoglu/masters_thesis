import numpy as np
import scipy.stats as ss
import pandas as pd
import matplotlib.pyplot as plt


def simulate_gbmj(S, maturity, steps, lambda_, sigma_y_, sigma_, mu_y_, mu_star):
    dt = maturity/steps
    M_simul = np.zeros(steps)
    jumps = np.zeros(steps)
    M_simul[0] = S
    for i in np.arange(1, steps):
        jumpnb = np.random.poisson( lambda_*dt, size=1)
        jump = np.random.normal(mu_y_ * ( jumpnb - lambda_ * dt ), sqrt( jumpnb ) * sigma_y_, size=1)
        jumps[i] = jump
        M_simul[i] = M_simul[i-1] * np.exp(mu_star * dt + sigma_ * sqrt( dt )* np.random.normal(0, 1, 1) + jump)
    return jumps, M_simul

