import QuantLib as ql
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from math import exp, log, sqrt, pi
import pandas as pd
import time

from scipy.optimize import minimize, rosen, rosen_der





class Heston(object):
    def __init__(self):
        pass

    @staticmethod
    def simulate(scheme, negvar, numPaths, S_0, maturity, dt, r, q, kappa, theta, sigma, v0, rho):
        num_time = int(maturity / dt)
        S = np.zeros((num_time + 1, numPaths))
        S[0, :] = S_0
        V = np.zeros((num_time + 1, numPaths))
        V[0, :] = v0
        Vcount0 = 0
        for i in range(numPaths):
            for t_step in range(1, num_time + 1):
                # the 2 stochastic drivers for variance V and asset price S and correlated
                Zv = np.random.randn(1)
                Zs = rho * Zv + sqrt(1 - rho ** 2) * np.random.randn(1)
                # users can choose either Euler or Milstein scheme
                if scheme == 'Euler':
                    V[t_step, i] = V[t_step - 1, i] + kappa * (theta - V[t_step - 1, i]) * dt + sigma * sqrt(
                        V[t_step - 1, i]) * sqrt(dt) * Zv
                elif scheme == 'Milstein':
                    V[t_step, i] = V[t_step - 1, i] + kappa * (theta - V[t_step - 1, i]) * dt + sigma * sqrt(
                        V[t_step - 1, i]) * sqrt(dt) * Zv \
                                   + 1 / 4 * sigma ** 2 * dt * (Zv ** 2 - 1)

                if V[t_step, i] <= 0:
                    Vcount0 = Vcount0 + 1
                    if negvar == 'Reflect':
                        V[t_step, i] = abs(V[t_step, i])
                    elif negvar == 'Trunca':
                        V[t_step, i] = max(V[t_step, i], 0)

                ################         simluations for asset price S              ########
                S[t_step, i] = S[t_step - 1, i] * np.exp((r - q - V[t_step - 1, i] / 2) * dt + sqrt(V[t_step - 1, i]) * sqrt(dt) * Zs)
        return S, V, Vcount0

    @staticmethod
    def likelihoodAW(param, x, r, q, dt, method):
        kappa = param[0]
        theta = param[1]
        sigma = param[2]
        v0 = param[3]
        rho = param[4]

        # Atiya and Wall parameterization
        alpha = kappa * theta
        beta = kappa

        #  Number of log - stock prices
        T = len(x)

        # Drift term
        mu = r - q

        # Equation (17)
        betap = 1 - beta*dt

        # Equation (18) - denominator of d(t)
        D = 2 * pi * sigma * np.sqrt(1 - rho** 2) * dt


        # Equation (14)
        a = (betap**2 + rho*sigma*betap*dt + sigma**2 * dt**2 / 4) / (2 * sigma**2 * (1-rho**2) *dt)

        # Variance and likelihood at time t = 0
        v = np.zeros(len(x))
        L = np.zeros(len(x))
        v[0] = v0
        if method == 1:
            L[0] = np.exp(-v[0])   # Construct the Likelihood
        elif method == 2:
            L[0] = -v[0]         # Construct the log-likelihood

        # Construction the likelihood for time t = 1 through t = T
        for t in range(T-1):
            # Stock price increment
            dx = x[t + 1] - x[t]
            # Equations (31) and (32)
            B = -alpha*dt - rho*sigma*(dx-mu*dt)
            C = alpha**2*dt**2 + 2*rho*sigma*alpha*dt*(dx-mu*dt) + sigma**2*(dx-mu*dt)**2 - 2*v[t]**2*a*sigma**2*(1-rho**2)*dt
            # Equation (30) to update the variance
            if B**2 - C > 0:
                v[t+1] = np.sqrt(B**2 - C) - B
            else:
                # If v(t+1) is imaginary use the approximation Equation (33)
                bt = ((v[t]-alpha*dt)**2 - 2*rho*sigma*(v[t]-alpha*dt)*(dx-mu*dt) + sigma**2*(dx-mu*dt)**2)  / (2*sigma**2*(1-rho**2)*dt)
                if bt/a > 0:
                    # Equation (33)
                    v[t+1] = np.sqrt(bt/a)
                else:
                    # If v(t+1) is still negative, take the previous value
                    v[t+1] = v[t]


            # Equation (15) and (16)
            bt = ((v[t+1]-alpha*dt)**2 - 2*rho*sigma*(v[t+1]-alpha*dt)*(dx-mu*dt) + sigma**2*(dx-mu*dt)**2)  / (2*sigma**2*(1-rho**2)*dt)
            x1 = ((2*betap+rho*sigma*dt)*(v[t+1]-alpha*dt) - (2*rho*sigma*betap+sigma**2*dt)*(dx-mu*dt))   / (2*sigma**2*(1-rho**2)*dt)
            x2 = -2*sqrt(a*bt)
            # Combined exponent for Equation (34)
            E = np.exp(x1 + x2) / D
            if method == 1:
                # Equation (34) for the likelihood L(t+1)
                L[t+1] = (a*bt)**(-1/4) * E * L[t]
            elif method == 2:
                # Alternatively, use the log-likelihood, log of Equation (34)
                L[t+1] = -1/4*np.log(a*bt) + x1 + x2 -np.log(D) + L[t]

        # Negative likelihood is the last term.
        # Since we maximize the likelihood, we minimize the negative likelihood.
        likelihood = -np.real(L[T-1])
        return likelihood/100000, v

    @staticmethod
    def calibrate_i(S, dt):
        bnds = ((0, 0.4),
                (0, 2.0),
                (0.05, 0.5),
                (0.005, 2.0),
                (-0.4, 0.4))
        x0 = [0.1, 0.05, 0.3, 0.1, 0.25]
        r = 0; q = 0.0

        method = 2  # Select method : 1 = Likelihood, 2 = Log-Likelihood.  Set the options.
        min_func = lambda param: (Heston.likelihoodAW(param, np.log(S), r, q, dt, method=method))[0]
        res = minimize(min_func, x0, method='SLSQP', bounds=bnds)
        param_list = res.x.tolist()
        likelihood_2, estimated_v = Heston.likelihoodAW(res.x, np.log(S), r, q, dt, method=method)
        params = {"kappa": param_list[0], "theta": param_list[1], "sigma":param_list[2],
                  "v0": param_list[3], "rho": param_list[4]}
        return params, estimated_v

    @staticmethod
    def calibrate(data_frame, dt):
        params_dict = {}
        for isin in data_frame.columns:
            S = data_frame[[isin]].values
            params, estimated_v = Heston.calibrate_i(S, dt)
            params_dict[isin] = params
        return params_dict









# Utility function to pull out spot and vol paths as Pandas dataframes
def _generate_multi_paths_df(spot, seq, num_paths):
    spot_paths = []
    vol_paths = []

    for i in range(num_paths):
        sample_path = seq.next()
        values = sample_path.value()

        spot, vol = values

        spot_paths.append([x for x in spot])
        vol_paths.append([x for x in vol])

    df_spot = pd.DataFrame(spot_paths, columns=[spot.time(x) for x in range(len(spot))])
    df_vol = pd.DataFrame(vol_paths, columns=[spot.time(x) for x in range(len(spot))])

    return df_spot, df_vol


def simulate_heston_dont_use_quantlib(today, timestep, length, N, spot, rate, v0, kappa, theta, sigma, rho):
    # Set up the flat risk-free curves
    riskFreeCurve = ql.FlatForward(today, rate, ql.Actual365Fixed())
    flat_ts = ql.YieldTermStructureHandle(riskFreeCurve)
    dividend_ts = ql.YieldTermStructureHandle(riskFreeCurve)
    heston_process = ql.HestonProcess(flat_ts, dividend_ts, ql.QuoteHandle(ql.SimpleQuote(spot)), v0, kappa, theta,
                                      sigma, rho)
    times = ql.TimeGrid(length, timestep)
    dimension = heston_process.factors()
    rng = ql.GaussianRandomSequenceGenerator(
        ql.UniformRandomSequenceGenerator(dimension * timestep, ql.UniformRandomGenerator()))
    seq = ql.GaussianMultiPathGenerator(heston_process, list(times), rng, False)
    df_spot, df_vol = _generate_multi_paths_df(spot, seq, N)
    return df_spot, df_vol

if __name__ == "__main__":
    ###########   SIMULATION    ##################   Parameters Values     ##############################################
    start_time = time.time()
    r = 0.05  # risk-free interest rate
    q = 0.0  # dividend
    maturity = 3  # longest maturity
    dt = 1 / 252  # size of time-step
    S_0 = 100  # initila asset price

    kappa = 0.2  # mean-reversion rate
    theta = 0.4  # long-run variance
    sigma = 0.25  # volatility of volatility
    v0 = 0.1  # initial variance
    rho = -0.25  # correlation of the bivariables
    param_original = [kappa, theta, sigma, v0, rho]
    numPaths = 1

    scheme = 'Milstein'
    negvar = 'Trunca'
    S, V, Vcount0 = Heston.simulate(scheme, negvar, numPaths, S_0, maturity, dt, r, q, kappa, theta, sigma, v0, rho)
    df_spot = pd.DataFrame(S).T
    df_vol = pd.DataFrame(V).T

    param_estimated, estimated_v = Heston.calibrate_i(S)

    df_estimates = pd.concat(
        [pd.DataFrame(param_estimated, columns=['AW estimate']),
         pd.DataFrame(param_original, columns=['Original values'])],
        axis=1)
    print(df_estimates)
    plt.plot(estimated_v)
    plt.plot(V)
    plt.show()
    print()
