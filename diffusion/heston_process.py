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
        # implementation of Atiya-wall article, taken from the book called "Heston model and its extensions in matlab"
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

    @staticmethod
    def ekf_step_i(para,z, choice, v0,var_s,var_v,P0,kappa_set,kappa_theta_set,sigma_set,ro_set,mu_set,mu,delta):
        # implementation of article "Parameter Estimates of Heston Stochastic Volatility
        # Model with MLE and Consistent EKF Algorithm" - table 1
        P_design_old = 2

        kappa = para[0]
        ka_the = para[1]
        sigma = para[2]
        ro = para[3]

        kappa_upper = max(abs(kappa_set))
        kappa_theta_upper = max(abs(kappa_theta_set))
        ro_upper = max(abs(ro_set))
        sigma_upper = max(abs(sigma_set))
        mu_upper = max(abs(mu_set))

        kappa_lower = min(abs(kappa_set))
        kappa_theta_lower = min(abs(kappa_theta_set))
        ro_lower = min(abs(ro_set))
        sigma_lower = min(abs(sigma_set))
        mu_lower = min(abs(mu_set))
        one_min_ro2_max = max(1 - ro_set ** 2)

        Q = np.diag([var_s, var_v]) ** 2


        F = 1 - kappa * delta
        L = np.array([0, sigma * sqrt(max(v0 * delta, 0))])

        if choice == 1:

            P_bar = P0 * (1 - kappa_lower * delta) ** 2 + delta ** 2 * kappa_theta_upper ** 2 + sigma_upper ** 2 * max(
                v0 * delta, 0) * Q[1, 1]
            v_bar = max(v0 + ka_the * delta - kappa * v0 * delta, 0)
            H = -0.5 * delta
            M = np.array([sqrt(max((1 - ro ** 2) * v_bar * delta, 0)), ro * sqrt(max(v_bar * delta, 0))])

            K = (P_bar * H + L @  M.T) * (H * P_bar * H + M @ M.T + H*L @ M.T + M @ L.T * H)**(-1)
            v = v_bar + K * (z - (mu-0.5 * v_bar) * delta)

            P0 = (1 + K * 0.5 * delta) ** 2 * P_bar + K ** 2 * delta ** 2 * (mu_upper - mu_lower) ** 2 + \
                 2 * K ** 2 * max(v_bar * delta, 0) * ((1 - ro ** 2) * Q[0, 0] + ro ** 2 * Q[1,1])
        else:
            P_bar = F * P0 * F + L @ Q @ L.T

            v_bar = max(v0 + ka_the * delta - kappa * v0 * delta, 0)
            H = -0.5 * delta
            M = np.array([sqrt((1 - ro ** 2) * v_bar * delta), ro * sqrt(v_bar * delta)])
            K = (P_bar * H + L @ M.T) * (H * P_bar * H + M @ Q @ M.T + H * L @ Q @ M.T + M @ Q @ L.T * H)**(-1)
            v = v_bar+K * (z- (mu-0.5 * v_bar) * delta)
            P0 = P_bar - K * (H * P_bar + M @ Q @ L.T)


        output_v = max(v, 0.00001)

        return output_v, P0

    @staticmethod
    def estimate_heston_given_volatilities(vols, spots, dt):
        # implementation of article "Parameter Estimates of Heston Stochastic Volatility
        # Model with MLE and Consistent EKF Algorithm"- equation 10
        n = len(vols)
        dt = dt
        P_numerator = ((vols.shift(1) * vols) ** (1 / 2)).sum() / n - 1 / (n ** 2) * (
                (vols / vols.shift(1)) ** (1 / 2)).sum() * vols.shift(1).sum()
        P_denumerator = dt / 2 - dt / 2 * (1 / n ** 2) * (1 / vols.shift(1)).sum() * vols.shift(1).sum()
        P = P_numerator / P_denumerator

        kappa_est = 2 / dt * (
                1 + P * dt / 2 * 1 / n * (1 / vols.shift(1)).sum() - 1 / n * ((vols / vols.shift(1)) ** (1 / 2)).sum())
        sigma_est = (4 / dt / n * ((vols ** (1 / 2) - vols.shift(1) ** (1 / 2) - dt / (2 * vols.shift(1) ** (1 / 2)) * (
                P - kappa_est * vols.shift())) ** 2).sum()) ** (1 / 2)
        theta_est = (P + 1 / 4 * sigma_est ** 2) / kappa_est

        sigma_gbm_est = np.log(spots).diff().std() / np.sqrt(dt)
        r = ((np.log(spots).diff()).mean(axis=0) * (1 / dt) + (sigma_gbm_est ** 2) / 2)

        dW1 = (np.log(spots) - np.log(spots.shift(1)) - (r - 1 / 2 * vols.shift(1)) * dt) / vols.shift(1) ** (1 / 2)
        dW2 = (vols - vols.shift(1) - kappa_est * (theta_est - vols.shift(1)) * dt) / (
                    sigma_est * vols.shift(1) ** (1 / 2))
        ro = 1 / n / dt * (dW1 * dW2).sum()

        return r, kappa_est, sigma_est, theta_est, ro

    @staticmethod
    def calibrate_ekf(S):
        delta = 1 / 252
        Y = np.log(S)
        z_m = Y[1:] - Y[:-1]

        mu = 0.05
        r = mu
        V10 = 0.01

        var_s = 1
        var_v = 1
        P0 = 0.5

        kappa_ini = 1
        theta_ini = 0.03
        kappa_theta_ini = kappa_ini * theta_ini
        sigma_ini = 0.2

        kappa_lb = 0.1
        kappa_ub = 2

        kappa_theta_lb = 0.002
        kappa_theta_ub = 0.7
        sigma_lb = 0.1
        sigma_ub = 0.6
        ro_lb = -0.6
        ro_ub = 0.6
        kappa_theta_set = np.array([kappa_theta_lb, kappa_theta_ub])
        kappa_set = np.array([kappa_lb, kappa_ub])
        sigma_set = np.array([sigma_lb, sigma_ub])
        ro_set = np.array([ro_lb, ro_ub])
        mu_set = np.array([mu, mu])
        step_set = 20

        choice1 = 1
        v = V10

        lb = np.array([kappa_lb, kappa_theta_lb, sigma_lb, ro_lb])
        up = np.array([kappa_ub, kappa_theta_ub, sigma_ub, ro_ub])


        ro_ini = 0.2
        parax_final1 = np.zeros((len(z_m), 4))

        V_k1 = np.zeros(len(z_m))
        P_k1 = np.zeros(len(z_m))
        V_k2 = np.zeros(len(z_m))
        P_k2 = np.zeros(len(z_m))

        parax_final1[0, :] = np.array([kappa_ini, kappa_theta_ini, sigma_ini, ro_ini])

        V_k1[0] = V10
        P_k1[0] = P0
        V_k2[0] = V10
        P_k2[0] = P0


        for k in range(1, len(z_m)):
            V_k1[k], P_k1[k] = Heston.ekf_step_i(parax_final1[k-1, :], z_m[k], choice1, V_k1[k - 1], var_s, var_v,
                                                                         P_k1[k - 1], kappa_set, kappa_theta_set,
                                                                         sigma_set, ro_set, mu_set, mu, delta)
            V_k1_ba = np.array([*V_k1[:k-1].tolist(), V_k1[k]])
            if V_k1[k] > 0.2:
                V_k1_ba = np.array([*V_k1[:k - 1].tolist(), 0.1])
            [kappa_ba_new1, theta_ba_new1, sigma_ba_new1] = Heston.SQRT_CIR_esti(V_k1_ba, delta)

            parax_final1[k, 0] = kappa_ba_new1
            parax_final1[k, 1] = kappa_ba_new1 * theta_ba_new1
            parax_final1[k, 2] = sigma_ba_new1
            parax_final1[k, 3] = ro_ini

        plt.plot(parax_final1[:, 0])
        plt.show()

        plt.plot(parax_final1[:, 1] / parax_final1[:, 0])
        plt.show()

        plt.plot(parax_final1[:, 2])
        plt.show()

        plt.plot(parax_final1[:, 3])
        plt.show()

        plt.plot(V_k1)
        plt.show()

        plt.plot(P_k1)
        plt.show()
        return parax_final1[-1, :]

    @staticmethod
    def SQRT_CIR_esti(vReal, dt):
        d1 = 0
        d2 = 0
        d3 = 0
        d4 = 0
        n = len(vReal)
        for i in range(1, n):
            d1 = d1 + sqrt(vReal[i - 1] * vReal[i])
            d2 = d2 + sqrt(vReal[i] / vReal[i - 1])
            d3 = d3 + vReal[i - 1]
            d4 = d4 + (1 / vReal[i - 1])

        P = (d1 - (1 / (n)) * d2 * d3) / (dt * (n) * 0.5 - dt / (2 * (n)) * d4 * d3)
        kappa = (1 + P * dt / (2 * (n)) * d4 - d2 / (n)) * 2 / dt
        d5 = 0
        for j in range(1, n):
            d5 = d5 + (sqrt(vReal[j]) - sqrt(vReal[j - 1]) - dt * (P - kappa * vReal[j - 1]) / (2 * sqrt(vReal[j - 1]))) ** 2

        sigma2 = 4 * d5 / ((n) * dt)
        sigma = sqrt(sigma2)
        theta = (P + 1 / 4 * sigma2) / kappa
        return kappa, theta, sigma










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
    r = 0  # risk-free interest rate
    q = 0.0  # dividend
    maturity = 3  # longest maturity
    dt = 1 / 52  # size of time-step
    S_0 = 100  # initila asset price

    kappa = 2  # mean-reversion rate
    theta = 0.1  # long-run variance
    sigma = 0.25  # volatility of volatility
    v0 = 0.1  # initial variance
    rho = -0.25  # correlation of the bivariables
    param_original = [kappa, theta, sigma, v0, rho]
    numPaths = 1

    scheme = 'Milstein'
    negvar = 'Trunca'
    S, V, Vcount0 = Heston.simulate(scheme, negvar, numPaths, S_0, maturity, dt, r, q, kappa, theta, sigma, v0, rho)

    # plt.plot(S)
    # plt.show()
    #
    # plt.plot(V)
    # plt.show()
    params_ekf = Heston.calibrate_ekf(S)

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
