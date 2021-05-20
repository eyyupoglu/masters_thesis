import QuantLib as ql
import numpy as np
import pandas as pd


class GBM(object):
    def __init__(self):
        pass

    @staticmethod
    def simulate_gbm_pointwise(initial_value, params, horizon, N):
        rand = np.random.standard_normal(N)
        mu = params['mu']
        sigma = params['sigma']
        dt = 1 / 52 * horizon
        simulated = initial_value * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * rand)
        return pd.DataFrame(data=simulated, columns=[horizon])

    @staticmethod
    def calibrate_gbm(data_frame, dt):
        df_sigma = pd.DataFrame(np.log(data_frame).diff().std() / np.sqrt(dt)).rename(
            columns={0: 'sigma'})
        df_mu = ((np.log(data_frame).diff()).mean(axis=0).to_frame() * (1 / dt) + (df_sigma.values ** 2) / 2).rename(
            columns={0: 'mu'})

        df_params = pd.concat([df_mu, df_sigma], axis=1).T

        params = df_params.to_dict()
        # print(params)
        return params


# process = QuantLib 1-dimensional stochastic process object
def simulate_gbm(initial_value, date_grid, n_paths, params):
    mu, sigma = params['mu'], params['sigma']
    process = ql.GeometricBrownianMotionProcess(initial_value, mu, sigma)
    nSteps = int((date_grid[-1] - date_grid[0]))
    maturity = (date_grid[-1] - date_grid[0]) / 365

    generator = ql.UniformRandomGenerator()
    sequenceGenerator = ql.UniformRandomSequenceGenerator(nSteps, generator)
    gaussianSequenceGenerator = ql.GaussianRandomSequenceGenerator(sequenceGenerator)
    paths = np.zeros(shape=(n_paths, nSteps + 1))
    pathGenerator = ql.GaussianPathGenerator(process, maturity, nSteps, gaussianSequenceGenerator, False)
    for i in range(n_paths):
        path = pathGenerator.next().value()
        paths[i, :] = np.array([path[j] for j in range(nSteps + 1)])
    df = pd.DataFrame(paths.T)
    df.index = date_grid
    return df


