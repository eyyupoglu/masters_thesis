import QuantLib as ql
import numpy as np
import pandas as pd


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

