import QuantLib as ql
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



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


def simulate_heston(today, timestep, length, N, spot, rate, v0, kappa, theta, sigma, rho):
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

