import pandas as pd
import numpy as np


def calibrate_gbm(data_frame, delta):
    df_sigma = pd.DataFrame(np.log(data_frame).diff().std() / np.sqrt(delta)).rename(
        columns={0: 'sigma'})
    df_mu = ((np.log(data_frame).diff()).mean(axis=0).to_frame() * (1 / delta) + (df_sigma.values ** 2) / 2).rename(
        columns={0: 'mu'})

    df_params = pd.concat([df_mu, df_sigma], axis=1).T

    params = df_params.to_dict()
    print(params)
    return params
