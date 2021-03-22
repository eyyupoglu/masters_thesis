from calibration.gbm import calibrate_gbm
from diffusion.gbm_process import simulate_gbm
from backtesting.backtesting import backtest_gbm_i

import numpy as np
import pandas as pd
import QuantLib as ql




def generate_critical_values_gbm(df, equity, dict_bt, dict_cal, scenario_amount=200):
    quantile_list = []
    params = calibrate_gbm(df[[equity]], delta=1 / 52)
    for el in range(scenario_amount):
        today = ql.Date(1, 7, 2020)
        initialValue = df[equity][0]

        df_dates = pd.date_range(start=df.index[0], end=df.index[-1], freq='D').to_frame()
        date_grid = [today + ql.Period(i, ql.Days) for i in range(0, len(df_dates))]
        nPaths = 1

        df_bootstrapped = simulate_gbm(initialValue, date_grid, nPaths, params[equity])
        df_bootstrapped.index = pd.to_datetime([el.to_date() for el in df_bootstrapped.index])
        df_bootstrapped = df_bootstrapped[df_bootstrapped.index.weekday == 2]
        df_dates_w = df_dates[df_dates.index.weekday == 2]
        df_bootstrapped.index = df_dates_w.index
        df_bootstrapped = df_bootstrapped.rename(columns={0: equity})

        p_list = backtest_gbm_i(df_bootstrapped,
                                dict_bt['backtesting_horizon_w'],
                                dict_bt['backtesting_window_w'],
                                dict_bt['backtesting_frequency_w'],
                                dict_bt['calibration_freq_w'],
                                equity,
                                dict_cal['general_settings']['data_length_y'],
                                N=dict_bt['simulation_amount'],
                                bt_beginning_date=ql.Date(7, 1, 2015))
        quantile_list.append(np.sort(p_list))
        print(el)
    return pd.DataFrame(np.array(quantile_list))

