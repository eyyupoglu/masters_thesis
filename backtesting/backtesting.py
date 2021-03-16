import numpy as np
import pandas as pd
import QuantLib as ql
from os.path import join
import matplotlib.pyplot as plt

from calibration.gbm import calibrate_gbm
from diffusion.gbm_process import simulate_gbm


def backtest_gbm(df_x, horizon_w, backtesting_window_w, calibration_freq_w, equity_name,
                 bt_beginning_date = ql.Date(7, 1, 2015), N=1000):
    df_calibration = df_x[(bt_beginning_date - ql.Period(3, ql.Years)).to_date():
                          bt_beginning_date.to_date()]
    params = calibrate_gbm(df_calibration, delta=1 / 52)
    p_list = []
    for i in range(backtesting_window_w):
        df_calibration = df_x[(bt_beginning_date - ql.Period(3, ql.Years)).to_date():
                              bt_beginning_date.to_date()]
        if i % calibration_freq_w == 0:
            params = calibrate_gbm(df_calibration, delta=1 / 52)

        initial_value = df_x.ix[bt_beginning_date.to_date()].to_frame().T

        date_grid = [bt_beginning_date + ql.Period(i, ql.Days) for i in range(0, horizon_w * 7 + 5)]
        date_grid_sim = [bt_beginning_date + ql.Period(i, ql.Weeks) for i in range(0, horizon_w)]
        df_simulations = simulate_gbm(initial_value[equity_name].values[0], date_grid, N, params[equity_name])
        df_simulations.index = [el.to_date() for el in df_simulations.index]

        d = date_grid_sim[-1].to_date()

        try:
            realisation = df_x.xs(d)[equity_name]
        except:
            print('%s window came to the end of the data, no more realisation' % d)
        simulations = df_simulations.xs(d)

        p = len(simulations[simulations > realisation]) / len(simulations)
        p_list.append(p)

        bt_beginning_date = bt_beginning_date + ql.Period(1, ql.Weeks)
        print(bt_beginning_date)

    return p_list


df = pd.read_csv(join('C:/Users/eyyup/Desktop/packages/masters_thesis/static/data/', 'data_from_mars.csv'))
df = df.pivot_table(index = 'EOD_DATE', values='PRICE', columns=['NAME'])
df.index = pd.to_datetime(df.index)


df = df[[key for key in df.isna().sum().to_frame().sort_values(0).to_dict()[0]]]
df = df.iloc[:, :5].dropna()
df.plot(figsize=(16, 4))
plt.show()


df_x = df
idx = pd.date_range('2010-02-03', '2021-03-10', freq='D')
df_x = df_x.reindex(idx, fill_value=np.nan)
df_x = df_x[df_x.index.weekday == 2].interpolate(method='linear')

bt_beginning = ql.Date(7, 1, 2015)

horizon_w = 2
calibration_freq_w = 12
backtesting_window_w = 6 * 52
equity_name = 'DAIMLER AG'




p_list = backtest_gbm(df_x, horizon_w, backtesting_window_w, calibration_freq_w,
                      equity_name, bt_beginning_date=bt_beginning)

plt.plot(p_list)
plt.show()
plt.plot(np.arange(0, np.sort(p_list).shape[0]) / np.sort(p_list).shape[0], np.sort(p_list))
plt.show()

