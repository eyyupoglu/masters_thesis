import json
import numpy as np
import pandas as pd
import QuantLib as ql
from os.path import join
import datetime
import matplotlib.pyplot as plt
from docutils.nodes import legend
from pandas import ExcelWriter
import seaborn as sns
sns.set()

from calibration.gbm import calibrate_gbm
from diffusion.gbm_process import simulate_gbm, simulated_gbm_pointwise


def backtest_gbm_i(df_x, horizon_w, backtesting_window_w, backtesting_frequency_w,
                   calibration_freq_w, equity_name, calibration_years,
                   bt_beginning_date = ql.Date(7, 1, 2015), N=1000):
    df_calibration = df_x[(bt_beginning_date - ql.Period(calibration_years, ql.Years)).to_date():
                          bt_beginning_date.to_date()]
    params = calibrate_gbm(df_calibration, delta=1 / 52)
    p_list = []
    for i in range(backtesting_window_w):
        df_calibration = df_x[(bt_beginning_date - ql.Period(calibration_years, ql.Years)).to_date():
                              bt_beginning_date.to_date()]
        if i % calibration_freq_w == 0:
            params = calibrate_gbm(df_calibration, delta=1 / 52)

        initial_value = df_x.ix[bt_beginning_date.to_date()].to_frame().T
        simulations = simulated_gbm_pointwise(initial_value[equity_name].values[0], params[equity_name], horizon_w, N)
        # df_simulations.index = [el.to_date() for el in df_simulations.index]

        try:
            realisation_date = pd.Timestamp((df_x.ix[bt_beginning_date.to_date()].to_frame().T.index +
                                             datetime.timedelta(7*horizon_w))[0])
            realisation = df_x.xs(realisation_date)[equity_name]
        except:
            print('Data point does not exists for  %s ' % realisation_date)
            continue

        p = len(simulations[simulations[horizon_w] > realisation]) / len(simulations)
        p_list.append(p)

        bt_beginning_date = bt_beginning_date + ql.Period(backtesting_frequency_w, ql.Weeks)
        # print(bt_beginning_date)

    return p_list


def backtest_gbm(calibration_config_path, backtest_config_path, visualize=True):
    with open(backtest_config_path) as f:
        dict_bt = json.load(f)
    with open(calibration_config_path) as f:
        dict_cal = json.load(f)

    df = pd.read_csv(dict_cal['general_settings']['input_data_path'])
    df = df.pivot_table(index='EOD_DATE', values='PRICE', columns=['ISIN'])
    df.index = pd.to_datetime(df.index)
    df = df[[key for key in df.isna().sum().to_frame().sort_values(0).to_dict()[0]]] # sort the least nans
    df = df[dict_cal['risk_factors']['equity']['scope']].dropna() # filter the equity scope
    idx = pd.date_range(df.index[0], df.index[-1], freq='D')
    df = df.reindex(idx, fill_value=np.nan)
    df = df[df.index.weekday == 2].interpolate(method='linear') #add the holiday wednesdays to not to fail in BT

    rgb_dict = {}
    for horizon in dict_bt['backtesting_horizon_w']:
        rgb_dict[horizon] = {}
        for equity in dict_cal['risk_factors']['equity']['scope']:
            p_list = backtest_gbm_i(df, horizon,
                                       dict_bt['backtesting_window_w'],
                                       dict_bt['backtesting_frequency_w'],
                                       dict_bt['calibration_freq_w'],
                                       equity,
                                        dict_cal['general_settings']['data_length_y'],
                                       N=dict_bt['simulation_amount'],
                                       bt_beginning_date=ql.Date(7, 1, 2015))
            df_critical_paths = pd.read_csv(join(dict_bt['back_testing_critical_values_path'],
                                            ('critical_values_horizon_%s.csv' % str(horizon))),
                                            index_col=False)
            uniform = np.arange(0, len(df_critical_paths.T), 1) / len(df_critical_paths.T)
            ks_metrics_dist = np.max(np.abs(np.sort(df_critical_paths) - uniform), axis=0)
            ks_metric = np.max(np.abs(np.sort(p_list) - uniform[:len(p_list)]))

            rgb_dict[horizon][equity] = {}
            if ks_metric > np.quantile(ks_metrics_dist, 0.99):
                rgb_dict[horizon][equity] = 'RED'
            elif ks_metric <= np.quantile(ks_metrics_dist, 0.99) and ks_metric >= np.quantile(ks_metrics_dist, 0.95):
                rgb_dict[horizon][equity] = 'AMBER'
            elif ks_metric <= np.quantile(ks_metrics_dist, 0.95):
                rgb_dict[horizon][equity] = 'GREEN'
            if visualize is True:
                df_critical_paths.T.plot(color='k', legend=False)
                plt.plot(np.sort(p_list), color='red')
                plt.title(equity)
                plt.show()

                plt.hist(ks_metrics_dist, color='k')
                plt.axvline(x=ks_metric, color='r')
                plt.title(equity)
                plt.show()

                fig, axes = plt.subplots(nrows=1, ncols=2)
                df_critical_paths.T.plot(color='k', legend=False).plot(ax=axes[0])
                pd.DataFrame(ks_metrics_dist).hist(ax=axes[1])
                plt.show()

                plt.figure(1)
                plt.subplot(2, 1, 1)
                df_critical_paths.T.plot(color='k', legend=False).plot()  # no need to specify for first axis
                plt.subplot(2, 1, 2)
                pd.DataFrame(ks_metrics_dist).hist(ax=plt.gca())


    xls_path = join(dict_bt['back_testing_output_path'], 'rgb_results_%s.xlsx' % dict_bt['backtesting_horizon_w'])
    with ExcelWriter(xls_path) as writer:
        for n, df in enumerate([pd.DataFrame(rgb_dict), pd.DataFrame(dict_bt).T]):
            df.to_excel(writer,'sheet%s' % n)






if __name__ == "__main__":
    calibration_config_path = 'C:/Users/eyyup/Desktop/packages/masters_thesis/static/settings/calibration_settings.json'
    backtest_config_path = 'C:/Users/eyyup/Desktop/packages/masters_thesis/static/settings/backtesting_configuration.json'

    backtest_gbm(calibration_config_path, backtest_config_path, visualize=True)
