from diffusion.heston_process import simulate_heston_dont_use_quantlib
from pricing.option_heston import option_heston
from pricing.option_bsm import option_european_bsm
import QuantLib as ql
import matplotlib.pyplot as plt
import numpy as np


df_spot, df_vol = simulate_heston_dont_use_quantlib(today=ql.Date(1, 7, 2020), timestep=100, length=2, spot=100, rate=0.0,
                                                    v0=0.01, kappa=1.0, theta=0.04, rho=-0.3, sigma=0.4)

option_heston(calculation_date=ql.Date(8, 5, 2015), maturity_date=ql.Date(15, 1, 2016), spot_price=127.62,
              strike_price=130, dividend_rate=0.0163, option_type=ql.Option.Call,
              risk_free_rate=0.001, spot_variance=0.2*0.2, kappa=0.1, theta=0.04, sigma=0.1, rho=-0.75)

option_european_bsm(calculation_date=ql.Date(8, 5, 2015), maturity_date=ql.Date(15, 1, 2016), spot_price=127.62,
                    strike_price=130, dividend_rate=0.0163, option_type=ql.Option.Call,
                    risk_free_rate=0.001, volatility = 0.20)





