from QuantLib import *



def option_european_bsm(calculation_date, maturity_date, spot_price, strike_price, dividend_rate, option_type,
                        risk_free_rate, volatility):
    calendar = UnitedStates()
    day_count = Actual365Fixed()
    Settings.instance().evaluationDate = calculation_date
    # construct the European Option
    payoff = PlainVanillaPayoff(option_type, strike_price)
    exercise = EuropeanExercise(maturity_date)
    european_option = VanillaOption(payoff, exercise)

    spot_handle = QuoteHandle(SimpleQuote(spot_price))
    flat_ts = YieldTermStructureHandle(FlatForward(calculation_date, risk_free_rate, day_count))
    dividend_yield = YieldTermStructureHandle(FlatForward(calculation_date, dividend_rate, day_count))
    flat_vol_ts = BlackVolTermStructureHandle(BlackConstantVol(calculation_date, calendar, volatility, day_count))
    bsm_process = BlackScholesMertonProcess(spot_handle, dividend_yield, flat_ts, flat_vol_ts)
    european_option.setPricingEngine(AnalyticEuropeanEngine(bsm_process))
    bs_price = european_option.NPV()
    # print("The Black-Scholes-Merton model price is ", bs_price)
    return bs_price



