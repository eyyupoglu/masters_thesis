from QuantLib import *



def option_heston(calculation_date, maturity_date, spot_price, strike_price, dividend_rate, option_type,
                  risk_free_rate, spot_variance, kappa, theta, sigma, rho):
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
    heston_process = HestonProcess(flat_ts, dividend_yield, spot_handle, spot_variance, kappa, theta, sigma, rho)
    engine = AnalyticHestonEngine(HestonModel(heston_process), 0.01, 1000)
    european_option.setPricingEngine(engine)
    h_price = european_option.NPV()
    # print("The Heston model price is", h_price)
    return h_price



