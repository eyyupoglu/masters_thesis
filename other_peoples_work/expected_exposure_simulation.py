# import the used libraries
import numpy as np
import matplotlib.pyplot as plt
import QuantLib as ql



# Setting evaluation date
today = ql.Date(7,4,2015)
ql.Settings.instance().setEvaluationDate(today)


# Setup Marketdata
rate = ql.SimpleQuote(0.03)
rate_handle = ql.QuoteHandle(rate)
dc = ql.Actual365Fixed()
yts = ql.FlatForward(today, rate_handle, dc)
yts.enableExtrapolation()
hyts = ql.RelinkableYieldTermStructureHandle(yts)
t0_curve = ql.YieldTermStructureHandle(yts)
euribor6m = ql.Euribor6M(hyts)


# Setup a dummy portfolio with two Swaps
def makeSwap(start, maturity, nominal, fixedRate, index, typ=ql.VanillaSwap.Payer):
    """
    creates a plain vanilla swap with fixedLegTenor 1Y

    parameter:

        start (ql.Date) : Start Date

        maturity (ql.Period) : SwapTenor

        nominal (float) : Nominal

        fixedRate (float) : rate paid on fixed leg

        index (ql.IborIndex) : Index

    return: tuple(ql.Swap, list<Dates>) Swap and all fixing dates


    """
    end = ql.TARGET().advance(start, maturity)
    fixedLegTenor = ql.Period("1y")
    fixedLegBDC = ql.ModifiedFollowing
    fixedLegDC = ql.Thirty360(ql.Thirty360.BondBasis)
    spread = 0.0
    fixedSchedule = ql.Schedule(start,
                                end,
                                fixedLegTenor,
                                index.fixingCalendar(),
                                fixedLegBDC,
                                fixedLegBDC,
                                ql.DateGeneration.Backward,
                                False)
    floatSchedule = ql.Schedule(start,
                                end,
                                index.tenor(),
                                index.fixingCalendar(),
                                index.businessDayConvention(),
                                index.businessDayConvention(),
                                ql.DateGeneration.Backward,
                                False)
    swap = ql.VanillaSwap(typ,
                          nominal,
                          fixedSchedule,
                          fixedRate,
                          fixedLegDC,
                          floatSchedule,
                          index,
                          spread,
                          index.dayCounter())
    return swap, [index.fixingDate(x) for x in floatSchedule][:-1]


portfolio = [makeSwap(today + ql.Period("2d"),
                      ql.Period("5Y"),
                      1e6,
                      0.03,
                      euribor6m),
             makeSwap(today + ql.Period("2d"),
                      ql.Period("4Y"),
                      5e5,
                      0.03,
                      euribor6m,
                      ql.VanillaSwap.Receiver),
             ]

#%%timeit
# Setup pricing engine and calculate the npv
engine = ql.DiscountingSwapEngine(hyts)
for deal, fixingDates in portfolio:
    deal.setPricingEngine(engine)
    deal.NPV()
    #print(deal.NPV())


# Assume the model is already calibrated either historical or market implied
volas = [ql.QuoteHandle(ql.SimpleQuote(0.0075)),
         ql.QuoteHandle(ql.SimpleQuote(0.0075))]
meanRev = [ql.QuoteHandle(ql.SimpleQuote(0.02))]
model = ql.Gsr(t0_curve, [today+100], volas, meanRev, 16.)
process = model.stateProcess()


# Define evaluation grid
date_grid = [today + ql.Period(i,ql.Weeks) for i in range(0,52*6)]
for deal in portfolio:
    date_grid += deal[1]

date_grid = np.unique(np.sort(date_grid))
time_grid = np.vectorize(lambda x: ql.ActualActual().yearFraction(today, x))(date_grid)
dt = time_grid[1:] - time_grid[:-1]

print(len(time_grid)*1500*2*29e-6)


# Random number generator
seed = 1
urng = ql.MersenneTwisterUniformRng(seed)
usrg = ql.MersenneTwisterUniformRsg(len(time_grid)-1,urng)
generator = ql.InvCumulativeMersenneTwisterGaussianRsg(usrg)

# %%timeit
# Generate N paths
N = 1500
x = np.zeros((N, len(time_grid)))
y = np.zeros((N, len(time_grid)))
pillars = np.array([0.0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
zero_bonds = np.zeros((N, len(time_grid), 12))
for j in range(12):
    zero_bonds[:, 0, j] = model.zerobond(pillars[j],
                                         0,
                                         0)
for n in range(0, N):
    dWs = generator.nextSequence().value()
    for i in range(1, len(time_grid)):
        t0 = time_grid[i - 1]
        t1 = time_grid[i]
        x[n, i] = process.expectation(t0,
                                      x[n, i - 1],
                                      dt[i - 1]) + dWs[i - 1] * process.stdDeviation(t0,
                                                                                     x[n, i - 1],
                                                                                     dt[i - 1])
        y[n, i] = (x[n, i] - process.expectation(0, 0, t1)) / process.stdDeviation(0, 0, t1)
        for j in range(12):
            zero_bonds[n, i, j] = model.zerobond(t1 + pillars[j],
                                                 t1,
                                                 y[n, i])


# plot the paths
for i in range(0,N):
    plt.plot(time_grid, x[i,:])

# plt.show()
plt.savefig('simulation_rf.png')


#%%timeit
#Swap pricing under each scenario
npv_cube = np.zeros((N,len(date_grid), len(portfolio)))
for p in range(0,N):
    for t in range(0, len(date_grid)):
        date = date_grid[t]
        ql.Settings.instance().setEvaluationDate(date)
        ycDates = [date,
                   date + ql.Period(6, ql.Months)]
        ycDates += [date + ql.Period(i,ql.Years) for i in range(1,11)]
        yc = ql.DiscountCurve(ycDates,
                              zero_bonds[p, t, :],
                              ql.Actual365Fixed())
        yc.enableExtrapolation()
        hyts.linkTo(yc)
        if euribor6m.isValidFixingDate(date):
            fixing = euribor6m.fixing(date)
            euribor6m.addFixing(date, fixing)
        for i in range(len(portfolio)):
            npv_cube[p, t, i] = portfolio[i][0].NPV()
    ql.IndexManager.instance().clearHistories()
ql.Settings.instance().setEvaluationDate(today)
hyts.linkTo(yts)

# Calculate the portfolio npv by netting all NPV
portfolio_npv = np.sum(npv_cube,axis=2)


# Plot the first 30 NPV paths
plt.figure(figsize=(7,5), dpi=300)
for i in range(0,30):
    plt.plot(time_grid, portfolio_npv[i,:])
plt.xlabel("Time in years")
plt.ylabel("NPV")
plt.title("Simulated npv paths")
# plt.show()
plt.savefig('NPVs.png')





