import numpy as np
import scipy.stats as ss
import pandas as pd
import matplotlib.pyplot as plt
from numpy import exp, log, sqrt, linspace, sort, floor, mean, std
from numpy.random import rand, randn, RandomState

class JumpDiffusion:

    @staticmethod
    def simulate(numPaths, Szero, dt, Years,  mu, vol, lambda_, q1, q2):
        num_time = int(Years / dt)
        S = np.zeros((num_time, numPaths))
        JumpDist = np.zeros((num_time, numPaths))
        JumpDistDiff = np.zeros((num_time, numPaths))
        S[0, :] = Szero
        JumpDist[0, :] = 0
        JumpDistDiff[0, :] = Szero

        SQRTdt = sqrt(dt)
        musig2 = mu - 0.5 * vol ** 2
        nuMean = (exp(q2) - exp(q1)) / (q2 - q1) - 1
        lambdadt = lambda_ *dt
        TimeLength = dt * num_time
        time = linspace(0, TimeLength, num_time)

        for i in range(numPaths):
            UniDist = rand(num_time)
            for t_step in range(1, num_time):
                if (lambdadt > UniDist[t_step]):
                    Q = q1 + (q2 - q1) * rand(1)
                    S[t_step, i] = S[t_step - 1, i] * exp(musig2 * dt + vol * randn(1) * SQRTdt + Q)

                    JumpDistDiff[t_step, i] = JumpDistDiff[t_step - 1, i] * exp(musig2 * dt + vol * randn(1) * SQRTdt + Q)
                    JumpDist[t_step, i] = JumpDistDiff[t_step - 1, i] * (exp(Q) - 1)
                else:
                    S[t_step, i] = S[t_step - 1, i] * exp(musig2 * dt + vol * randn(1) * SQRTdt)
                    JumpDistDiff[t_step, i] = JumpDistDiff[t_step - 1, i] * exp(musig2 * dt + vol * randn(1) * SQRTdt)
                    JumpDist[t_step, i] = 0



        # plt.plot(S[:, 2])
        # plt.plot(JumpDist[:, 2])
        # plt.show()
        print()

        return S, JumpDist

    @staticmethod
    def calibrate_i(S, dt):
        SQRTdt=sqrt(dt)
        LnS = log(S)
        Years = len(S) * dt

        LogDelta = log(S[1:])-log(S[:-1])
        lengthLD = len(LogDelta)
        M1 = mean(LogDelta)
        StanDev = std(LogDelta)
        M2 = StanDev ** 2
        M3 = mean((LogDelta - M1)** 3)
        M4 = mean((LogDelta - M1)** 4)
        Skew = M3 / (M2 ** 1.5)
        Kurtosis = M4 / (M2 ** 2) - 3

        xmin = min(LogDelta)
        estQ1 = xmin
        xmax = max(LogDelta)
        estQ2 = xmax
        sorted = sort(LogDelta.squeeze())
        q25 = sorted[int(floor(0.25 * lengthLD))]
        q75 = sorted[int((floor(0.75 * lengthLD)))]
        estMuJump = (estQ1 + estQ2) / 2
        estNuMean = (exp(estQ2) - exp(estQ1)) / (estQ2 - estQ1) - 1


        # Count returns +/- 3 standard deviations as number of jumps and

        # divide by Years of data to estimate lambda
        # recalculate diffusion volatility without outliers
        # Modification of approach presented in L. Clewlow, C. Strickland,
        # V. Kaminski, "Extending Mean-Reversion Jump Diffusion"

        outliersBottom=0
        outliersTop=0
        neg3sd=M1-3*StanDev
        pos3sd=M1+3*StanDev
        bottom = 1
        while (sorted[bottom]<neg3sd):
            outliersBottom=outliersBottom+1
            bottom=bottom+1

        top = lengthLD - 1
        while (sorted[top]>pos3sd):
            outliersTop=outliersTop+1
            top=top-1

        StanDev = std(LogDelta[bottom:top])
        estVol=StanDev/sqrt(dt) # Estimated annualized volatility

        estLambda=(outliersTop+outliersBottom)/Years
        estMuDsig2=(M1-estMuJump*estLambda*dt)/dt
        estMuD=estMuDsig2+0.5*estVol**2

        return {'estMuD': estMuD, 'estVol': estVol, 'estLambda': estLambda, 'estQ1': estQ1, 'estQ2': estQ2}

    @staticmethod
    def calibrate(data_frame, dt):
        params_dict = {}
        for isin in data_frame.columns:
            S = data_frame[[isin]].values
            params = JumpDiffusion.calibrate_i(S.squeeze(), dt)
            params_dict[isin] = params
        return params_dict



def simulate_gbmj(S, maturity, steps, lambda_, sigma_y_, sigma_, mu_y_, mu_star):
    dt = maturity/steps
    M_simul = np.zeros(steps)
    jumps = np.zeros(steps)
    M_simul[0] = S
    for i in np.arange(1, steps):
        jumpnb = np.random.poisson( lambda_*dt, size=1)
        jump = np.random.normal(mu_y_ * ( jumpnb - lambda_ * dt ), sqrt( jumpnb ) * sigma_y_, size=1)
        jumps[i] = jump
        M_simul[i] = M_simul[i-1] * np.exp(mu_star * dt + sigma_ * sqrt( dt )* np.random.normal(0, 1, 1) + jump)
    return jumps, M_simul


if __name__ == "__main__":
    numPaths = 100
    Years = 3
    dt = 1 / 252
    Szero = 50
    mu = 0.11
    vol = 0.25
    # musig2 = mu - 0.5 * vol ** 2
    lambda_ = 5
    q1 = -0.14
    q2 = 0.15
    S, J = JumpDiffusion.simulate(numPaths, Szero, dt, Years, mu, vol, lambda_, q1, q2)
    params = JumpDiffusion.calibrate(pd.DataFrame(S), dt)
    df_res = pd.DataFrame(params)
    print(df_res)

    df_orig = pd.DataFrame([[mu, vol, lambda_, q1, q2] ], columns=['mu', 'vol', 'lambda_', 'q1', 'q2'],
                 index=['original'])
    print(df_orig)