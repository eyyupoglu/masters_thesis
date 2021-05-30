import numpy as np





from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from math import exp, log, sqrt, pi
import pandas as pd
import time

start_time = time.time()


# simulate the asset paths under Heston model
def EulerMilsteinSim(scheme, negvar, numPaths, rho, S_0, V_0, T, kappa, theta, sigma, r, q, dt):
    num_time = int(T / dt)
    S = np.zeros((num_time + 1, numPaths))
    S[0, :] = S_0
    V = np.zeros((num_time + 1, numPaths))
    V[0, :] = V_0
    Vcount0 = 0
    for i in range(numPaths):
        for t_step in range(1, num_time + 1):
            # the 2 stochastic drivers for variance V and asset price S and correlated
            Zv = np.random.randn(1)
            Zs = rho * Zv + sqrt(1 - rho ** 2) * np.random.randn(1)
            # users can choose either Euler or Milstein scheme
            if scheme == 'Euler':
                V[t_step, i] = V[t_step - 1, i] + kappa * (theta - V[t_step - 1, i]) * dt + sigma * sqrt(
                    V[t_step - 1, i]) * sqrt(dt) * Zv
            elif scheme == 'Milstein':
                V[t_step, i] = V[t_step - 1, i] + kappa * (theta - V[t_step - 1, i]) * dt + sigma * sqrt(
                    V[t_step - 1, i]) * sqrt(dt) * Zv \
                               + 1 / 4 * sigma ** 2 * dt * (Zv ** 2 - 1)

            if V[t_step, i] <= 0:
                Vcount0 = Vcount0 + 1
                if negvar == 'Reflect':
                    V[t_step, i] = abs(V[t_step, i])
                elif negvar == 'Trunca':
                    V[t_step, i] = max(V[t_step, i], 0)

            ################         simluations for asset price S              ########
            S[t_step, i] = S[t_step - 1, i] * np.exp(
                (r - q - V[t_step - 1, i] / 2) * dt + sqrt(V[t_step - 1, i]) * sqrt(dt) * Zs)
    return S, V, Vcount0




##############################################   Parameters Values     ##############################################
numPaths = 2
rho = 0.8         # correlation of the bivariables
S_0 = 1             # initila asset price
V_0 = (0.2)**2      # initial variance
kappa = 2           # mean-reversion rate
theta = (0.6)**2    # long-run variance
sigma = 0.3         # volatility of volatility
r = 0.3            # risk-free interest rate
q = 0.0             # dividend
dt = 1/52         # size of time-step
Tmax = 10            # longest maturity


scheme='Milstein'
negvar='Trunca'
S, V, Vcount0 = EulerMilsteinSim(scheme, negvar, numPaths, rho, S_0 , V_0, Tmax, kappa, theta, sigma, r, q, dt)
df_spot = pd.DataFrame(S).T
df_vol=pd.DataFrame(V).T

((np.log(df_spot.xs(1).shift(1)) - np.log(df_spot.xs(1))).rolling(20).std() * np.sqrt(12)).plot(label='std(logreturn) vols')
df_vol.xs(1).plot(label='original vols')
plt.legend()



# Create some mock data
t = np.arange(0.01, 10.0, 0.01)
data1 = np.exp(t)
data2 = np.sin(2 * np.pi * t)

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('time (s)')
ax1.set_ylabel('exp', color=color)
ax1.plot(df_spot.xs(1), color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('sin', color=color)  # we already handled the x-label with ax1
ax2.plot(df_vol.xs(1), color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()



vols_est = ((np.log(df_spot.xs(1).shift(1)) - np.log(df_spot.xs(1))).rolling(20).std() * np.sqrt(12))
vols_original = df_vol.xs(1).dropna()

spots = df_spot.xs(1)
vols = vols_original



mu_est, kappa_est, sigma_est, theta_est, rho_est = estimate_heston(vols, spots, dt)

print('Estimated mu: %.2f, kappa: %.2f, sigma: %.2f, theta %.2f, ro %.2f ' % (mu_est, kappa_est, sigma_est, theta_est, rho_est))
print('Original mu: %.2f, kappa: %.2f, sigma: %.2f, theta %.2f, ro %.2f  ' % (r, kappa, sigma, theta, rho))

# Initialization
Vk, Pk = 0.8, 0.1
# kappa, theta, sigma, rho = 4, 0.03, 0.1, 0.1
Q = np.array([[0.5, 0],
              [0, 0.5]])
kappa = 2  # mean-reversion rate
theta = (0.6) ** 2  # long-run variance
sigma = 0.3  # volatility of volatility
rho = rho
r = 0.3  # risk-free interest rate
q = 0.0

vk_list = []
for k in range(len(spots) - 1):
    # Linearization matrices of the state function
    F = 1 - kappa * dt
    L = np.array([0, sigma * np.sqrt(Vk * dt)])

    # Update the state prediction estimation and the prediction estimation-error covariance
    Vk1 = Vk + kappa * theta * dt - kappa * Vk * dt

    dQ = Pk * np.abs(1 - kappa * dt) ** 2 + dt ** 2 * np.abs(kappa * theta) ** 2 + np.abs(sigma) ** 2 * dt * Vk * Q[
        1, 1] - F * Pk * F + L @ Q @ L.T
    Pk1 = F * Pk * F + L @ Q @ L.T + dQ

    # Linearization matrices of the measurement function
    H = -1 / 2 * dt
    M = np.array([np.sqrt((1 - rho ** 2) * Vk * dt), rho * np.sqrt(Vk * dt)])

    # Update the state estimate and error covariance
    K = (Pk1 * H + L @ Q @ M.T) * (H * Pk * H + M @ Q @ M.T + H * L @ Q @ M.T + M @ Q @ L * H) ** (-1)

    zk = np.log(spots[k + 1]) - np.log(spots[k])
    Vk1 = Vk1 + K * (zk - 1 / 2 * Vk1)

    dR = Pk1 * (1 + K * dt / 2) ** 2 + 2 * K ** 2 * dt * Vk * (
                (1 - rho ** 2) * Q[0, 0] + rho ** 2 * Q[1, 1]) - Pk1 + K * (H * Pk1 + M @ L.T)

    Pk1 = Pk1 - K * (H * Pk1 + M @ L.T) + dR

    Vk, Pk = np.abs(Vk1), np.abs(Pk1)

    vk_list.append(Vk)

plt.plot(vk_list)
plt.show()


