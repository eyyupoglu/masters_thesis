clc; clear;

% Option features
r = 0.05;           % Risk free rate
q = 0;           % Dividend yield
Mat = 3;          % Maturity in years
S0 = 100;           % Spot price

% Heston parameters
kappa =  0.2;       % Variance reversion speed
theta =  1;    % Variance reversion level
sigma =  0.25;     % Volatility of Variance
v0    =  0.4;    % Initial variance
lambda = 0;       % Risk parameter

% Simulation features
N = 1;             % Number of stock price paths
T = 150;           % Number of time steps per path
alpha = 0.5;       % Weight for explicit-implicit scheme
negvar = 'T';      % Use the truncation scheme for negative variances
rho   =  -0.25;      % Correlation between Brownian motions
params = [kappa theta sigma v0 rho ];


%% Simulate the processes and obtain the option prices
schemeV = 'M';
% [S V F Price] = EulerMilsteinPrice(schemeV,negvar,params,PutCall,S0,K,Mat,r,q,T,N,alpha);
[S V F] = EulerMilsteinSim(schemeV,negvar,params,S0,Mat,r,q,T,N,alpha);

%%
% Select method : 1 = Likelihood, 2 = Log-Likelihood.  Set the options.
x = log(S);
Lmethod = 2;
dt=1/50;
% kappa theta sigma v0 rho
e = 1e-5;
lb = [e   e  0.05  0.005 -.4];  % Lower bound on the estimates
ub = [0.4  2  0.5  2  .4];  % Upper bound on the estimates
start = [0.1 0.05 0.3 0.1 -0.8];
% Optimization options
options = optimset('MaxFunEvals',1e6,'MaxIter',1e9);

param = fmincon(@(p) LikelihoodAW(p,x,r,q,dt,Lmethod),start,[],[],[],[],lb,ub,[],options);
[y v] = LikelihoodAW(param,x,r,q,dt,Lmethod);
figure

plot(v);


hold
plot(V);
ylim([0 1])
%% Display the true parameter values and compare to the estimates
clc;
fprintf('Estimates                       kappa     theta     sigma     v0        rho    \n')
fprintf('-------------------------------------------------------------------------------\n')
fmtstring = repmat('%10.4f',1,5);

fprintf(['Atiya-Wall MLE Estimates    ' fmtstring ' \n'], param)
fprintf('-------------------------------------------------------------------------------\n')
fprintf(['Original values             ' fmtstring ' \n'], params)
fprintf('-------------------------------------------------------------------------------\n')
%% Plot the results
% X = (1:T);
% [a,h1,h2] = plotyy(X,S,X,V,'plot');
% set(h1,'Color','k');
% set(h2,'Color','r');
% set(a(1),'YColor','k');
% set(a(2),'YColor','r');
% legend('Variance Level','Stock Price')
% clear h1 h2 a
