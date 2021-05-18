dt=1/52;
N_Sim =1;
S0=1;
T=156;

% mu_star = 0; sigma_ = 0.15; lambda_ = 10;
% mu_y_ = 0.03; sigma_y_ = 0.001;

lambda_ = 0.9; sigma_y_ = 0.3; sigma_ = 0.2;
mu_y_ = -0.1; mu_star = 0.02; 

[S, jumps] = JGBM_simulation( N_Sim,T , dt , [mu_star, sigma_,  lambda_, mu_y_, sigma_y_] ,  S0 );
% plot(S)
figure
% plot(jumps(1, 1:end))
plot(S)
% close

figure
hist(log(S(2:end, end)) - log(S(1:end-1, end)), 50)
% close

figure
qqplot(log(S(2:end, end)) - log(S(1:end-1, end)))
% close

% 
% close
% close
% close
[ mu_star_est, sigma__est, lambda__est, mu_y__est, sigma_y__est ] = JGBM_calibration( S(2:end, end) ,1/252 , [0, 0.1,  5, 0.09, 0.02] );

disp([ mu_star_est, sigma__est, lambda__est, mu_y__est, sigma_y__est ])
disp([mu_star, sigma_,  lambda_, mu_y_, sigma_y_])


[S, jumps] = JGBM_simulation( N_Sim,T , dt , [mu_star_est, sigma__est, lambda__est, mu_y__est, sigma_y__est] ,  S0 );
% plot(S)
figure
% plot(jumps(1, 1:end))
plot(S)