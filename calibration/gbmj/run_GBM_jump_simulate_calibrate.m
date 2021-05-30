dt=1/252;
N_Sim =1;
S0=10;
T=752;

% mu_star = 0; sigma_ = 0.15; lambda_ = 10;
% mu_y_ = 0.03; sigma_y_ = 0.001;

lambda_ = 2; sigma_y_ = 0.003; sigma_ = 0.2;
mu_y_ = 0.1; mu_star = 0.2; 

[S, jumps] = JGBM_simulation( N_Sim,T , dt , [mu_star, sigma_,  lambda_, mu_y_, sigma_y_] ,  S0 );
% plot(S)
figure
plot(jumps(1, 1:end))
figure
plot(S)
% close

% figure
% hist(log(S(2:end, end)) - log(S(1:end-1, end)), 50)
% % close
% 
% figure
% qqplot(log(S(2:end, end)) - log(S(1:end-1, end)))
% % close

% 
% close
% close
% close
% returns = log(S(2:end)) - log(S(1:end-1));
[ mu_star_est, sigma__est, lambda__est, mu_y__est, sigma_y__est ] = JGBM_calibration( S...
    ,dt , [0.2, 0.2,  0.2, 0.2, 0.2] );

disp([ mu_star_est, sigma__est, lambda__est, mu_y__est, sigma_y__est ])
disp([mu_star, sigma_,  lambda_, mu_y_, sigma_y_])


[S2, jumps] = JGBM_simulation( 20,T , dt , [mu_star_est, sigma__est, lambda__est, mu_y__est, sigma_y__est] ,  S0 );
% plot(S)
figure
% plot(jumps(1, 1:end))
plot(S2)