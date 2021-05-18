function [S, jump_simul] = JGBM_simulation( N_Sim ,T ,dt , params , S0 )
mu_star = params (1); 
sigma_ = params (2); 
lambda_ = params (3);
mu_y_ = params (4); 
sigma_y_ = params (5);
M_simul = zeros ( N_Sim ,T );
jump_simul = zeros ( N_Sim ,T );
for t =1: T
    jumpnb = poissrnd ( lambda_ * dt , N_Sim ,1);
    jump = normrnd ( mu_y_ * ( jumpnb - lambda_ * dt ), sqrt ( jumpnb )* sigma_y_ );
    M_simul(: , t) = mu_star * dt + sigma_ * sqrt( dt )* randn ( N_Sim ,1) + jump ;
    jump_simul(:, t) = jump;
end
S= ret2price( M_simul' , S0' ) ;
end


