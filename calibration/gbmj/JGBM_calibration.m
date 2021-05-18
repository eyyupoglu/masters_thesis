function [ mu_star, sigma_, lambda_, mu_y_, sigma_y_ ] = JGBM_calibration( data ,dt , params )
    returns = price2ret(data);
    dataLength= length (data);
    options = optimset ( 'MaxFunEvals', 1000000 , 'MaxIter', 1000000);
    params = fminsearch( @FT_JGBM_LL , params , options );
    mu_star = params (1); 
    sigma_ = abs( params (2)); 
    lambda_ = abs( params (3));
    mu_y_ = params (4); 
    sigma_y_ = abs( params (5));
    function mll = FT_JGBM_LL( params )
        mu_ = params (1); 
        sigma_ = abs( params (2)); 
        lambda_ = abs( params (3));
        mu_y_ = params (4); 
        sigma_y_ = abs( params (5));
        Max_jumps = 5;
        factoriel = factorial(0: Max_jumps);
        LogLikelihood = 0;
        for time =1: dataLength
            ProbDensity = 0;
            for jumps =0: Max_jumps -1
                jumpProb = exp(- lambda_ * dt )*( lambda_ * dt )^ jumps / factoriel( jumps +1);
                condVol = dt * sigma_ ^2+ jumps * sigma_y_ ^2;
                condMean = mu_ * dt + mu_y_ *( jumps - lambda_ * dt );
                condToJumps = exp ( -( data(time)- condMean )^2 / condVol /2)/ sqrt(pi *2* condVol);
                ProbDensity = ProbDensity + jumpProb * condToJumps;
            end
            LogLikelihood = LogLikelihood + log( ProbDensity);
        end
        mll = - LogLikelihood;
    end
end