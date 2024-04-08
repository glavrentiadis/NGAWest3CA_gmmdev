/******************************************************
This Stan program partion the total residuals of a 
ground motion model into within-event-site, 
between-event, between-event path, and site residuals. 

******************************************************/
functions {
  real calc_emp_std(vector y, real mu) {
    //decelerations
    real N = size(y);
    
    //empirical standard deviation
    real sigma_emp = sqrt( (y-mu)' * (y-mu) / N );
        
    return sigma_emp;
  }
}

data {
  //dataset size
  int N;    //number of data points
  int NEQ;  //number of events
  int NSTA; //number of stations

  //indices
  array[N] int<lower=1, upper=NEQ>  eq;  //event indices
  array[N] int<lower=1, upper=NSTA> sta; //station indices

  //aleatory standard deviations
  real phi0;
  real phiS;
  real tau0;
  real tauP;

  //aleatory var scaling
  vector[N]    sclWS;
  vector[NSTA] sclS;
  vector[NEQ]  sclB;
  vector[NEQ]  sclBP;  
  
  //constraint weights
  vector[N]    wtWS;
  vector[NSTA] wtS;
  vector[NEQ]  wtB;
  vector[NEQ]  wtBP;  

  //input arrays
  vector[N] Rrup; //rupture distance

  //output
  vector[N] Y; //total residuals
}

parameters {
  //random effects (unscaled, raw)
  vector[NEQ]  deltaBprime_raw;
  vector[NEQ]  deltaBPprime_raw;
  vector[NSTA] deltaSprime_raw;
}

transformed parameters {
  //empirical adjusted raw deviations
  real emp_std_dBprime_raw  = calc_emp_std(deltaBprime_raw,  0);
  real emp_std_dBPprime_raw = calc_emp_std(deltaBPprime_raw, 0);
  real emp_std_dSprime_raw  = calc_emp_std(deltaSprime_raw,  0);
  
  //random effects (unscaled, constrained)
  vector[NEQ]  deltaBprime  = deltaBprime_raw  .* (1. + wtB  * (tau0 - emp_std_dBprime_raw)  / emp_std_dBprime_raw);
  vector[NEQ]  deltaBPprime = deltaBPprime_raw .* (1. + wtBP * (tauP - emp_std_dBPprime_raw) / emp_std_dBPprime_raw);
  vector[NSTA] deltaSprime  = deltaSprime_raw  .* (1. + wtS  * (phiS - emp_std_dSprime_raw)  / emp_std_dSprime_raw);
    
  //empirical adjusted standard deviations
  real emp_std_dBprime  = calc_emp_std(deltaBprime,  0);
  real emp_std_dBPprime = calc_emp_std(deltaBPprime, 0);
  real emp_std_dSprime  = calc_emp_std(deltaSprime,  0);
  
  //random effects
  vector[NEQ]  deltaB  = sclB  .* deltaBprime;
  vector[NEQ]  deltaBP = sclBP .* deltaBPprime;
  vector[NSTA] deltaS  = sclS  .* deltaSprime;
}

model {
  vector[N] reff_tot;
  vector[N] deltaWS;
  vector[N] deltaWSprime;
  
  //random effects - priors
  //------------------------------------
  //earthquake dependent random effects
  deltaBprime  ~ normal(0., tau0);
  deltaBPprime ~ normal(0., tauP);
  //site dependent random effects
  deltaSprime  ~ normal(0., phiS);

  //total random effects' contribution
  reff_tot = deltaB[eq] + Rrup .* deltaBP[eq] + deltaS[sta];

  //within-event residuas
  //------------------------------------
  //within-event terms
  deltaWS = Y - reff_tot;
  //normalized within-event terms
  deltaWSprime = deltaWS ./ sclWS;

  //likelihood function
  //------------------------------------
  deltaWSprime ~ normal(0., phi0);
}
