/******************************************************
This Stan program partion the total residuals of a 
ground motion model into within-event-site, 
between-event, between-event path, and site residuals. 

Uses hard constraints to satisfy standard deviation 
requirements
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
  //random effects (unscaled, constrained)
  vector[NEQ]  deltaBprime  = deltaBprime_raw  * tau0 / calc_emp_std(deltaBprime_raw,0);
  vector[NEQ]  deltaBPprime = deltaBPprime_raw * tauP / calc_emp_std(deltaBPprime_raw,0);
  vector[NSTA] deltaSprime  = deltaSprime_raw  * phiS / calc_emp_std(deltaSprime_raw,0);

  //random effects
  vector[NEQ]  deltaB  = sclB  .* deltaBprime;
  vector[NEQ]  deltaBP = sclBP .* deltaBPprime;
  vector[NSTA] deltaS  = sclS  .* deltaSprime;
  
  //empirical variances
  real emp_std_dBprime  = calc_emp_std(deltaBprime,  0);
  real emp_std_dBPprime = calc_emp_std(deltaBPprime, 0);
  real emp_std_dSprime  = calc_emp_std(deltaSprime,  0);
}


model {
  vector[N] reff_tot;
  vector[N] deltaWS;
  vector[N] deltaWSprime;
  
  //random effects - priors
  //------------------------------------
  //earthquake dependent random effects
  target += normal_lpdf(deltaBprime  | 0, tau0); //deltaB  ~ normal(0., tau0_array);
  target += normal_lpdf(deltaBPprime | 0, tauP); //deltaBP ~ normal(0., tauP_array);
  //site dependent random effects
  target += normal_lpdf(deltaSprime | 0, phiS);  //deltaS  ~ normal(0., phiS_array);

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
  target += normal_lpdf(deltaWSprime | 0, phi0);  //Y ~ normal(reff_tot, phi0_array);
}
