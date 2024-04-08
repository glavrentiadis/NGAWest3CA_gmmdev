/******************************************************
This Stan program partion the total residuals of a 
ground motion model into within-event-site, 
between-event, between-event path, and site residuals. 
******************************************************/
functions {
  real calc_emp_std(vector y, real mu) {

    //empirical standard deviation
    return sqrt( mean( (y-mu).^2 ) );
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

transformed data {
  //mag dependent aleatory variability
  vector[N]    phi0_array = sclWS * phi0;
  vector[NSTA] phiS_array = sclS  * phiS;
  vector[NEQ]  tau0_array = sclB  * tau0;
  vector[NEQ]  tauP_array = sclBP * tauP;
}

parameters {
  //random effects
  vector[NEQ]  deltaB;
  vector[NEQ]  deltaBP;
  vector[NSTA] deltaS;
}

transformed parameters {
  //random effects (unscaled)
  vector[NEQ]  deltaBprime  = deltaB  ./ sclB;
  vector[NEQ]  deltaBPprime = deltaBP ./ sclBP;
  vector[NSTA] deltaSprime  = deltaS  ./ sclS;;

  //empirical standard deviations  
  real SDdeltaBprime  = calc_emp_std(deltaBprime,  0);
  real SDdeltaBPprime = calc_emp_std(deltaBPprime, 0);
  real SDdeltaSprime  = calc_emp_std(deltaSprime,  0);
}

model {
  vector[N] reff_tot;
  vector[N] deltaWS;
  vector[N] deltaWSprime;
  
  //priors
  //------------------------------------
  //earthquake dependent random effects
  deltaB  ~ normal(0., tau0_array);
  deltaBP ~ normal(0., tauP_array);
  //site dependent random effects
  deltaS  ~ normal(0., phiS_array);

  //total random effects
  reff_tot = deltaB[eq] + Rrup .* deltaBP[eq] + deltaS[sta];

  //within-event residuas
  //------------------------------------
  //within-event terms
  deltaWS = Y - reff_tot;
  //normalized within-event terms
  deltaWSprime = deltaWS ./ sclWS;

  //likelihood function
  //------------------------------------
  deltaWS ~ normal(0., phi0_array);
}
