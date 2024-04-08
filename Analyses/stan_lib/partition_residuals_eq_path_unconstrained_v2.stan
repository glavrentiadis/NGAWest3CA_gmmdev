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

parameters {
  //random effects (unscaled)
  vector[NEQ]  deltaBprime;
  vector[NEQ]  deltaBPprime;
  vector[NSTA] deltaSprime;
}

transformed parameters {
  //random effects
  vector[NEQ]  deltaB  = sclB  .* deltaBprime;
  vector[NEQ]  deltaBP = sclBP .* deltaBPprime;
  vector[NSTA] deltaS  = sclS  .* deltaSprime;
  
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
  deltaBprime  ~ normal(0., tau0);
  deltaBPprime ~ normal(0., tauP);
  //site dependent random effects
  deltaSprime  ~ normal(0., phiS);

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
  deltaWSprime ~ normal(0., phi0);
}
