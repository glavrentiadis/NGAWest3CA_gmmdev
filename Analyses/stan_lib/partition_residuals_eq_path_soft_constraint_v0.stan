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

  //lagrange std penalty
  real lambda;

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
  
  real SDdeltaBprime  = calc_emp_std(deltaBprime,  0);
  real SDdeltaBPprime = calc_emp_std(deltaBPprime, 0);
  real SDdeltaSprime  = calc_emp_std(deltaSprime,  0);
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
  
  //penalty terms (soft constraint)
  //------------------------------------
  // //scaled by lambda
  // target += lambda * normal_lpdf(calc_emp_std(deltaBprime,  0) | tau0, 1/lambda);
  // target += lambda * normal_lpdf(calc_emp_std(deltaBPprime, 0) | tauP, 1/lambda);
  // target += lambda * normal_lpdf(calc_emp_std(deltaSprime,  0) | phiS, 1/lambda);
  // target += lambda * normal_lpdf(calc_emp_std(deltaWSprime, 0) | phi0, 1/lambda);
  // //scaled by the number of observations 
  // target += N * normal_lpdf(calc_emp_std(deltaBprime,  0) | tau0, 1/lambda);
  // target += N * normal_lpdf(calc_emp_std(deltaBPprime, 0) | tauP, 1/lambda);
  // target += N * normal_lpdf(calc_emp_std(deltaSprime,  0) | phiS, 1/lambda);
  // target += N * normal_lpdf(calc_emp_std(deltaWSprime, 0) | phi0, 1/lambda);
  //scaled by the number of samples
  target += NEQ  * normal_lpdf(calc_emp_std(deltaBprime,  0) | tau0, 1/lambda);
  target += NEQ  * normal_lpdf(calc_emp_std(deltaBPprime, 0) | tauP, 1/lambda);
  target += NSTA * normal_lpdf(calc_emp_std(deltaSprime,  0) | phiS, 1/lambda);
  target += N    * normal_lpdf(calc_emp_std(deltaWSprime, 0) | phi0, 1/lambda);
}
