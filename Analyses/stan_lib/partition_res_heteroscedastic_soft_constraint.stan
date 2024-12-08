/******************************************************
This Stan program partitions the total residuals of a 
ground motion model into between-event, between-event-path, 
between-site, and within-event-site residuals. 

Aleatory variability includes:
  * between-event term (mag dependent)
  * between-event-path term
  * between-site
  * within-event-site term (mag dependent)
******************************************************/
functions {
  //empirical standard deviation
  real calc_emp_std(vector y, real mu) {

    return sqrt( mean( (y-mu).^2 ) );
  }
  
  // linear interpolation
  real interp(real x1, real x2, real y1, real y2, real x) {
    
    if (x <= x1)
      return y1;
    else if (x >= x2)
      return y2;
    else
      return y1 + (y2-y1)/(x2-x1)*(x-x1);
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

  //Lagrange std penalty
  real lambda;

  //aleatory standard deviations
  real s_1;
  real s_2;
  real s_3;
  real s_4;
  real s_5;
  real s_6;
  //aleatory magnitude breaks
  real s_1mag;
  real s_2mag;
  real s_5mag;
  real s_6mag; 

  //input arrays
  vector[NEQ] mag;  //magnitude
  vector[N]   rrup; //rupture distance
  
  //scale factor for dBP
  real scl_dBP;
  //rupture distance offset
  real rrup_offset_dBP;
  
  //output
  vector[N] Y; //total residuals
}

transformed data{
  //between event path scaling
  vector[N] f_dBP = scl_dBP * (rrup - rrup_offset_dBP);
}

parameters {
  //random effects
  vector[NEQ]  deltaB;
  vector[NSTA] deltaS;
  //scaled random effects
  vector[NEQ]  deltaBP_scl;
}

transformed parameters {
  //magnitude-dependent aleatory standard dev
  vector[NEQ] tau0;
  vector[NEQ] phi0;
  for (i in 1:NEQ) {
    tau0[i]  = interp(s_1mag, s_2mag, s_1, s_2, mag[i]);
    phi0[i]  = interp(s_5mag, s_6mag, s_5, s_6, mag[i]);
  }
  //constant aleatory standard dev
  real tauP = s_3;
  real phiS = s_4;  
  real tauP_scl = s_2 / scl_dBP;
  
  //between-event-path random effects
  vector[NEQ] deltaBP = scl_dBP * deltaBP_scl;
  
  //within-event terms
  vector[N] deltaWS = Y - (deltaB[eq] + f_dBP .* deltaBP_scl[eq] + deltaS[sta]);
}

model {
  //random effects - priors
  //------------------------------------
  //earthquake dependent random effects
  deltaB      ~ normal(0., tau0);
  deltaBP_scl ~ normal(0., tauP_scl);
  //site dependent random effects
  deltaS      ~ normal(0., phiS);

  //likelihood function
  //------------------------------------
  deltaWS ~ normal(0., phi0[eq]);

  //standardized random effects
  //------------------------------------
  vector[NEQ]  epsB  = deltaB  ./ tau0;
  vector[NSTA] epsS  = deltaS   / phiS;
  vector[N]    epsWS = deltaWS ./ phi0[eq];
  //standardized effects of scaled terms
  vector[NEQ]  epsBP = deltaBP_scl / tauP_scl;

  //penalty terms (soft constraint)
  //------------------------------------
  //scaled by the total number of samples
  rep_array(calc_emp_std(epsB,  0), N) ~ normal(1., 1./lambda);
  rep_array(calc_emp_std(epsBP, 0), N) ~ normal(1., 1./lambda);
  rep_array(calc_emp_std(epsS,  0), N) ~ normal(1., 1./lambda);
  rep_array(calc_emp_std(epsWS, 0), N) ~ normal(1., 4./lambda);
  // //scaled by the number of event and station samples
  // rep_array(calc_emp_std(epsB,  0), NEQ)  ~ normal(1., 1/lambda);
  // rep_array(calc_emp_std(epsBP, 0), NEQ)  ~ normal(1., 1/lambda);
  // rep_array(calc_emp_std(epsS,  0), NSTA) ~ normal(1., 1/lambda);
  // rep_array(calc_emp_std(epsWS, 0), N)    ~ normal(1., 1/lambda);
}
