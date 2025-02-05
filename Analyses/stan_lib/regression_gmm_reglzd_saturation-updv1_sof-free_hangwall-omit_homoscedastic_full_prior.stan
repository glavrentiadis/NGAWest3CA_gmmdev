/******************************************************
Full MCMc regression to determine the coefficients of 
EAS ergodic GMM. Assumes homoscedastic aleatory variability.
Includes updated saturation scaling.

Mean scaling includes:
  * magnitude scaling (small-to-medium mag scaling)
  * depth to top of rupture
  * normal and reverse scaling
  * geometrical spreading
  * anealstic attenuation
  * vs30 scaling

Fixed terms include:
  * magnitude scaling (large events)
  * short distance finite-fault saturation
  * magnitude break in mag scaling
  * width of magnitude transition
  * maximum depth to top of rupture

Aleatory variability includes:
  * between-event term
  * between-event-path term
  * between-site
  * within-event-site term

Regionalized terms:
 median scaling:
  * anelastic attenuation
  * vs30 scaling
 aleatory variability:
  * aleatory between-event path variability
  * aleatory between-site variability
  * aleatory within-event-site variability
******************************************************/

functions {
  // linear interpolation
  real interp(real x1, real x2, real y1, real y2, real x) {
    real y; //deceleration
    
    if (x <= x1)
      y = y1;
    else if (x >= x2)
      y = y2;
    else
      y = y1 + (y2-y1)/(x2-x1)*(x-x1);
        
    return y;
  }  
}

data {
  //dataset size
  int N;      //number of data points
  int NEQ;    //number of events
  int NST;    //number of stations
  int NREG; //number of regions for event parameters

  //indices
  array[N]   int <lower=1, upper=NEQ>  eq;    //event indices
  array[N]   int <lower=1, upper=NST>  st;    //station indices
  array[N]   int <lower=1, upper=NREG> reg;   //event region indices
  array[NEQ] int <lower=1, upper=NREG> regeq; //event region indices
  array[NST] int <lower=1, upper=NREG> regst; //event region indices
    
  //input arrays
  vector[NEQ] mag;  //magnitude
  vector[NEQ] ztor; //depth to top of rupture
  vector[NEQ] sof;  //style of faulting
  vector[N]   rrup; //rupture distance
  vector[NST] vs30; //rupture distance
  
  //scale factor for anelastic attenuation
  real scl_atten;
  //scale factor for dBP
  real scl_dBP;

  //mean parameters
  real c_1mu;
  real c_3mu;
  real c_4mu;
  real c_7mu;
  real c_8mu;
  real c_9mu;
  real c_10amu;
  real c_10bmu;
  
  //fixed parameters
  real c_2fxd;
  //short distance
  real c_5fxd;
  real c_6fxd;
  //magnitude breaks
  real c_nfxd;
  real c_hmfxd;
  real c_magfxd;
  //max depth to top of rupture
  real ztor_max;

  //output
  vector[N] Y; //total residuals
}

transformed data {
  //scaled priors
  real c_7mu_scl = 1/scl_atten * c_7mu;

  //source scaling
  // - - - - - - - - - -
  //depth to top of rupture scaling
  vector[NEQ] f_ztor;
  for (i in 1:NEQ){
    f_ztor[i] = min([ztor[i], ztor_max]);
  }
  //style of faulting scaling
  vector[NEQ] f_r;
  vector[NEQ] f_n;
  for (i in 1:NEQ){
    f_r[i] = sof[i] >  0.25 ?  sof[i] : 0.; //reverse fault (sof== 1)
    f_n[i] = sof[i] < -0.25 ? -sof[i] : 0.; //normal fault (sof==-1)
  }
  
  //path scaling
  // - - - - - - - - - -
  vector[N] f_gs;
  for (i in 1:N)
    f_gs[i] = log( rrup[i] + c_5fxd * exp(c_6fxd * max([mag[eq[i]]-c_hmfxd, 0.])) );
  //geometrical spreading scaling at large distances
  vector[N] f_gs_lrup = log( sqrt(rrup.^2 + 50.^2.) );
  //anelastic attenuation
  vector[N] f_atten = scl_atten * rrup;
  
  //site scaling
  // - - - - - - - - - -
  //vs30 scaling
  vector[NST] f_vs30;
  for (i in 1:NST)
    f_vs30[i] = log( min([vs30[i], 1000.]) / 800. );
    
  //between event path scaling
  vector[N] f_dBP = scl_dBP * rrup;
}

parameters {
  //coefficients
  //------------------------------------
  //intercept
  real c_1;
  //source scaling
  real c_3;  //small-to-medium mag scaling
  real c_9;
  real c_10a;
  real c_10b;
  //path scaling
  real<lower=-10.0, upper=0.0> c_4;  //geometrical spreading
  //site scaling
  real c_8; 
  //scaled parameters
  real<lower=-10.0,  upper=0.0> c_7_scl;  //scaled anelastic attenuation

  //coefficient regionalization
  //------------------------------------
  vector<lower=0.1, upper=10.0>[NREG-1] lambda_1r;
  vector<lower=0.1, upper=10.0>[NREG-1] lambda_7r;
  vector<lower=0.1, upper=10.0>[NREG-1] lambda_8r;

  //aleatory standard deviations
  //------------------------------------
  //event std
  real<lower=0.01> tau;
  //site std
  real<lower=0.01> phi_S;
  //nugget
  real<lower=0.01> phi;
  //scaled terms
  real<lower=0.01> tau_P_scl;

  //aleatory standard regionalization
  //------------------------------------
  vector<lower=0.1, upper=10.0>[NREG-1] lambda_tauPr;
  vector<lower=0.1, upper=10.0>[NREG-1] lambda_phiSr;
  vector<lower=0.1, upper=10.0>[NREG-1] lambda_phir;
  
  //random terms
  //------------------------------------
  vector[NEQ] deltaB;
  vector[NST] deltaS;
  //scaled random effects
  vector[NEQ] deltaBP_scl;
}

transformed parameters {
  //fixed parameters
  //------------------------------------
  real c_2   = c_2fxd;
  real c_5   = c_5fxd;
  real c_6   = c_6fxd;
  real c_n   = c_nfxd;
  real c_hm  = c_hmfxd;
  real c_mag = c_magfxd;  
  
  //regional terms
  //------------------------------------
  //coefficients
  vector[NREG] c_1r = c_1 * append_row(1., lambda_1r);
  vector[NREG] c_8r = c_8 * append_row(1., lambda_8r);
  //aleatory standard deviations
  vector[NREG] phi_Sr = phi_S * append_row(1., lambda_phiSr);
  vector[NREG] phi_r  = phi   * append_row(1., lambda_phir);
  //scaled terms
  vector[NREG] c_7r_scl = c_7_scl * append_row(1., lambda_7r);  
  vector[NREG] tau_Pr_scl = tau_P_scl * append_row(1., lambda_tauPr);
 
  //original-scale parameters
  //------------------------------------
  //median scaling
  real c_7 = scl_atten * c_7_scl;
  vector[NREG] c_7r = scl_atten * c_7r_scl;
  //aleatory terms
  real tau_P = scl_dBP * tau_P_scl;
  vector[NREG] tau_Pr = scl_dBP * tau_Pr_scl;
  //random effects
  vector[NEQ] deltaBP =  scl_dBP * deltaBP_scl;

  //mean scaling
  //------------------------------------
  vector[N] f_med;
  //intercept
  f_med  = c_1r[reg];
  //source scaling
  f_med  += (c_2 * (mag[eq] - 6.) + 
            (c_2 - c_3) / c_n * log( 1 + exp(c_n * (c_mag - mag[eq])) ) +
            c_9   * f_ztor[eq] + 
            c_10a * f_r[eq] +
            c_10b * f_n[eq]);
  //path scaling
  f_med += (c_4 * f_gs - 
            c_4 * f_gs_lrup -
            0.5 * f_gs_lrup +
            c_7r_scl[reg] .* f_atten);
  //site scaling
  f_med += c_8r[reg] .* f_vs30[st];
  
  //aleatory variability
  //------------------------------------
  real s_1 = tau;
  real s_2 = tau;
  real s_3 = tau_P;
  real s_4 = phi_S;
  real s_5 = phi;
  real s_6 = phi;
  //regionalized aleatory std
  vector[NREG] s_3r = tau_Pr;
  vector[NREG] s_4r = phi_Sr;
  vector[NREG] s_5r = phi_r;
  vector[NREG] s_6r = phi_r;

  //within event-site residuals
  //------------------------------------
  vector[N] deltaWS = Y - f_med - (deltaB[eq] + f_dBP .* deltaBP_scl[eq] + deltaS[st]);

}

model {
  //evaluate priors
  //------------------------------------
  //global coefficients
  c_1   ~ normal(c_1mu, 0.5);
  c_3   ~ normal(c_3mu, 0.5);
  c_4   ~ normal(c_4mu, 0.5);
  c_8   ~ normal(c_8mu, 0.5);
  c_9   ~ normal(c_9mu, 0.01);
  c_10b ~ normal(c_10amu, 0.5);
  c_10a ~ normal(c_10bmu, 0.5);
  //scaled coefficients coefficients
  c_7_scl   ~ normal(c_7mu_scl, 0.25);

  //aleatory std
  tau   ~ lognormal(-0.35, 0.55);
  phi_S ~ lognormal(-0.35, 0.55);
  phi   ~ lognormal(-0.35, 0.55);
  //scaled aleatory var
  tau_P_scl ~ lognormal(-1.50, 0.40);

  //regional coefficient adjustments 
  lambda_1r ~ normal(1., 0.10);
  lambda_7r ~ normal(1., 0.10);
  lambda_8r ~ normal(1., 0.10);

  //regional aleatory std adjustments 
  lambda_tauPr ~ normal(1., 0.10);
  lambda_phiSr ~ normal(1., 0.10);
  lambda_phir  ~ normal(1., 0.10);

  //evaluate likelihood
  //------------------------------------
  //random effects
  deltaB  ~ normal(0,  tau);
  deltaS  ~ normal(0., phi_Sr[regst]);
  //scaled parameters
  deltaBP_scl ~ normal(0., tau_Pr_scl[regeq]);
  //noise
  deltaWS ~ normal(0., phi_r[reg]);
}


generated quantities {
  //hanging wall scaling
  real c_13 = 0.;
 }

