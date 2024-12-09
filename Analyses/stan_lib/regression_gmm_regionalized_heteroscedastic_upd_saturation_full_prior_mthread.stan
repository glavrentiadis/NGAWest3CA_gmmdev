/******************************************************
Full MCMc regression to determine the coefficients of 
EAS ergodic GMM. Assumes heteroscedastic aleatory variability.
Includes updated saturation scaling.

Multi-threading partitioning on likelihood evaluation

Mean scaling includes:
  * magnitude scaling (small-to-medium mag scaling)
  * geometrical spreading
  * anealstic attenuation
  * vs30 scaling
  * normal and reverse scaling

Fixed terms include:
  * magnitude scaling (large events)
  * short distance saturation
  * magnitude break in mag scaling
  * width of magnitude transition
  * magnitude scaling for short distance saturation
  * maximum depth to top of rupture

Aleatory variability includes:
  * between-event term (mag dependent)
  * between-event-path term
  * between-site
  * within-event-site term (mag dependent)

Regionalized terms:
 median scaling:
  * small magnitude scaling
  * anelastic attenuation
  * vs30 scaling
 aleatory variability:
  * aleatory between-event variability
  * aleatory between-event-path variability
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

  // normal log-likelihood for multi-threading
  real partial_normal_lpdf(array[] real y_slice,
                           int start, int end,
                           real mu, vector sigma) {
    return normal_lpdf(y_slice | mu, sigma[start:end] );
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
  //rupture distance offset
  real rrup_offset_dBP;

  //mean parameters
  real c_1mu;
  real c_3mu;
  real c_4mu;
  real c_7mu;
  real c_8mu;
  real c_9mu;
  real c_10amu;
  real c_10bmu;
  
  //aleat magnitude breaks
  real s_1mag;
  real s_2mag;
  real s_5mag;
  real s_6mag;
  
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
  
  //multi-treading partitioning
  int grainsize;
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
  vector[N] f_dBP = scl_dBP * (rrup - rrup_offset_dBP);
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
  vector<lower=0.1, upper=10.0>[NREG-1] lambda_3r;
  vector<lower=0.1, upper=10.0>[NREG-1] lambda_7r;
  vector<lower=0.1, upper=10.0>[NREG-1] lambda_8r;

  //aleatory standard deviations
  //------------------------------------
  //event std
  real<lower=0.01> s_1;
  real<lower=0.01> s_2;
  //site std
  real<lower=0.01> s_4;
  //nugget
  real<lower=0.01, upper=5.0> s_5;
  real<lower=0.01, upper=5.0> s_6;
  //scaled terms
  real<lower=0.01> s_3_scl;

  //aleatory standard regionalization
  //------------------------------------
  vector<lower=0.1, upper=10.0>[NREG-1] lambda_tau0r;
  vector<lower=0.1, upper=10.0>[NREG-1] lambda_tauPr;
  vector<lower=0.1, upper=10.0>[NREG-1] lambda_phiSr;
  vector<lower=0.1, upper=10.0>[NREG-1] lambda_phi0r;
  
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
  vector[NREG] c_3r = c_3 * append_row(1., lambda_3r);
  vector[NREG] c_8r = c_8 * append_row(1., lambda_8r);
  //aleatory standard deviations
  vector[NREG] s_1r = s_1 * append_row(1., lambda_tau0r);
  vector[NREG] s_4r = s_4 * append_row(1., lambda_phiSr);
  vector[NREG] s_5r = s_5 * append_row(1., lambda_phi0r);
  vector[NREG] s_6r = s_6 * append_row(1., lambda_phi0r);
  //scaled terms
  vector[NREG] c_7r_scl = c_7_scl * append_row(1., lambda_7r);
  vector[NREG] s_3r_scl = s_3_scl * append_row(1., lambda_tauPr);

  //original-scale parameters
  //------------------------------------
  //median scaling
  real c_7 = scl_atten * c_7_scl;
  vector[NREG] c_7r = scl_atten * c_7r_scl;
  //aleatory terms
  real s_3 = scl_dBP * s_3_scl;
  vector[NREG] s_3r = scl_dBP * s_3r_scl;
  //random effects
  vector[NEQ] deltaBP = scl_dBP * deltaBP_scl;

  //mean scaling
  //------------------------------------
  vector[N] f_med;
  //intercept
  f_med  = c_1r[reg];
  //source scaling
  f_med  += (c_2 * (mag[eq] - 6.) + 
            (c_2 - c_3r[reg]) / c_n .* log( 1 + exp(c_n * (c_mag - mag[eq])) ) +
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
  real tau   = s_1 ;
  real tau_P = s_3;
  real phi_S = s_4;
  real phi   = s_5;
  //regionalized aleatory std
  vector[NREG] tau0r  = s_1r;
  vector[NREG] tauPr = s_3r;
  vector[NREG] phiSr = s_4r;
  vector[NREG] phi0r  = s_5r;
  //scaled regionalized aleatory std
  vector[NREG] tauPr_scl = s_3r_scl;

  //within event-site residuals
  //------------------------------------
  vector[N] deltaWS = Y - f_med - (deltaB[eq] + f_dBP .* deltaBP_scl[eq] + deltaS[st]);
}

model {
  vector[NEQ] tau0r_array;
  vector[NEQ] phi0r_array;
  //evaluate priors
  //------------------------------------
  //global coefficients
  target += normal_lpdf(c_1   | c_1mu, 0.5);
  target += normal_lpdf(c_3   | c_3mu, 0.5);
  target += normal_lpdf(c_4   | c_4mu, 0.5);
  target += normal_lpdf(c_8   | c_8mu, 0.5);
  target += normal_lpdf(c_9   | c_9mu, 0.01);
  target += normal_lpdf(c_10b | c_10amu, 0.5);
  target += normal_lpdf(c_10a | c_10bmu, 0.5);
  //scaled coefficients coefficients
  target += normal_lpdf(c_7_scl | c_7mu_scl, 0.25);
  
  //aleatory std
  target += lognormal_lpdf(s_1 | -0.35, 0.55);
  target += lognormal_lpdf(s_2 | -0.35, 0.55);
  target += lognormal_lpdf(s_4 | -0.35, 0.55);
  target += lognormal_lpdf(s_5 | -0.35, 0.55);
  target += lognormal_lpdf(s_6 | -0.35, 0.55);
  //scaled aleatory var
  target += lognormal_lpdf(s_3_scl | -1.50, 0.40);
  
  //magnitude scaling
  for (i in 1:NEQ) {
    tau0r_array[i]  = interp(s_1mag, s_2mag, s_1r[regeq[i]], s_2, mag[i]);
    phi0r_array[i]  = interp(s_5mag, s_6mag, s_5r[regeq[i]], s_6r[regeq[i]], mag[i]);
  }
  
  //regional coefficient adjustments 
  target += normal_lpdf(lambda_1r | 1., 0.10);
  target += normal_lpdf(lambda_3r | 1., 0.10);
  target += normal_lpdf(lambda_7r | 1., 0.10);
  target += normal_lpdf(lambda_8r | 1., 0.10);

  //regional aleatory std adjustments 
  target += normal_lpdf(lambda_tau0r | 1., 0.10);
  target += normal_lpdf(lambda_tauPr | 1., 0.10);
  target += normal_lpdf(lambda_phiSr | 1., 0.10);
  target += normal_lpdf(lambda_phi0r | 1., 0.10);

  //evaluate likelihood
  //------------------------------------
  //random effects
  target += reduce_sum(partial_normal_lpdf, to_array_1d(deltaB),      grainsize, 0., tau0r_array);
  target += reduce_sum(partial_normal_lpdf, to_array_1d(deltaS),      grainsize, 0., phiSr[regst]);
  //scaled parameters
  target += reduce_sum(partial_normal_lpdf, to_array_1d(deltaBP_scl), grainsize, 0., tauPr_scl[regeq]);
  //noise
  target += reduce_sum(partial_normal_lpdf, to_array_1d(deltaWS),     grainsize, 0., phi0r_array[eq]);
}

