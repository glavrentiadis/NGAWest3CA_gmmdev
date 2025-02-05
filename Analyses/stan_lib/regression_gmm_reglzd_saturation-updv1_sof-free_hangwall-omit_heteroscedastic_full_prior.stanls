/******************************************************
Full MCMc regression to determine the coefficients of 
EAS ergodic GMM. Assumes heteroscedastic aleatory variability.
Includes updated saturation scaling, and hanging wall effects. 

Mean scaling includes:
  * magnitude scaling (small-to-medium mag scaling)
  * depth to top of rupture
  * normal and reverse scaling
  * geometrical spreading
  * anealstic attenuation
  * vs30 scaling
  * hanging wall scaling
  
Fixed terms include:
  * magnitude scaling (large events)
  * short distance finite-fault saturation
  * magnitude break in mag scaling
  * width of magnitude transition
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
  
  //ground motion
  vector gmm(int N, array[] int reg, array[] int eq, array[] int st,
             vector mag,
             vector f_smag, vector f_ztor, vector f_r, vector f_n, vector f_hw,
             vector f_gs, vector f_gs_lrup, vector f_atten, 
             vector f_vs30,
             real c_2, real c_4, real c_9, real c_10a, real c_10b, real c_13,
             vector c_1r, vector c_3r, vector c_7r_scl, vector c_8r){
  
    //intercept
    vector [N] f = c_1r[reg];
    //source scaling
    f  += (c_2 * (mag[eq] - 6.) + 
          (c_2 - c_3r[reg]) .* f_smag[eq] +
          c_9   * f_ztor[eq] + 
          c_10a * f_r[eq] +
          c_10b * f_n[eq] +
          c_13  * f_hw);
    //path scaling
    f += (c_4 * f_gs - 
          c_4 * f_gs_lrup -
          0.5 * f_gs_lrup +
          c_7r_scl[reg] .* f_atten);
    //site scaling
    f += c_8r[reg] .* f_vs30[st];
  
    return f;
  }
  //ground motion
  vector gmm_src(int N, array[] int reg, array[] int eq, array[] int st,
                 vector mag,
                 vector f_smag, vector f_ztor, vector f_r, vector f_n, vector f_hw,
                 real c_2, real c_9, real c_10a, real c_10b, real c_13,
                 vector c_3r){
  
    //source scaling
    vector [N] f_src = (c_2 * (mag[eq] - 6.) + 
                       (c_2 - c_3r[reg]) .* f_smag[eq] +
                       c_9   * f_ztor[eq] + 
                       c_10a * f_r[eq] +
                       c_10b * f_n[eq] +
                       c_13  * f_hw);
  
    return f_src;
  }
  //ground motion (path)
  vector gmm_path(int N, array[] int reg, array[] int eq, array[] int st,
                  vector f_gs, vector f_gs_lrup, vector f_atten, 
                  real c_4, vector c_7r_scl){
  
    //path scaling
    vector [N] f_path = (c_4 * f_gs - 
                         c_4 * f_gs_lrup -
                         0.5 * f_gs_lrup +
                         c_7r_scl[reg] .* f_atten);
  
    return f_path;
  }
  //ground motion (site)
  vector gmm_site(int N, array[] int reg, array[] int eq, array[] int st,
                  vector f_vs30,
                  vector c_8r){
  
    //site scaling
    vector [N] f_site = c_8r[reg] .* f_vs30[st];
  
    return f_site;
  }

}

data {
  //dataset size
  int N;      //number of data points
  int NEQ;    //number of events
  int NST;    //number of stations
  int NREG;   //number of regions for event parameters

  //indices
  array[N]   int <lower=1, upper=NEQ>  eq;    //event indices
  array[N]   int <lower=1, upper=NST>  st;    //station indices
  array[N]   int <lower=1, upper=NREG> reg;   //event region indices
  array[NEQ] int <lower=1, upper=NREG> regeq; //event region indices
  array[NST] int <lower=1, upper=NREG> regst; //event region indices
    
  //input arrays
  vector[NEQ] mag;   //magnitude
  vector[NEQ] ztor;  //depth to top of rupture
  vector[NEQ] sof;   //style of faulting
  vector[NEQ] width; //width
  vector[NEQ] dip;   //dip
  vector[N]   rrup;  //rupture distance
  vector[N]   rx;    //normal distance
  vector[N]   ry;    //parallel distance
  vector[N]   ry0;   //parallel distance from edge
  vector[NST] vs30;  //rupture distance
  
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
  real c_13mu;
    
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
  //mag tapering coeff
  real a_2hwfxd;
  //distance tapering coefficients
  real h_1fxd;
  real h_2fxd;
  real h_3fxd;

  //output
  vector[N] Y; //total residuals
}

transformed data {
  //scaled priors
  real c_7mu_scl = 1/scl_atten * c_7mu;

  //source scaling
  // - - - - - - - - - -
  //small magnitude scaling
  vector[NEQ] f_smag = 1. / c_nfxd * log( 1 + exp(c_nfxd * (c_magfxd - mag)) );
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
  //hanging wall
  vector[N] f_hw;
  {
    //distance measures
    vector[N] r1 = width[eq] .* cos( pi()/180. * dip[eq] );
    vector[N] r2 = 3.0 * r1;
    vector[N] ry1 = rx * tan( pi()/180. * 20. );
    //tapering terms  
    real t_1;
    real t_2; 
    real t_3;
    real t_4;
    real t_5;
    for (i in 1:N){
      //dip tapering
      t_1  = min([(90.-dip[eq[i]])/45., 60./45.]);
      //mag tapering
      t_2  = 1. + a_2hwfxd * max([mag[eq[i]]-6.5,-1.]);
      t_2 -= (1. - a_2hwfxd) * min([ max([mag[eq[i]]-6.5,-1.]), .0])^2.;
      //normal distance tapering
      if (rx[i] < 0.)
        t_3 = 0.;
      else if (rx[i] < r1[i])
        t_3 = h_1fxd + h_2fxd * (rx[i] / r1[i]) + h_3fxd * (rx[i] / r1[i])^2.;
      else if (rx[i] < r2[i])
        t_3 = 1. - (rx[i] - r1[i]) / (r2[i] - r1[i]);
      else
        t_3 = 0.;
      //depth to top of rupture tapering
      t_4 = 1. - min([ztor[eq[i]], 10.])^2./1000;
      //parallel distance tapering
      if (ry0[i] < ry1[i])
        t_5 = 1.;
      else if (ry0[i] < ry1[i] + 5.)
        t_5 =  1. - (ry0[i] - ry1[i])/5.0;
      else
        t_5 = 0.;
      //
      f_hw[i] = t_1 * t_2 * t_3 * t_4 * t_5;
    }
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
    
  //aleatory scaling
  // - - - - - - - - - -
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
  real<lower=0.0> c_10a;
  real<upper=0.0> c_10b;
  real<lower=0.0> c_13;
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
  real<lower=0.01, upper=5.0> s_1;
  real<lower=0.01, upper=5.0> s_2;
  //site std
  real<lower=0.01, upper=5.0> s_4;
  //nugget
  real<lower=0.01, upper=5.0> s_5;
  real<lower=0.01, upper=5.0> s_6;
  //scaled terms
  real<lower=0.01, upper=5.0> s_3_scl;

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
  
  //evaluate median ground motion
  //------------------------------------
  vector[N] f_gmm = gmm(N, reg, eq, st,
                        mag,
                        f_smag, f_ztor, f_r, f_n, f_hw,
                        f_gs, f_gs_lrup, f_atten, 
                        f_vs30,
                        c_2fxd, c_4, c_9, c_10a, c_10b, c_13,
                        c_1r, c_3r, c_7r_scl, c_8r);
}

model {
  vector[NEQ] tau0r_array;
  vector[NEQ] phi0r_array;
  //evaluate priors
  //------------------------------------
  //global coefficients
  c_1   ~ normal(c_1mu,   0.5);
  c_3   ~ normal(c_3mu,   0.5);
  c_4   ~ normal(c_4mu,   0.5);
  c_8   ~ normal(c_8mu,   0.5);
  c_9   ~ normal(c_9mu,   0.01);
  c_10a ~ normal(c_10amu, 0.05);
  c_10b ~ normal(c_10bmu, 0.05);
  c_13  ~ normal(c_13mu,  0.1);
  //scaled coefficients coefficients
  c_7_scl ~ normal(c_7mu_scl, 0.25);
  
  //aleatory std
  s_1 ~ lognormal(-0.35, 0.55);
  s_2 ~ lognormal(-0.35, 0.55);
  s_4 ~ lognormal(-0.35, 0.55);
  s_5 ~ lognormal(-0.35, 0.55);
  s_6 ~ lognormal(-0.35, 0.55);
  //scaled aleatory var
  s_3_scl ~ lognormal(-1.50, 0.40);
    
  //magnitude scaling
  for (i in 1:NEQ) {
    tau0r_array[i]  = interp(s_1mag, s_2mag, s_1r[regeq[i]], s_2, mag[i]);
    phi0r_array[i]  = interp(s_5mag, s_6mag, s_5r[regeq[i]], s_6r[regeq[i]], mag[i]);
  }
  
  //regional coefficient adjustments 
  lambda_1r ~ normal(1., 0.15);
  lambda_3r ~ normal(1., 0.15);
  lambda_7r ~ normal(1., 0.15);
  lambda_8r ~ normal(1., 0.15);

  //regional aleatory std adjustments 
  lambda_tau0r ~ normal(1., 0.15);
  lambda_tauPr ~ normal(1., 0.15);
  lambda_phiSr ~ normal(1., 0.15);
  lambda_phi0r ~ normal(1., 0.15);
  
  //evaluate likelihood
  //------------------------------------
  //random effects
  deltaB  ~ normal(0,  tau0r_array);
  deltaS  ~ normal(0., s_4r[regst]);
  //scaled parameters
  deltaBP_scl ~ normal(0., s_3r_scl[regeq]);  
  //noise
  Y ~ normal(f_gmm + (deltaB[eq] + f_dBP .* deltaBP_scl[eq] + deltaS[st]), phi0r_array[eq]);
}

generated quantities {
  //fixed parameters
  //------------------------------------
  real c_2   = c_2fxd;
  real c_5   = c_5fxd;
  real c_6   = c_6fxd;
  real c_n   = c_nfxd;
  real c_hm  = c_hmfxd;
  real c_mag = c_magfxd;  
  
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

  //aleatory variability
  //------------------------------------
  real tau   = s_1 ;
  real tau_P = s_3;
  real phi_S = s_4;
  real phi   = s_5;
  //regionalized aleatory std
  vector[NREG] tau0r = s_1r;
  vector[NREG] tauPr = s_3r;
  vector[NREG] phiSr = s_4r;
  vector[NREG] phi0r = s_5r;
  //scaled regionalized aleatory std
  vector[NREG] tauPr_scl = s_3r_scl;
  
  //ground motion components
  //------------------------------------
  vector[N] f_gmm_src = gmm_src(N, reg, eq, st,
                                mag,
                                f_smag, f_ztor, f_r, f_n, f_hw,
                                c_2fxd, c_9, c_10a, c_10b, c_13,
                                c_3r);

  vector[N] f_gmm_path = gmm_path(N, reg, eq, st,
                                  f_gs, f_gs_lrup, f_atten, 
                                  c_4,
                                  c_7r_scl);
  
  vector[N] f_gmm_site = gmm_site(N, reg, eq, st,
                                  f_vs30,
                                  c_8r);

  //within event-site residuals
  //------------------------------------
  vector[N] deltaWS = Y - f_gmm - (deltaB[eq] + f_dBP .* deltaBP_scl[eq] + deltaS[st]);
}


