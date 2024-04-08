## STAN Regression Scripts
All following scripts partition total residuals into between-event, between-path, between-site, and within-event-site
### Unconstrained Regression
 * `partition_residuals_eq_path_unconstrained_v0.stan`: uses log-increment density
 * `partition_residuals_eq_path_unconstrained_v1.stan`: samples normalized random terms out of the prior distributions
 * `partition_residuals_eq_path_unconstrained_v2.stan`: samples scaled random terms (magnitude dependent tau0, tauP, phiS, and phi0)  
### Regression w\ Hard Constraints
 * `partition_residuals_eq_path_hard_constraint_v0.stan`: uses log-increment density
 * `partition_residuals_eq_path_hard_constraint_v1.stan`: samples normalized random terms out of the prior distributions
### Regression w\ Soft Constraints
 * `partition_residuals_eq_path_soft_constraint_v0.stan`: uses log-increment density
 * `partition_residuals_eq_path_soft_constraint_v1.stan`: samples normalized random terms out of the prior distributions

