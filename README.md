# GMM
Hi! This is a sandbox for studying GMM!!
I will upload coding files implementing GMM estimations described in Hansen's Econometrics and some seminal papers!! 
Also I am going to upload simulation codings to explrore/discover some useful features of GMM!!

## File description

### 3-1 GMM class
#### Problem Set 1
'ps1.m' and 'ps1_1.m' are the main code files. They are implementing the Toy Example monte carlo simulations with different correlation parameter \rho, misspecification parameter \delta and sample size n.

#### Problem Set 2
'ps2.m', 'ps2_with_GMMboot...', 'ps2_with_repAEsubsample.m' are essentially the same code files, but using self-defined functions 'GMMbootstrap.m' and 'repAEsubsample.m' respectively. I separated these function files to improve computation speed.
'VarMR.m' and 'VarMR.cpp' are codes to implement M-R variance with improved performance(but failed - I leave them for future work)
'untitled.m' : I don't know for what this is.

#### Term paper codes and files
'termpaper.m' is the main code implementing Monte Carlo simulation for the Toy Example introduced in Lee(2014,JoE) with averaging Gmm estimator.
'termpaper_sim_result.mat' is a simulation result(but this result is wrong because I used wrong weight \omega).
