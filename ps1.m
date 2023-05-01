%%%%% Topics in Econometrics %%%%%
%%%%% GMM under Misspecification %%%%%
%%%%% Problem Set #1 %%%%%

% x = -2:0.25:2;
% y = x;
% [X,Y] = meshgrid(x);
% 
% F = X.*exp(-X.^2-Y.^2);
% surf(X,Y,F)


%% 1. Variations in rho

% Initialize parameters
rho = -0.99:0.1:0.99;
N = 500;

% 1-1. Under correct specification : delta = 0
delta = 0;
mu = [delta,0];

% Vectors for estimates
theta1 = zeros(length(rho),1);
theta2 = zeros(length(rho),1);
theta3 = zeros(length(rho),1);
theta_it = zeros(length(rho),1);


% Simulation with default seed

for i = 1:length(rho)
    sigma = [[1,rho(1,i)];[rho(1,i),1]];
    rng(1);
    X = mvnrnd(mu,sigma,N); % Generate random sample from Bivariate Normal
    
    % Intermediate statistics
    Xbar = sum(X,1)/N;
    Ybar = Xbar(1,1);
    Zbar = Xbar(1,2);
    Sighat = (X-Xbar)'*(X-Xbar)/N;
    sigY = Sighat(1,1);
    sigZ = Sighat(2,2);
    sigZY = Sighat(1,2);

    % One step GMM estimation(identity)
    theta1(i,1) = Zbar;

    % Two step GMM estimation
    theta2(i,1) = Zbar-((sigZY/(sigY+Ybar^2))*Ybar);

    % Three-step GMM estimation
    theta3(i,1) = Zbar-(((sigZY*(sigY+2*Ybar^2)))/((sigY+Ybar^2)^2))*Ybar;

    % Iterated GMM estimation
    theta_it(i,1) = Zbar - ((sigZY/sigY)*Ybar);

end

figure
plot(rho,theta1,rho,theta2,rho,theta_it)
legend('theta1','theta2','theta^{it}')


% 1-2. Under Misspecification : delta = 0.05

% Initialize parameters
delta = 0.05;
mu = [delta,0];

% Simulation with default seed

for i = 1:length(rho)
    sigma = [[1,rho(1,i)];[rho(1,i),1]];
    rng(1);
    X = mvnrnd(mu,sigma,N); % Generate random sample from Bivariate Normal
    
    % Intermediate statistics
    Xbar = sum(X,1)/N;
    Ybar = Xbar(1,1);
    Zbar = Xbar(1,2);
    Sighat = (X-Xbar)'*(X-Xbar)/N;
    sigY = Sighat(1,1);
    sigZ = Sighat(2,2);
    sigZY = Sighat(1,2);
    % One step GMM estimation(identity)
    theta1(i,1) = Zbar;

    % Two step GMM estimation
    theta2(i,1) = Zbar-((sigZY/(sigY+Ybar^2))*Ybar);

    % Iterated GMM estimation
    theta_it(i,1) = Zbar - ((sigZY/sigY)*Ybar);

end

figure
plot(rho,theta1,rho,theta2,rho,theta_it)
legend('theta1','theta2','theta^{it}')
    
% 1-3. Under Misspecification : delta = 0.99

% Initialize parameters
delta = 0.099;
mu = [delta,0];

% Simulation with default seed

for i = 1:length(rho)
    sigma = [[1,rho(1,i)];[rho(1,i),1]];
    rng(1);
    X = mvnrnd(mu,sigma,N); % Generate random sample from Bivariate Normal
    
    % Intermediate statistics
    Xbar = sum(X,1)/N;
    Ybar = Xbar(1,1);
    Zbar = Xbar(1,2);
    Sighat = (X-Xbar)'*(X-Xbar)/N;
    sigY = Sighat(1,1);
    sigZ = Sighat(2,2);
    sigZY = Sighat(1,2);
    % One step GMM estimation(identity)
    theta1(i,1) = Zbar;

    % Two step GMM estimation
    theta2(i,1) = Zbar-((sigZY/(sigY+Ybar^2))*Ybar);

    % Iterated GMM estimation
    theta_it(i,1) = Zbar - ((sigZY/sigY)*Ybar);

end

figure
plot(rho,theta1,rho,theta2,rho,theta_it)
legend('theta1','theta2','theta^{it}')

% 1-4. Under Misspecification : delta = 2

% Initialize parameters
delta = 1;
mu = [delta,0];

% Simulation with default seed

for i = 1:length(rho)
    sigma = [[1,rho(1,i)];[rho(1,i),1]];
    rng(1);
    X = mvnrnd(mu,sigma,N); % Generate random sample from Bivariate Normal
    
    % Intermediate statistics
    Xbar = sum(X,1)/N;
    Ybar = Xbar(1,1);
    Zbar = Xbar(1,2);
    Sighat = (X-Xbar)'*(X-Xbar)/N;
    sigY = Sighat(1,1);
    sigZ = Sighat(2,2);
    sigZY = Sighat(1,2);
    % One step GMM estimation(identity)
    theta1(i,1) = Zbar;

    % Two step GMM estimation
    theta2(i,1) = Zbar-((sigZY/(sigY+Ybar^2))*Ybar);

    % Iterated GMM estimation
    theta_it(i,1) = Zbar - ((sigZY/sigY)*Ybar);

end

figure
plot(rho,theta1,rho,theta2,rho,theta_it)
legend('theta1','theta2','theta^{it}')



%% 2. Variation in delta

% 2-1. rho = 0 (No correlation)
rho = 0;
sigma = [[1,rho];[rho,1]];

N = 5000;
delta = -2:0.25:2;


% Vectors for estimates
theta1 = zeros(length(delta),1);
theta2 = zeros(length(delta),1);
theta_it = zeros(length(delta),1);

% Simulation with default seed

for i = 1:length(delta)
    mu = [delta(1,i),0];
    rng(1);
    X = mvnrnd(mu,sigma,N); % Generate random sample from Bivariate Normal
    
    % Intermediate statistics
    Xbar = sum(X,1)/N;
    Ybar = Xbar(1,1);
    Zbar = Xbar(1,2);
    Sighat = (X-Xbar)'*(X-Xbar)/N;
    sigY = Sighat(1,1);
    sigZ = Sighat(2,2);
    sigZY = Sighat(1,2);
    % One step GMM estimation(identity)
    theta1(i,1) = Zbar;

    % Two step GMM estimation
    theta2(i,1) = Zbar-((sigZY/(sigY+Ybar^2))*Ybar);

    % Iterated GMM estimation
    theta_it(i,1) = Zbar - ((sigZY/sigY)*Ybar);

end

figure
plot(delta,theta1,delta,theta2,delta,theta_it)
legend('theta1','theta2','theta^{it}')

% 2-2. rho = 0.5 (Intermediate correlation)
rho = 0.5;
sigma = [[1,rho];[rho,1]];

% Simulation with default seed

for i = 1:length(delta)
    mu = [delta(1,i),0];
    rng(1);
    X = mvnrnd(mu,sigma,N); % Generate random sample from Bivariate Normal
    
    % Intermediate statistics
    Xbar = sum(X,1)/N;
    Ybar = Xbar(1,1);
    Zbar = Xbar(1,2);
    Sighat = (X-Xbar)'*(X-Xbar)/N;
    sigY = Sighat(1,1);
    sigZ = Sighat(2,2);
    sigZY = Sighat(1,2);
    % One step GMM estimation(identity)
    theta1(i,1) = Zbar;

    % Two step GMM estimation
    theta2(i,1) = Zbar-((sigZY/(sigY+Ybar^2))*Ybar);

    % Iterated GMM estimation
    theta_it(i,1) = Zbar - ((sigZY/sigY)*Ybar);

end

figure
plot(delta,theta1,delta,theta2,delta,theta_it)
legend('theta1','theta2','theta^{it}')

% 2-3. rho = 0.9 (Strong correlation)
rho = 0.9;
sigma = [[1,rho];[rho,1]];

% Simulation with default seed

for i = 1:length(delta)
    mu = [delta(1,i),0];
    rng(1);
    X = mvnrnd(mu,sigma,N); % Generate random sample from Bivariate Normal
    
    % Intermediate statistics
    Xbar = sum(X,1)/N;
    Ybar = Xbar(1,1);
    Zbar = Xbar(1,2);
    Sighat = (X-Xbar)'*(X-Xbar)/N;
    sigY = Sighat(1,1);
    sigZ = Sighat(2,2);
    sigZY = Sighat(1,2);
    % One step GMM estimation(identity)
    theta1(i,1) = Zbar;

    % Two step GMM estimation
    theta2(i,1) = Zbar-((sigZY/(sigY+Ybar^2))*Ybar);

    % Iterated GMM estimation
    theta_it(i,1) = Zbar - ((sigZY/sigY)*Ybar);

end

figure
plot(delta,theta1,delta,theta2,delta,theta_it)
legend('theta1','theta2','theta^{it}')


%% 3. Variation in N

% 3-1. Correct Specification with valid info : delta = 0, rho = 0.9
delta = 0;
mu = [delta,0];
rho = 0.9;
sigma = [[1,rho];[rho,1]];
N = 5:20:1000;

% Vectors for estimates
theta1 = zeros(length(N),1);
theta2 = zeros(length(N),1);
theta_it = zeros(length(N),1);

% Simulation with default seed

for i = 1:length(N)
    
    rng(1);
    X = mvnrnd(mu,sigma,N(1,i)); % Generate random sample from Bivariate Normal
    
    % Intermediate statistics
    Xbar = sum(X,1)/N(1,i);
    Ybar = Xbar(1,1);
    Zbar = Xbar(1,2);
    Sighat = (X-Xbar)'*(X-Xbar)/N(1,i);
    sigY = Sighat(1,1);
    sigZ = Sighat(2,2);
    sigZY = Sighat(1,2);
    % One step GMM estimation(identity)
    theta1(i,1) = Zbar;

    % Two step GMM estimation
    theta2(i,1) = Zbar-((sigZY/(sigY+Ybar^2))*Ybar);

    % Iterated GMM estimation
    theta_it(i,1) = Zbar - ((sigZY/sigY)*Ybar);

end

figure
plot(N,theta1,N,theta2,N,theta_it)
legend('theta1','theta2','theta^{it}')


% 3-2. Under Misspecification : delta = 0.99 , rho = 0
delta = 0.99;
mu = [delta,0];
rho = 0;
sigma = [[1,rho];[rho,1]];

% Vectors for estimates
theta1 = zeros(length(N),1);
theta2 = zeros(length(N),1);
theta_it = zeros(length(N),1);

% Simulation with default seed

for i = 1:length(N)
    
    rng(1);
    X = mvnrnd(mu,sigma,N(1,i)); % Generate random sample from Bivariate Normal
    
    % Intermediate statistics
    Xbar = sum(X,1)/N(1,i);
    Ybar = Xbar(1,1);
    Zbar = Xbar(1,2);
    Sighat = (X-Xbar)'*(X-Xbar)/N(1,i);
    sigY = Sighat(1,1);
    sigZ = Sighat(2,2);
    sigZY = Sighat(1,2);
    % One step GMM estimation(identity)
    theta1(i,1) = Zbar;

    % Two step GMM estimation
    theta2(i,1) = Zbar-((sigZY/(sigY+Ybar^2))*Ybar);

    % Iterated GMM estimation
    theta_it(i,1) = Zbar - ((sigZY/sigY)*Ybar);

end

figure
plot(N,theta1,N,theta2,N,theta_it)
legend('theta1','theta2','theta^{it}')


% 3-3. Under Misspecification : delta = 0.99 , rho = 0.5
delta = 0.99;
mu = [delta,0];
rho = 0.5;
sigma = [[1,rho];[rho,1]];

% Vectors for estimates
theta1 = zeros(length(N),1);
theta2 = zeros(length(N),1);
theta_it = zeros(length(N),1);

% Simulation with default seed

for i = 1:length(N)
    
    rng(1);
    X = mvnrnd(mu,sigma,N(1,i)); % Generate random sample from Bivariate Normal
    
    % Intermediate statistics
    Xbar = sum(X,1)/N(1,i);
    Ybar = Xbar(1,1);
    Zbar = Xbar(1,2);
    Sighat = (X-Xbar)'*(X-Xbar)/N(1,i);
    sigY = Sighat(1,1);
    sigZ = Sighat(2,2);
    sigZY = Sighat(1,2);
    % One step GMM estimation(identity)
    theta1(i,1) = Zbar;

    % Two step GMM estimation
    theta2(i,1) = Zbar-((sigZY/(sigY+Ybar^2))*Ybar);

    % Iterated GMM estimation
    theta_it(i,1) = Zbar - ((sigZY/sigY)*Ybar);

end

figure
plot(N,theta1,N,theta2,N,theta_it)
legend('theta1','theta2','theta^{it}')


% 3-4. Under Misspecification : delta = 0.3 , rho = 0.5
delta = 0.3;
mu = [delta,0];
rho = 0.5;
sigma = [[1,rho];[rho,1]];

% Vectors for estimates
theta1 = zeros(length(N),1);
theta2 = zeros(length(N),1);
theta_it = zeros(length(N),1);

% Simulation with default seed

for i = 1:length(N)
    
    rng(1);
    X = mvnrnd(mu,sigma,N(1,i)); % Generate random sample from Bivariate Normal
    
    % Intermediate statistics
    Xbar = sum(X,1)/N(1,i);
    Ybar = Xbar(1,1);
    Zbar = Xbar(1,2);
    Sighat = (X-Xbar)'*(X-Xbar)/N(1,i);
    sigY = Sighat(1,1);
    sigZ = Sighat(2,2);
    sigZY = Sighat(1,2);
    % One step GMM estimation(identity)
    theta1(i,1) = Zbar;

    % Two step GMM estimation
    theta2(i,1) = Zbar-((sigZY/(sigY+Ybar^2))*Ybar);

    % Iterated GMM estimation
    theta_it(i,1) = Zbar - ((sigZY/sigY)*Ybar);

end

figure
plot(N,theta1,N,theta2,N,theta_it)
legend('theta1','theta2','theta^{it}')


%% 


