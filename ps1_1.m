%% 1. Variations in rho

% Initialize parameters
rho = -0.99:0.01:0.99;
N = 200;
B = 5000;
G = [0;-1];
C = 1.96;

% Vectors for statistics
mean_theta = zeros(4,length(rho));
sd_theta = zeros(4,length(rho));
cp_theta = zeros(4,length(rho));

% 1-1. Under correct specification : delta = 0
delta = 0;
mu = [delta,0];

for i = 1:length(rho)

    % Simulation with default seed
    rng('default');
    
    % Var-cov matrix for each rho value
    sigma = [[1,rho(1,i)];[rho(1,i),1]];

    % Vectors for estimates
    theta = zeros(B,4);
    se = zeros(B,4);
    isCov = zeros(B,4);

    for j = 1:B
        X = mvnrnd(mu,sigma,N); % Generate random sample from Bivariate Normal

        % Intermediate variables
        Xbar = sum(X,1)/N;
        Ybar = Xbar(1,1);
        Zbar = Xbar(1,2);

        Sighat = (X-Xbar)'*(X-Xbar)/N;
        sigY = Sighat(1,1);
        sigZ = Sighat(2,2);
        sigZY = Sighat(1,2);

        omega1 = [[sigY+Ybar^2,sigZY];[sigZY,sigZ]];
        W1 = eye(2)/omega1;
        omega2 = [[sigY+Ybar^2,sigZY+((sigZY*Ybar^2)/(sigY+Ybar^2))];
            [sigZY+((sigZY*Ybar^2)/(sigY+Ybar^2)),sigZ+((sigZY*Ybar)/(sigY+Ybar^2))^2]];
        W2 = eye(2)/omega2;
    
        % One step GMM estimation(identity)
        theta(j,1) = Zbar;
        se(j,1) = sqrt((G'*omega1*G)/N);
        isCov(j,1) = (theta(j,1)-C*se(j,1)<=0) & (theta(j,1)+C*se(j,1)>=0);
    
        % Two step GMM estimation
        theta(j,2) = Zbar-((sigZY/(sigY+Ybar^2))*Ybar);
        se(j,2) = sqrt((inv(G'*W1*G))/N);
        isCov(j,2) = (theta(j,2)-C*se(j,2)<=0) & (theta(j,2)+C*se(j,2)>=0);
    
        % Three-step GMM estimation
        theta(j,3) = Zbar-(((sigZY*(sigY+2*Ybar^2)))/((sigY+Ybar^2)^2))*Ybar;
        se(j,3) = sqrt((inv(G'*W2*G))/N);
        isCov(j,3) = (theta(j,3)-C*se(j,3)<=0) & (theta(j,3)+C*se(j,3)>=0);
    
        % Iterated GMM estimation
        theta(j,4) = Zbar - ((sigZY/sigY)*Ybar);

        omega_it = [[sigY+Ybar^2,sigZY+(Ybar*(Zbar-theta(j,4)))];
            [sigZY+(Ybar*(Zbar-theta(j,4))), sigZ+(Zbar-theta(j,4))^2]];
        W_it = eye(2)/omega_it;
        se(j,4) = sqrt((inv(G'*W_it*G))/N);
        isCov(j,4) = (theta(j,4)-C*se(j,4)<=0) & (theta(j,4)+C*se(j,4)>=0);
    end

    mean_theta(:,i) = (sum(theta,1)/B)';
    sd_theta(1,i) = sqrt((sum(((theta(:,1)-mean_theta(1,i)).^2),1))/(B-1));
    sd_theta(2,i) = sqrt((sum(((theta(:,2)-mean_theta(2,i)).^2),1))/(B-1));
    sd_theta(3,i) = sqrt((sum(((theta(:,3)-mean_theta(3,i)).^2),1))/(B-1));
    sd_theta(4,i) = sqrt((sum(((theta(:,4)-mean_theta(4,i)).^2),1))/(B-1));
    cp_theta(:,i) = (sum(isCov,1)/B)';
end

figure
tiledlayout(3,1)
zero = zeros(1,length(rho));

ax1 = nexttile;
plot(ax1,rho,mean_theta(1,:),':', ...
    rho,mean_theta(2,:),'.', ...
    rho,mean_theta(3,:),'--', ...
    rho,mean_theta(4,:), ...
    rho,zero)
title(ax1,'Bias; \delta = 0')
ylabel(ax1,'bias')
xlabel(ax1,'\rho')

ax2 = nexttile;
plot(ax2,rho,sd_theta(1,:),':', ...
    rho,sd_theta(2,:),'.', ...
    rho,sd_theta(3,:),'--', ...
    rho,sd_theta(4,:))
title(ax2,'Standard Deviation')
ylabel(ax2,'sd')
xlabel(ax2,'\rho')


ax3 = nexttile;
plot(ax3,rho,cp_theta(1,:),':', ...
    rho,cp_theta(2,:),'.', ...
    rho,cp_theta(3,:),'--', ...
    rho,cp_theta(4,:))
title(ax3,'Coverage Probability of 95% CI')
ylabel(ax3,'coverage probability')
xlabel(ax3,'\rho')
legend('theta_1','theta_2','theta_3','theta_{it}')

%% 1-2. Under moderate misspecification : delta = 0.3
delta = 0.3;
mu = [delta,0];

% Initialize parameters
rho = -0.99:0.01:0.99;
N = 200;
B = 5000;
G = [0;-1];
C = 1.96;

% Vectors for statistics
mean_theta = zeros(4,length(rho));
sd_theta = zeros(4,length(rho));
cp_theta = zeros(4,length(rho));


for i = 1:length(rho)

    % Simulation with default seed
    rng('default');
    
    % Var-cov matrix for each rho value
    sigma = [[1,rho(1,i)];[rho(1,i),1]];

    % Vectors for estimates
    theta = zeros(B,4);
    se = zeros(B,4);
    isCov = zeros(B,4);

    for j = 1:B
        X = mvnrnd(mu,sigma,N); % Generate random sample from Bivariate Normal

        % Intermediate variables
        Xbar = sum(X,1)/N;
        Ybar = Xbar(1,1);
        Zbar = Xbar(1,2);

        Sighat = (X-Xbar)'*(X-Xbar)/N;
        sigY = Sighat(1,1);
        sigZ = Sighat(2,2);
        sigZY = Sighat(1,2);

        omega1 = [[sigY+Ybar^2,sigZY];[sigZY,sigZ]];
        W1 = eye(2)/omega1;
        omega2 = [[sigY+Ybar^2,sigZY+((sigZY*Ybar^2)/(sigY+Ybar^2))];
            [sigZY+((sigZY*Ybar^2)/(sigY+Ybar^2)),sigZ+((sigZY*Ybar)/(sigY+Ybar^2))^2]];
        W2 = eye(2)/omega2;
    
        % One step GMM estimation(identity)
        theta(j,1) = Zbar;
        se(j,1) = sqrt((G'*omega1*G)/N);
        isCov(j,1) = (theta(j,1)-C*se(j,1)<=0) & (theta(j,1)+C*se(j,1)>=0);
    
        % Two step GMM estimation
        theta(j,2) = Zbar-((sigZY/(sigY+Ybar^2))*Ybar);
        se(j,2) = sqrt((inv(G'*W1*G))/N);
        isCov(j,2) = (theta(j,2)-C*se(j,2)<=0) & (theta(j,2)+C*se(j,2)>=0);
    
        % Three-step GMM estimation
        theta(j,3) = Zbar-(((sigZY*(sigY+2*Ybar^2)))/((sigY+Ybar^2)^2))*Ybar;
        se(j,3) = sqrt((inv(G'*W2*G))/N);
        isCov(j,3) = (theta(j,3)-C*se(j,3)<=0) & (theta(j,3)+C*se(j,3)>=0);
    
        % Iterated GMM estimation
        theta(j,4) = Zbar - ((sigZY/sigY)*Ybar);

        omega_it = [[sigY+Ybar^2,sigZY+(Ybar*(Zbar-theta(j,4)))];
            [sigZY+(Ybar*(Zbar-theta(j,4))), sigZ+(Zbar-theta(j,4))^2]];
        W_it = eye(2)/omega_it;
        se(j,4) = sqrt((inv(G'*W_it*G))/N);
        isCov(j,4) = (theta(j,4)-C*se(j,4)<=0) & (theta(j,4)+C*se(j,4)>=0);
    end

    mean_theta(:,i) = (sum(theta,1)/B)';
    sd_theta(1,i) = sqrt((sum(((theta(:,1)-mean_theta(1,i)).^2),1))/(B-1));
    sd_theta(2,i) = sqrt((sum(((theta(:,2)-mean_theta(2,i)).^2),1))/(B-1));
    sd_theta(3,i) = sqrt((sum(((theta(:,3)-mean_theta(3,i)).^2),1))/(B-1));
    sd_theta(4,i) = sqrt((sum(((theta(:,4)-mean_theta(4,i)).^2),1))/(B-1));
    cp_theta(:,i) = (sum(isCov,1)/B)';
end

figure
tiledlayout(3,1)
zero = zeros(1,length(rho));

ax1 = nexttile;
plot(ax1,rho,mean_theta(1,:),':', ...
    rho,mean_theta(2,:),'.', ...
    rho,mean_theta(3,:),'--', ...
    rho,mean_theta(4,:), ...
    rho,zero)
title(ax1,'Bias; \delta = 0')
ylabel(ax1,'bias')
xlabel(ax1,'\rho')

ax2 = nexttile;
plot(ax2,rho,sd_theta(1,:),':', ...
    rho,sd_theta(2,:),'.', ...
    rho,sd_theta(3,:),'--', ...
    rho,sd_theta(4,:))
title(ax2,'Standard Deviation')
ylabel(ax2,'sd')
xlabel(ax2,'\rho')


ax3 = nexttile;
plot(ax3,rho,cp_theta(1,:),':', ...
    rho,cp_theta(2,:),'.', ...
    rho,cp_theta(3,:),'--', ...
    rho,cp_theta(4,:))
title(ax3,'Coverage Probability of 95% CI')
ylabel(ax3,'coverage probability')
xlabel(ax3,'\rho')
legend('theta_1','theta_2','theta_3','theta_{it}')


%% 1-3. Under large misspecification : delta = 0.6
delta = 0.6;
mu = [delta,0];

% Initialize parameters
rho = -0.99:0.01:0.99;
N = 200;
B = 5000;
G = [0;-1];
C = 1.96;

% Vectors for statistics
mean_theta = zeros(4,length(rho));
sd_theta = zeros(4,length(rho));
cp_theta = zeros(4,length(rho));


for i = 1:length(rho)

    % Simulation with default seed
    rng('default');
    
    % Var-cov matrix for each rho value
    sigma = [[1,rho(1,i)];[rho(1,i),1]];

    % Vectors for estimates
    theta = zeros(B,4);
    se = zeros(B,4);
    isCov = zeros(B,4);

    for j = 1:B
        X = mvnrnd(mu,sigma,N); % Generate random sample from Bivariate Normal

        % Intermediate variables
        Xbar = sum(X,1)/N;
        Ybar = Xbar(1,1);
        Zbar = Xbar(1,2);

        Sighat = (X-Xbar)'*(X-Xbar)/N;
        sigY = Sighat(1,1);
        sigZ = Sighat(2,2);
        sigZY = Sighat(1,2);

        omega1 = [[sigY+Ybar^2,sigZY];[sigZY,sigZ]];
        W1 = eye(2)/omega1;
        omega2 = [[sigY+Ybar^2,sigZY+((sigZY*Ybar^2)/(sigY+Ybar^2))];
            [sigZY+((sigZY*Ybar^2)/(sigY+Ybar^2)),sigZ+((sigZY*Ybar)/(sigY+Ybar^2))^2]];
        W2 = eye(2)/omega2;
    
        % One step GMM estimation(identity)
        theta(j,1) = Zbar;
        se(j,1) = sqrt((G'*omega1*G)/N);
        isCov(j,1) = (theta(j,1)-C*se(j,1)<=0) & (theta(j,1)+C*se(j,1)>=0);
    
        % Two step GMM estimation
        theta(j,2) = Zbar-((sigZY/(sigY+Ybar^2))*Ybar);
        se(j,2) = sqrt((inv(G'*W1*G))/N);
        isCov(j,2) = (theta(j,2)-C*se(j,2)<=0) & (theta(j,2)+C*se(j,2)>=0);
    
        % Three-step GMM estimation
        theta(j,3) = Zbar-(((sigZY*(sigY+2*Ybar^2)))/((sigY+Ybar^2)^2))*Ybar;
        se(j,3) = sqrt((inv(G'*W2*G))/N);
        isCov(j,3) = (theta(j,3)-C*se(j,3)<=0) & (theta(j,3)+C*se(j,3)>=0);
    
        % Iterated GMM estimation
        theta(j,4) = Zbar - ((sigZY/sigY)*Ybar);

        omega_it = [[sigY+Ybar^2,sigZY+(Ybar*(Zbar-theta(j,4)))];
            [sigZY+(Ybar*(Zbar-theta(j,4))), sigZ+(Zbar-theta(j,4))^2]];
        W_it = eye(2)/omega_it;
        se(j,4) = sqrt((inv(G'*W_it*G))/N);
        isCov(j,4) = (theta(j,4)-C*se(j,4)<=0) & (theta(j,4)+C*se(j,4)>=0);
    end

    mean_theta(:,i) = (sum(theta,1)/B)';
    sd_theta(1,i) = sqrt((sum(((theta(:,1)-mean_theta(1,i)).^2),1))/(B-1));
    sd_theta(2,i) = sqrt((sum(((theta(:,2)-mean_theta(2,i)).^2),1))/(B-1));
    sd_theta(3,i) = sqrt((sum(((theta(:,3)-mean_theta(3,i)).^2),1))/(B-1));
    sd_theta(4,i) = sqrt((sum(((theta(:,4)-mean_theta(4,i)).^2),1))/(B-1));
    cp_theta(:,i) = (sum(isCov,1)/B)';
end

figure
tiledlayout(3,1)
zero = zeros(1,length(rho));

ax1 = nexttile;
plot(ax1,rho,mean_theta(1,:),':', ...
    rho,mean_theta(2,:),'.', ...
    rho,mean_theta(3,:),'--', ...
    rho,mean_theta(4,:), ...
    rho,zero)
title(ax1,'Bias; \delta = 0')
ylabel(ax1,'bias')
xlabel(ax1,'\rho')

ax2 = nexttile;
plot(ax2,rho,sd_theta(1,:),':', ...
    rho,sd_theta(2,:),'.', ...
    rho,sd_theta(3,:),'--', ...
    rho,sd_theta(4,:))
title(ax2,'Standard Deviation')
ylabel(ax2,'sd')
xlabel(ax2,'\rho')


ax3 = nexttile;
plot(ax3,rho,cp_theta(1,:),':', ...
    rho,cp_theta(2,:),'.', ...
    rho,cp_theta(3,:),'--', ...
    rho,cp_theta(4,:))
title(ax3,'Coverage Probability of 95% CI')
ylabel(ax3,'coverage probability')
xlabel(ax3,'\rho')
legend('theta_1','theta_2','theta_3','theta_{it}')

%% 2. Variation in delta

% Initialize parameters
%delta = -1:0.05:1;
delta = -5:0.05:5;
N = 200;
B = 5000;
G = [0;-1];
C = 1.96;

% Vectors for statistics
mean_theta = zeros(4,length(delta));
sd_theta = zeros(4,length(delta));
cp_theta = zeros(4,length(delta));

% 2-1. Under High correlation : rho = 0.9
rho = 0.9;
sigma = [[1,rho];[rho,1]];

for i = 1:length(delta)

    % Simulation with default seed
    rng('default');
    
    % mu vector for each delta value
    mu = [delta(1,i),0];

    % Vectors for estimates
    theta = zeros(B,4);
    se = zeros(B,4);
    isCov = zeros(B,4);

    for j = 1:B
        X = mvnrnd(mu,sigma,N); % Generate random sample from Bivariate Normal

        % Intermediate variables
        Xbar = sum(X,1)/N;
        Ybar = Xbar(1,1);
        Zbar = Xbar(1,2);

        Sighat = (X-Xbar)'*(X-Xbar)/N;
        sigY = Sighat(1,1);
        sigZ = Sighat(2,2);
        sigZY = Sighat(1,2);

        omega1 = [[sigY+Ybar^2,sigZY];[sigZY,sigZ]];
        W1 = eye(2)/omega1;
        omega2 = [[sigY+Ybar^2,sigZY+((sigZY*Ybar^2)/(sigY+Ybar^2))];
            [sigZY+((sigZY*Ybar^2)/(sigY+Ybar^2)),sigZ+((sigZY*Ybar)/(sigY+Ybar^2))^2]];
        W2 = eye(2)/omega2;
    
        % One step GMM estimation(identity)
        theta(j,1) = Zbar;
        se(j,1) = sqrt((G'*omega1*G)/N);
        isCov(j,1) = (theta(j,1)-C*se(j,1)<=0) & (theta(j,1)+C*se(j,1)>=0);
    
        % Two step GMM estimation
        theta(j,2) = Zbar-((sigZY/(sigY+Ybar^2))*Ybar);
        se(j,2) = sqrt((inv(G'*W1*G))/N);
        isCov(j,2) = (theta(j,2)-C*se(j,2)<=0) & (theta(j,2)+C*se(j,2)>=0);
    
        % Three-step GMM estimation
        theta(j,3) = Zbar-(((sigZY*(sigY+2*Ybar^2)))/((sigY+Ybar^2)^2))*Ybar;
        se(j,3) = sqrt((inv(G'*W2*G))/N);
        isCov(j,3) = (theta(j,3)-C*se(j,3)<=0) & (theta(j,3)+C*se(j,3)>=0);
    
        % Iterated GMM estimation
        theta(j,4) = Zbar - ((sigZY/sigY)*Ybar);

        omega_it = [[sigY+Ybar^2,sigZY+(Ybar*(Zbar-theta(j,4)))];
            [sigZY+(Ybar*(Zbar-theta(j,4))), sigZ+(Zbar-theta(j,4))^2]];
        W_it = eye(2)/omega_it;
        se(j,4) = sqrt((inv(G'*W_it*G))/N);
        isCov(j,4) = (theta(j,4)-C*se(j,4)<=0) & (theta(j,4)+C*se(j,4)>=0);
    end

    mean_theta(:,i) = (sum(theta,1)/B)';
    sd_theta(1,i) = sqrt((sum(((theta(:,1)-mean_theta(1,i)).^2),1))/(B-1));
    sd_theta(2,i) = sqrt((sum(((theta(:,2)-mean_theta(2,i)).^2),1))/(B-1));
    sd_theta(3,i) = sqrt((sum(((theta(:,3)-mean_theta(3,i)).^2),1))/(B-1));
    sd_theta(4,i) = sqrt((sum(((theta(:,4)-mean_theta(4,i)).^2),1))/(B-1));
    cp_theta(:,i) = (sum(isCov,1)/B)';
end

figure
tiledlayout(3,1)
zero = zeros(1,length(delta));

ax1 = nexttile;
plot(ax1,delta,mean_theta(1,:),'.', ...
    delta,mean_theta(2,:),':', ...
    delta,mean_theta(3,:),'--', ...
    delta,mean_theta(4,:),delta,zero)
title(ax1,'Bias; \rho = 0.9')
ylabel(ax1,'bias')
xlabel(ax1,'\delta')

ax2 = nexttile;
plot(ax2,delta,sd_theta(1,:),'.', ...
    delta,sd_theta(2,:),':', ...
    delta,sd_theta(3,:),'--', ...
    delta,sd_theta(4,:))
title(ax2,'Standard Deviation')
ylabel(ax2,'sd')
xlabel(ax2,'\delta')


ax3 = nexttile;
plot(ax3,delta,cp_theta(1,:),'.', ...
    delta,cp_theta(2,:),':', ...
    delta,cp_theta(3,:),'--', ...
    delta,cp_theta(4,:))
title(ax3,'Coverage Probability of 95% CI')
ylabel(ax3,'coverage probability')
xlabel(ax3,'\delta')
legend('theta_1','theta_2','theta_3','theta_{it}')


%% 2-2. Uncorrelated case : rho = 0
% Initialize parameters
%delta = -1:0.05:1;
delta = -5:0.05:5;
N = 200;
B = 5000;
G = [0;-1];
C = 1.96;

% Vectors for statistics
mean_theta = zeros(4,length(delta));
sd_theta = zeros(4,length(delta));
cp_theta = zeros(4,length(delta));

% 2-1. Uncorrelated case : rho = 0
rho = 0;
sigma = [[1,rho];[rho,1]];

for i = 1:length(delta)

    % Simulation with default seed
    rng('default');
    
    % mu vector for each delta value
    mu = [delta(1,i),0];

    % Vectors for estimates
    theta = zeros(B,4);
    se = zeros(B,4);
    isCov = zeros(B,4);

    for j = 1:B
        X = mvnrnd(mu,sigma,N); % Generate random sample from Bivariate Normal

        % Intermediate variables
        Xbar = sum(X,1)/N;
        Ybar = Xbar(1,1);
        Zbar = Xbar(1,2);

        Sighat = (X-Xbar)'*(X-Xbar)/N;
        sigY = Sighat(1,1);
        sigZ = Sighat(2,2);
        sigZY = Sighat(1,2);

        omega1 = [[sigY+Ybar^2,sigZY];[sigZY,sigZ]];
        W1 = eye(2)/omega1;
        omega2 = [[sigY+Ybar^2,sigZY+((sigZY*Ybar^2)/(sigY+Ybar^2))];
            [sigZY+((sigZY*Ybar^2)/(sigY+Ybar^2)),sigZ+((sigZY*Ybar)/(sigY+Ybar^2))^2]];
        W2 = eye(2)/omega2;
    
        % One step GMM estimation(identity)
        theta(j,1) = Zbar;
        se(j,1) = sqrt((G'*omega1*G)/N);
        isCov(j,1) = (theta(j,1)-C*se(j,1)<=0) & (theta(j,1)+C*se(j,1)>=0);
    
        % Two step GMM estimation
        theta(j,2) = Zbar-((sigZY/(sigY+Ybar^2))*Ybar);
        se(j,2) = sqrt((inv(G'*W1*G))/N);
        isCov(j,2) = (theta(j,2)-C*se(j,2)<=0) & (theta(j,2)+C*se(j,2)>=0);
    
        % Three-step GMM estimation
        theta(j,3) = Zbar-(((sigZY*(sigY+2*Ybar^2)))/((sigY+Ybar^2)^2))*Ybar;
        se(j,3) = sqrt((inv(G'*W2*G))/N);
        isCov(j,3) = (theta(j,3)-C*se(j,3)<=0) & (theta(j,3)+C*se(j,3)>=0);
    
        % Iterated GMM estimation
        theta(j,4) = Zbar - ((sigZY/sigY)*Ybar);

        omega_it = [[sigY+Ybar^2,sigZY+(Ybar*(Zbar-theta(j,4)))];
            [sigZY+(Ybar*(Zbar-theta(j,4))), sigZ+(Zbar-theta(j,4))^2]];
        W_it = eye(2)/omega_it;
        se(j,4) = sqrt((inv(G'*W_it*G))/N);
        isCov(j,4) = (theta(j,4)-C*se(j,4)<=0) & (theta(j,4)+C*se(j,4)>=0);
    end

    mean_theta(:,i) = (sum(theta,1)/B)';
    sd_theta(1,i) = sqrt((sum(((theta(:,1)-mean_theta(1,i)).^2),1))/(B-1));
    sd_theta(2,i) = sqrt((sum(((theta(:,2)-mean_theta(2,i)).^2),1))/(B-1));
    sd_theta(3,i) = sqrt((sum(((theta(:,3)-mean_theta(3,i)).^2),1))/(B-1));
    sd_theta(4,i) = sqrt((sum(((theta(:,4)-mean_theta(4,i)).^2),1))/(B-1));
    cp_theta(:,i) = (sum(isCov,1)/B)';
end

figure
tiledlayout(3,1)
zero = zeros(1,length(delta));

ax1 = nexttile;
plot(ax1,delta,mean_theta(1,:),'.', ...
    delta,mean_theta(2,:),':', ...
    delta,mean_theta(3,:),'--', ...
    delta,mean_theta(4,:),delta,zero)
title(ax1,'Bias; \rho = 0.9')
ylabel(ax1,'bias')
xlabel(ax1,'\delta')

ax2 = nexttile;
plot(ax2,delta,sd_theta(1,:),'.', ...
    delta,sd_theta(2,:),':', ...
    delta,sd_theta(3,:),'--', ...
    delta,sd_theta(4,:))
title(ax2,'Standard Deviation')
ylabel(ax2,'sd')
xlabel(ax2,'\delta')


ax3 = nexttile;
plot(ax3,delta,cp_theta(1,:),'.', ...
    delta,cp_theta(2,:),':', ...
    delta,cp_theta(3,:),'--', ...
    delta,cp_theta(4,:))
title(ax3,'Coverage Probability of 95% CI')
ylabel(ax3,'coverage probability')
xlabel(ax3,'\delta')
legend('theta_1','theta_2','theta_3','theta_{it}')


%% 3. Variation in N

% Initialize parameters
n = 10:10:500;
B = 5000;
G = [0;-1];
C = 1.96;

% Vectors for statistics
mean_theta = zeros(4,length(n));
sd_theta = zeros(4,length(n));
cp_theta = zeros(4,length(n));

% 3-1. Under correct specification and High correlation : delta = 0, rho = 0.9
delta = 0;
mu = [delta,0];
rho = 0.9;
sigma = [[1,rho];[rho,1]];

for i = 1:length(n)

    % Simulation with default seed
    rng('default');
    
    % Number of obs for each simulation
    N = n(1,i);

    % Vectors for estimates
    theta = zeros(B,4);
    se = zeros(B,4);
    isCov = zeros(B,4);

    for j = 1:B
        X = mvnrnd(mu,sigma,N); % Generate random sample from Bivariate Normal

        % Intermediate variables
        Xbar = sum(X,1)/N;
        Ybar = Xbar(1,1);
        Zbar = Xbar(1,2);

        Sighat = (X-Xbar)'*(X-Xbar)/N;
        sigY = Sighat(1,1);
        sigZ = Sighat(2,2);
        sigZY = Sighat(1,2);

        omega1 = [[sigY+Ybar^2,sigZY];[sigZY,sigZ]];
        W1 = eye(2)/omega1;
        omega2 = [[sigY+Ybar^2,sigZY+((sigZY*Ybar^2)/(sigY+Ybar^2))];
            [sigZY+((sigZY*Ybar^2)/(sigY+Ybar^2)),sigZ+((sigZY*Ybar)/(sigY+Ybar^2))^2]];
        W2 = eye(2)/omega2;
    
        % One step GMM estimation(identity)
        theta(j,1) = Zbar;
        se(j,1) = sqrt((G'*omega1*G)/N);
        isCov(j,1) = (theta(j,1)-C*se(j,1)<=0) & (theta(j,1)+C*se(j,1)>=0);
    
        % Two step GMM estimation
        theta(j,2) = Zbar-((sigZY/(sigY+Ybar^2))*Ybar);
        se(j,2) = sqrt((inv(G'*W1*G))/N);
        isCov(j,2) = (theta(j,2)-C*se(j,2)<=0) & (theta(j,2)+C*se(j,2)>=0);
    
        % Three-step GMM estimation
        theta(j,3) = Zbar-(((sigZY*(sigY+2*Ybar^2)))/((sigY+Ybar^2)^2))*Ybar;
        se(j,3) = sqrt((inv(G'*W2*G))/N);
        isCov(j,3) = (theta(j,3)-C*se(j,3)<=0) & (theta(j,3)+C*se(j,3)>=0);
    
        % Iterated GMM estimation
        theta(j,4) = Zbar - ((sigZY/sigY)*Ybar);

        omega_it = [[sigY+Ybar^2,sigZY+(Ybar*(Zbar-theta(j,4)))];
            [sigZY+(Ybar*(Zbar-theta(j,4))), sigZ+(Zbar-theta(j,4))^2]];
        W_it = eye(2)/omega_it;
        se(j,4) = sqrt((inv(G'*W_it*G))/N);
        isCov(j,4) = (theta(j,4)-C*se(j,4)<=0) & (theta(j,4)+C*se(j,4)>=0);
    end

    mean_theta(:,i) = (sum(theta,1)/B)';
    sd_theta(1,i) = sqrt((sum(((theta(:,1)-mean_theta(1,i)).^2),1))/(B-1));
    sd_theta(2,i) = sqrt((sum(((theta(:,2)-mean_theta(2,i)).^2),1))/(B-1));
    sd_theta(3,i) = sqrt((sum(((theta(:,3)-mean_theta(3,i)).^2),1))/(B-1));
    sd_theta(4,i) = sqrt((sum(((theta(:,4)-mean_theta(4,i)).^2),1))/(B-1));
    cp_theta(:,i) = (sum(isCov,1)/B)';
end

figure
tiledlayout(3,1)
zero = zeros(1,length(n));

ax1 = nexttile;
plot(ax1,n,mean_theta(1,:),'.', ...
    n,mean_theta(2,:),':', ...
    n,mean_theta(3,:),'--', ...
    n,mean_theta(4,:),n,zero)
title(ax1,'Bias; \delta = 0, \rho = 0.9')
ylabel(ax1,'bias')
xlabel(ax1,'N')

ax2 = nexttile;
plot(ax2,n,sd_theta(1,:),'.', ...
    n,sd_theta(2,:),':', ...
    n,sd_theta(3,:),'--', ...
    n,sd_theta(4,:))
title(ax2,'Standard Deviation')
ylabel(ax2,'sd')
xlabel(ax2,'N')


ax3 = nexttile;
plot(ax3,n,cp_theta(1,:),'.', ...
    n,cp_theta(2,:),':', ...
    n,cp_theta(3,:),'--', ...
    n,cp_theta(4,:))
title(ax3,'Coverage Probability of 95% CI')
ylabel(ax3,'coverage probability')
xlabel(ax3,'N')
legend('theta_1','theta_2','theta_3','theta_{it}')


%% 3-2. Under large misspecification and High correlation : delta = 0.6, rho = 0.9

% Initialize parameters
n = 10:10:500;
B = 5000;
G = [0;-1];
C = 1.96;

% Vectors for statistics
mean_theta = zeros(4,length(n));
sd_theta = zeros(4,length(n));
cp_theta = zeros(4,length(n));

% 3-2. Under correct specification and High correlation : delta = 0, rho = 0.9
delta = 0.6;
mu = [delta,0];
rho = 0.9;
sigma = [[1,rho];[rho,1]];

for i = 1:length(n)

    % Simulation with default seed
    rng('default');
    
    % Number of obs for each simulation
    N = n(1,i);

    % Vectors for estimates
    theta = zeros(B,4);
    se = zeros(B,4);
    isCov = zeros(B,4);

    for j = 1:B
        X = mvnrnd(mu,sigma,N); % Generate random sample from Bivariate Normal

        % Intermediate variables
        Xbar = sum(X,1)/N;
        Ybar = Xbar(1,1);
        Zbar = Xbar(1,2);

        Sighat = (X-Xbar)'*(X-Xbar)/N;
        sigY = Sighat(1,1);
        sigZ = Sighat(2,2);
        sigZY = Sighat(1,2);

        omega1 = [[sigY+Ybar^2,sigZY];[sigZY,sigZ]];
        W1 = eye(2)/omega1;
        omega2 = [[sigY+Ybar^2,sigZY+((sigZY*Ybar^2)/(sigY+Ybar^2))];
            [sigZY+((sigZY*Ybar^2)/(sigY+Ybar^2)),sigZ+((sigZY*Ybar)/(sigY+Ybar^2))^2]];
        W2 = eye(2)/omega2;
    
        % One step GMM estimation(identity)
        theta(j,1) = Zbar;
        se(j,1) = sqrt((G'*omega1*G)/N);
        isCov(j,1) = (theta(j,1)-C*se(j,1)<=0) & (theta(j,1)+C*se(j,1)>=0);
    
        % Two step GMM estimation
        theta(j,2) = Zbar-((sigZY/(sigY+Ybar^2))*Ybar);
        se(j,2) = sqrt((inv(G'*W1*G))/N);
        isCov(j,2) = (theta(j,2)-C*se(j,2)<=0) & (theta(j,2)+C*se(j,2)>=0);
    
        % Three-step GMM estimation
        theta(j,3) = Zbar-(((sigZY*(sigY+2*Ybar^2)))/((sigY+Ybar^2)^2))*Ybar;
        se(j,3) = sqrt((inv(G'*W2*G))/N);
        isCov(j,3) = (theta(j,3)-C*se(j,3)<=0) & (theta(j,3)+C*se(j,3)>=0);
    
        % Iterated GMM estimation
        theta(j,4) = Zbar - ((sigZY/sigY)*Ybar);

        omega_it = [[sigY+Ybar^2,sigZY+(Ybar*(Zbar-theta(j,4)))];
            [sigZY+(Ybar*(Zbar-theta(j,4))), sigZ+(Zbar-theta(j,4))^2]];
        W_it = eye(2)/omega_it;
        se(j,4) = sqrt((inv(G'*W_it*G))/N);
        isCov(j,4) = (theta(j,4)-C*se(j,4)<=0) & (theta(j,4)+C*se(j,4)>=0);
    end

    mean_theta(:,i) = (sum(theta,1)/B)';
    sd_theta(1,i) = sqrt((sum(((theta(:,1)-mean_theta(1,i)).^2),1))/(B-1));
    sd_theta(2,i) = sqrt((sum(((theta(:,2)-mean_theta(2,i)).^2),1))/(B-1));
    sd_theta(3,i) = sqrt((sum(((theta(:,3)-mean_theta(3,i)).^2),1))/(B-1));
    sd_theta(4,i) = sqrt((sum(((theta(:,4)-mean_theta(4,i)).^2),1))/(B-1));
    cp_theta(:,i) = (sum(isCov,1)/B)';
end

figure
tiledlayout(3,1)
zero = zeros(1,length(n));

ax1 = nexttile;
plot(ax1,n,mean_theta(1,:),'.', ...
    n,mean_theta(2,:),':', ...
    n,mean_theta(3,:),'--', ...
    n,mean_theta(4,:),n,zero)
title(ax1,'Bias; \delta = 0, \rho = 0.9')
ylabel(ax1,'bias')
xlabel(ax1,'N')

ax2 = nexttile;
plot(ax2,n,sd_theta(1,:),'.', ...
    n,sd_theta(2,:),':', ...
    n,sd_theta(3,:),'--', ...
    n,sd_theta(4,:))
title(ax2,'Standard Deviation')
ylabel(ax2,'sd')
xlabel(ax2,'N')


ax3 = nexttile;
plot(ax3,n,cp_theta(1,:),'.', ...
    n,cp_theta(2,:),':', ...
    n,cp_theta(3,:),'--', ...
    n,cp_theta(4,:))
title(ax3,'Coverage Probability of 95% CI')
ylabel(ax3,'coverage probability')
xlabel(ax3,'N')
legend('theta_1','theta_2','theta_3','theta_{it}')

%% 3-3. Under large misspecification and no correlation : delta = 0.6, rho = 0

% Initialize parameters
n = 10:10:500;
B = 5000;
G = [0;-1];
C = 1.96;

% Vectors for statistics
mean_theta = zeros(4,length(n));
sd_theta = zeros(4,length(n));
cp_theta = zeros(4,length(n));

% 3-3. Under large misspecification and no correlation : delta = 0.6, rho = 0
delta = 0.6;
mu = [delta,0];
rho = 0;
sigma = [[1,rho];[rho,1]];

for i = 1:length(n)

    % Simulation with default seed
    rng('default');
    
    % Number of obs for each simulation
    N = n(1,i);

    % Vectors for estimates
    theta = zeros(B,4);
    se = zeros(B,4);
    isCov = zeros(B,4);

    for j = 1:B
        X = mvnrnd(mu,sigma,N); % Generate random sample from Bivariate Normal

        % Intermediate variables
        Xbar = sum(X,1)/N;
        Ybar = Xbar(1,1);
        Zbar = Xbar(1,2);

        Sighat = (X-Xbar)'*(X-Xbar)/N;
        sigY = Sighat(1,1);
        sigZ = Sighat(2,2);
        sigZY = Sighat(1,2);

        omega1 = [[sigY+Ybar^2,sigZY];[sigZY,sigZ]];
        W1 = eye(2)/omega1;
        omega2 = [[sigY+Ybar^2,sigZY+((sigZY*Ybar^2)/(sigY+Ybar^2))];
            [sigZY+((sigZY*Ybar^2)/(sigY+Ybar^2)),sigZ+((sigZY*Ybar)/(sigY+Ybar^2))^2]];
        W2 = eye(2)/omega2;
    
        % One step GMM estimation(identity)
        theta(j,1) = Zbar;
        se(j,1) = sqrt((G'*omega1*G)/N);
        isCov(j,1) = (theta(j,1)-C*se(j,1)<=0) & (theta(j,1)+C*se(j,1)>=0);
    
        % Two step GMM estimation
        theta(j,2) = Zbar-((sigZY/(sigY+Ybar^2))*Ybar);
        se(j,2) = sqrt((inv(G'*W1*G))/N);
        isCov(j,2) = (theta(j,2)-C*se(j,2)<=0) & (theta(j,2)+C*se(j,2)>=0);
    
        % Three-step GMM estimation
        theta(j,3) = Zbar-(((sigZY*(sigY+2*Ybar^2)))/((sigY+Ybar^2)^2))*Ybar;
        se(j,3) = sqrt((inv(G'*W2*G))/N);
        isCov(j,3) = (theta(j,3)-C*se(j,3)<=0) & (theta(j,3)+C*se(j,3)>=0);
    
        % Iterated GMM estimation
        theta(j,4) = Zbar - ((sigZY/sigY)*Ybar);

        omega_it = [[sigY+Ybar^2,sigZY+(Ybar*(Zbar-theta(j,4)))];
            [sigZY+(Ybar*(Zbar-theta(j,4))), sigZ+(Zbar-theta(j,4))^2]];
        W_it = eye(2)/omega_it;
        se(j,4) = sqrt((inv(G'*W_it*G))/N);
        isCov(j,4) = (theta(j,4)-C*se(j,4)<=0) & (theta(j,4)+C*se(j,4)>=0);
    end

    mean_theta(:,i) = (sum(theta,1)/B)';
    sd_theta(1,i) = sqrt((sum(((theta(:,1)-mean_theta(1,i)).^2),1))/(B-1));
    sd_theta(2,i) = sqrt((sum(((theta(:,2)-mean_theta(2,i)).^2),1))/(B-1));
    sd_theta(3,i) = sqrt((sum(((theta(:,3)-mean_theta(3,i)).^2),1))/(B-1));
    sd_theta(4,i) = sqrt((sum(((theta(:,4)-mean_theta(4,i)).^2),1))/(B-1));
    cp_theta(:,i) = (sum(isCov,1)/B)';
end

figure
tiledlayout(3,1)
zero = zeros(1,length(n));

ax1 = nexttile;
plot(ax1,n,mean_theta(1,:),'.', ...
    n,mean_theta(2,:),':', ...
    n,mean_theta(3,:),'--', ...
    n,mean_theta(4,:),n,zero)
title(ax1,'Bias; \delta = 0, \rho = 0.9')
ylabel(ax1,'bias')
xlabel(ax1,'N')

ax2 = nexttile;
plot(ax2,n,sd_theta(1,:),'.', ...
    n,sd_theta(2,:),':', ...
    n,sd_theta(3,:),'--', ...
    n,sd_theta(4,:))
title(ax2,'Standard Deviation')
ylabel(ax2,'sd')
xlabel(ax2,'N')


ax3 = nexttile;
plot(ax3,n,cp_theta(1,:),'.', ...
    n,cp_theta(2,:),':', ...
    n,cp_theta(3,:),'--', ...
    n,cp_theta(4,:))
title(ax3,'Coverage Probability of 95% CI')
ylabel(ax3,'coverage probability')
xlabel(ax3,'N')
legend('theta_1','theta_2','theta_3','theta_{it}')


%% 3-4. Under moderate misspecification and small correlation : delta = 0.3, rho = 0.1
% Initialize parameters
n = 10:10:500;
B = 5000;
G = [0;-1];
C = 1.96;

% Vectors for statistics
mean_theta = zeros(4,length(n));
sd_theta = zeros(4,length(n));
cp_theta = zeros(4,length(n));

% 3-4. Under moderate misspecification and small correlation : delta = 0.3, rho = 0.1
delta = 0.3;
mu = [delta,0];
rho = 0.1;
sigma = [[1,rho];[rho,1]];

for i = 1:length(n)

    % Simulation with default seed
    rng('default');
    
    % Number of obs for each simulation
    N = n(1,i);

    % Vectors for estimates
    theta = zeros(B,4);
    se = zeros(B,4);
    isCov = zeros(B,4);

    for j = 1:B
        X = mvnrnd(mu,sigma,N); % Generate random sample from Bivariate Normal

        % Intermediate variables
        Xbar = sum(X,1)/N;
        Ybar = Xbar(1,1);
        Zbar = Xbar(1,2);

        Sighat = (X-Xbar)'*(X-Xbar)/N;
        sigY = Sighat(1,1);
        sigZ = Sighat(2,2);
        sigZY = Sighat(1,2);

        omega1 = [[sigY+Ybar^2,sigZY];[sigZY,sigZ]];
        W1 = eye(2)/omega1;
        omega2 = [[sigY+Ybar^2,sigZY+((sigZY*Ybar^2)/(sigY+Ybar^2))];
            [sigZY+((sigZY*Ybar^2)/(sigY+Ybar^2)),sigZ+((sigZY*Ybar)/(sigY+Ybar^2))^2]];
        W2 = eye(2)/omega2;
    
        % One step GMM estimation(identity)
        theta(j,1) = Zbar;
        se(j,1) = sqrt((G'*omega1*G)/N);
        isCov(j,1) = (theta(j,1)-C*se(j,1)<=0) & (theta(j,1)+C*se(j,1)>=0);
    
        % Two step GMM estimation
        theta(j,2) = Zbar-((sigZY/(sigY+Ybar^2))*Ybar);
        se(j,2) = sqrt((inv(G'*W1*G))/N);
        isCov(j,2) = (theta(j,2)-C*se(j,2)<=0) & (theta(j,2)+C*se(j,2)>=0);
    
        % Three-step GMM estimation
        theta(j,3) = Zbar-(((sigZY*(sigY+2*Ybar^2)))/((sigY+Ybar^2)^2))*Ybar;
        se(j,3) = sqrt((inv(G'*W2*G))/N);
        isCov(j,3) = (theta(j,3)-C*se(j,3)<=0) & (theta(j,3)+C*se(j,3)>=0);
    
        % Iterated GMM estimation
        theta(j,4) = Zbar - ((sigZY/sigY)*Ybar);

        omega_it = [[sigY+Ybar^2,sigZY+(Ybar*(Zbar-theta(j,4)))];
            [sigZY+(Ybar*(Zbar-theta(j,4))), sigZ+(Zbar-theta(j,4))^2]];
        W_it = eye(2)/omega_it;
        se(j,4) = sqrt((inv(G'*W_it*G))/N);
        isCov(j,4) = (theta(j,4)-C*se(j,4)<=0) & (theta(j,4)+C*se(j,4)>=0);
    end

    mean_theta(:,i) = (sum(theta,1)/B)';
    sd_theta(1,i) = sqrt((sum(((theta(:,1)-mean_theta(1,i)).^2),1))/(B-1));
    sd_theta(2,i) = sqrt((sum(((theta(:,2)-mean_theta(2,i)).^2),1))/(B-1));
    sd_theta(3,i) = sqrt((sum(((theta(:,3)-mean_theta(3,i)).^2),1))/(B-1));
    sd_theta(4,i) = sqrt((sum(((theta(:,4)-mean_theta(4,i)).^2),1))/(B-1));
    cp_theta(:,i) = (sum(isCov,1)/B)';
end

figure
tiledlayout(3,1)
zero = zeros(1,length(n));

ax1 = nexttile;
plot(ax1,n,mean_theta(1,:),'.', ...
    n,mean_theta(2,:),':', ...
    n,mean_theta(3,:),'--', ...
    n,mean_theta(4,:),n,zero)
title(ax1,'Bias; \delta = 0, \rho = 0.9')
ylabel(ax1,'bias')
xlabel(ax1,'N')

ax2 = nexttile;
plot(ax2,n,sd_theta(1,:),'.', ...
    n,sd_theta(2,:),':', ...
    n,sd_theta(3,:),'--', ...
    n,sd_theta(4,:))
title(ax2,'Standard Deviation')
ylabel(ax2,'sd')
xlabel(ax2,'N')


ax3 = nexttile;
plot(ax3,n,cp_theta(1,:),'.', ...
    n,cp_theta(2,:),':', ...
    n,cp_theta(3,:),'--', ...
    n,cp_theta(4,:))
title(ax3,'Coverage Probability of 95% CI')
ylabel(ax3,'coverage probability')
xlabel(ax3,'N')
legend('theta_1','theta_2','theta_3','theta_{it}')

