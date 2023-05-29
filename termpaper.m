%% Moderately correlated case : rho = 0.3
% Initialize parameters
%delta = -1:0.05:1;
delta = -2:0.05:2;
N = 500;
B = 5000;
G = [0;-1];
C = 1.96;

% Vectors for statistics
mean_theta = zeros(7,length(delta));
sd_theta = zeros(7,length(delta));
cp_theta = zeros(4,length(delta));

% 2-1. Moderately correlated case : rho = 0.3
rho = 0.3;
sigma = [[1,rho];[rho,1]];

for i = 1:length(delta)
    i
    % Simulation with default seed
    rng('default');
    
    % mu vector for each delta value
    mu = [delta(1,i),0];

    % Vectors for estimates
    theta = zeros(B,7);
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

        % GMM Averaging estimation - one-step + iterated
        Y = X(:,1);
        Z = X(:,2);
        Z1 = Z-theta(j,1);
        Mgn1 = mean(Z1,1);
        Zit = Z-theta(j,4);
        Mgnit = mean([Y,Zit],1);
        J1 = Mgn1*Mgn1';
        Jit = Mgnit*W_it*Mgnit';
        weight1 = 1-(J1/(J1+Jit));
        weight_it = 1-(Jit/(J1+Jit));
        theta(j,5) = weight1*theta(j,1)+weight_it*theta(j,4);

        % GMM Averaging estimation - one-step + two-step + iterated
        Z2 = Z-theta(j,2);
        Mgn2 = mean([Y,Z2],1);
        J2 = Mgn2*W1*Mgn2';
        weight21 = (1 -(J1/(J1+J2+Jit)))/2;
        weight22 = (1 -(J2/(J1+J2+Jit)))/2;
        weight2_it = (1 -(Jit/(J1+J2+Jit)))/2;
        theta(j,6) = weight21*theta(j,1)+weight22*theta(j,2)+weight2_it*theta(j,4);

        % GMM Averaging estimation - one-step + two-step + three-step
        Z3 = Z-theta(j,3);
        Mgn3 = mean([Y,Z3],1);
        J3 = Mgn3*W2*Mgn3';
        weight31 = (1-(J1/(J1+J2+J3)))/2;
        weight32 = (1-(J2/(J1+J2+J3)))/2;
        weight33 = (1-(J3/(J1+J2+J3)))/2;
        %weight3_it = (1-(Jit/(J1+J2+J3+Jit)))/3;
        theta(j,7) = weight31*theta(j,1)+weight32*theta(j,2)+weight33*theta(j,3);

    end

    mean_theta(:,i) = (sum(theta,1)/B)';
    sd_theta(1,i) = sqrt((sum(((theta(:,1)-mean_theta(1,i)).^2),1))/(B-1));
    sd_theta(2,i) = sqrt((sum(((theta(:,2)-mean_theta(2,i)).^2),1))/(B-1));
    sd_theta(3,i) = sqrt((sum(((theta(:,3)-mean_theta(3,i)).^2),1))/(B-1));
    sd_theta(4,i) = sqrt((sum(((theta(:,4)-mean_theta(4,i)).^2),1))/(B-1));
    sd_theta(5,i) = std(theta(:,5));
    sd_theta(6,i) = std(theta(:,6));
    sd_theta(7,i) = std(theta(:,7));
    cp_theta(:,i) = (sum(isCov,1)/B)';
end

figure
tiledlayout(1,2)
zero = zeros(1,length(delta));

ax1 = nexttile;
plot(ax1,delta,mean_theta(1,:),'.', ...
    delta,mean_theta(2,:),':', ...
    delta,mean_theta(3,:),'--', ...
    delta,mean_theta(4,:),delta,zero)
title(ax1,'Bias')
ylim(ax1,[-0.4,0.4])
xlim(ax1,[-2,2])
ylabel(ax1,'bias')
xlabel(ax1,'\delta')
legend('theta_1','theta_2','theta_3','theta_{it}')

ax2 = nexttile;
plot(ax2,delta,sd_theta(1,:),'.', ...
    delta,sd_theta(2,:),':', ...
    delta,sd_theta(3,:),'--', ...
    delta,sd_theta(4,:))
title(ax2,'Standard Deviation')
ylim(ax2,[0.04,0.06])
xlim(ax2,[-2,2])
ylabel(ax2,'sd')
xlabel(ax2,'\delta')
legend('theta_1','theta_2','theta_3','theta_{it}')

% ax3 = nexttile;
% plot(ax3,delta,cp_theta(1,:),'.', ...
%     delta,cp_theta(2,:),':', ...
%     delta,cp_theta(3,:),'--', ...
%     delta,cp_theta(4,:))
% title(ax3,'Coverage Probability of 95% CI')
% ylabel(ax3,'coverage probability')
% xlabel(ax3,'\delta')
% legend('theta_1','theta_2','theta_3','theta_{it}')

figure
tiledlayout(1,2)
zero = zeros(1,length(delta));

ax1 = nexttile;
plot(ax1,delta,mean_theta(1,:),'.', ...
    delta,mean_theta(2,:),':', ...
    delta,mean_theta(3,:),'--', ...
    delta,mean_theta(4,:), ...
    delta,mean_theta(7,:), ...
    delta,zero)
title(ax1,'Bias')
ylim(ax1,[-0.4,0.4])
xlim(ax1,[-2,2])
ylabel(ax1,'bias')
xlabel(ax1,'\delta')
legend('theta_1','theta_2','theta_3','theta_{it}','theta_{av}')

ax2 = nexttile;
plot(ax2,delta,sd_theta(1,:),'.', ...
    delta,sd_theta(2,:),':', ...
    delta,sd_theta(3,:),'--', ...
    delta,sd_theta(4,:), ...
    delta,sd_theta(7,:))
title(ax2,'Standard Deviation')
ylim(ax2,[0.04,0.06])
xlim(ax2,[-2,2])
ylabel(ax2,'sd')
xlabel(ax2,'\delta')
legend('theta_1','theta_2','theta_3','theta_{it}','theta_{av}')