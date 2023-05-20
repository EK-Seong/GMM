%
% Seojeong Jay Lee 
% jay.lee@unsw.edu.au
% Apr 15 2016
%
% Replication of Angrist and Evans (1998)
% Multiple-LATEs-Robust standard errors are also calculated. 
%
%

%
% Eunkyu Seong
% uocup96@snu.ac.kr
% May 17 2023
%
% Replication of Angrist and Evans (1998)
% Simulating various CIs and comparing their performances
%
%

function repAEsub = repAEsubsample(n,r,B,data,dep,spec)

    b2sls_dist = zeros(r,1);
    %biv_dist = zeros(r,l1);
    SE1 = zeros(r,1);
    SE2 = zeros(r,1);
    %SEiv = zeros(r,l1);
    %J_pval = zeros(r,1);
    %J_homo_pval = zeros(r,1);
    %F = zeros(r,1);
    %concent = zeros(r,1);
    %
    % t1 = zeros(r,1);
    % t2 = zeros(r,1);
    % 
    CI1 = zeros(r,1);
    CI2 = zeros(r,1);
    
    %%%%% Eunkyu Seong's Code %%%%%
    
    CI_MRboot = zeros(r,1); % Lee(2014) Misspecification-robust bootstrap CI
    CI_HHboot = zeros(r,1); % Hall,Horowitz(1996) bootstrap CI

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    iter = 1;

    while iter <= r
        iter
    
        subdata = datasample(data,n,'Replace',false);
        agefstm = subdata(:,1);
        agem1 = subdata(:,2);
        black = subdata(:,3);
        boy1st = subdata(:,4);
        boy2nd = subdata(:,5);
        hispan = subdata(:,6);
        hourswm = subdata(:,7);
        incomem = subdata(:,8);
        kidcount = subdata(:,9);
        morekids = subdata(:,10);
        multi2nd = subdata(:,11);
        othrace = subdata(:,12);
        samesex = subdata(:,13);
        twoboys = subdata(:,14);
        twogirls = subdata(:,15);
        weeksm1 = subdata(:,16);
        workedm = subdata(:,17);
        
        if dep == 1
            Y = workedm;
            b0 = -0.0951033373675254;
        elseif dep == 2
            Y = weeksm1;
            b0 = -4.50813382397778;
        elseif dep == 3
            Y = hourswm;
            b0 = -3.84288285626525;
        elseif dep == 4
            Y = incomem;
            b0 = -1404.92796215925;
        end
      
        if spec == 1
            X = [morekids boy1st agem1 agefstm black hispan othrace ones(n,1)];
            Z = [twoboys twogirls boy1st agem1 agefstm black hispan othrace ones(n,1)];
        elseif spec == 2
            X = [morekids boy1st agem1 agefstm black hispan othrace ones(n,1)];
            Z = [multi2nd twoboys twogirls boy1st agem1 agefstm black hispan othrace ones(n,1)];
        elseif spec == 3
            X = [morekids boy1st boy2nd agem1 agefstm black hispan othrace ones(n,1)];
            Z = [multi2nd samesex boy1st boy2nd agem1 agefstm black hispan othrace ones(n,1)];
        end
        
        b2sls = ((X'*Z)/(Z'*Z)*(Z'*X))\((X'*Z)/(Z'*Z)*(Z'*Y));
        
        if isnan(b2sls)==1
            continue
        end
        
        b2sls_dist(iter) = b2sls(1);
    
        Mxz = (X'*Z)/n;
        Mzz = (Z'*Z)/n;
        ehat = Y-X*b2sls;
    
        Ze = repmat(ehat,1,length(Z(1,:))).*Z;
        Om = (Ze'*Ze)/n;
        Mze = mean(Ze)';
    
        Sig1 = (Mxz/Mzz*Mxz')\(Mxz/Mzz*Om/Mzz*Mxz')/(Mxz/Mzz*Mxz');
        % sec: conventional heteroskedasticity-robust s.e.
        se1 = sqrt(diag(Sig1/n));
    
        SE1(iter) = se1(1);
    
    %     J = n*(Mze'/Om*Mze);
    %     J_pval(iter) = 1-chi2cdf(J,length(Z(1,:))-length(X(1,:)));
    %     
    %     sig2 = ehat'*ehat/n;
    %     J_homo = n*(Mze'/(sig2*Mzz)*Mze);
    %     J_homo_pval(iter) = 1-chi2cdf(J_homo,length(Z(1,:))-length(X(1,:)));
    
        Vn = VarMR(n,X,Z,Ze,Mxz,Mzz,Mze);
    
        Sig2 = (Mxz/Mzz*Mxz')\Vn/(Mxz/Mzz*Mxz');
        % ser: heteroskedasticity and multiple-LATEs robust s.e.
        se2 = sqrt(diag(Sig2/n));
    
        SE2(iter) = se2(1);
    
    %     % IV estimator using one IV at a time
    %     for j = 1:l1
    %         Ziv = [Z(:,j) Z(:,(l1+1):length(Z(1,:)))];
    %         biv = (Ziv'*X)\(Ziv'*Y);
    %         biv_dist(iter,j) = biv(1);
    %         Mxziv = (X'*Ziv)/n;
    %         ehat1 = Y-X*biv;
    %         Zeiv = repmat(ehat1,1,length(Ziv(1,:))).*Ziv;
    %         Omiv = (Zeiv'*Zeiv)/n;
    %         Sigiv = Mxziv'\Omiv/Mxziv;
    %         seiv = sqrt(diag(Sigiv/n));
    %         SEiv(iter,j) = seiv(1);
    %     end
    
    %     % t statistic testing H0: b=0
    %     t1(iter) = (b2sls(1))/se1(1);
    %     t2(iter) = (b2sls(1))/se2(1);
        
        % confidence interval for b using the full sample
        CI1(iter) = (b2sls(1)-1.96*se1(1) < b0)&(b0 < b2sls(1)+1.96*se1(1));
        CI2(iter) = (b2sls(1)-1.96*se2(1) < b0)&(b0 < b2sls(1)+1.96*se2(1));
    
    
        %%%%% Eunkyu Seong's Code %%%%%
        
        % Misspecification Robust Bootstrap %
        % Hall-Horowitz GMM Bootstrap %
  
        Tn_dist = GMMbootstrap(n,B,Y,X,Z,Mze,b2sls);
        
        TnHH_dist = Tn_dist(:,1);
        TnHH_dist = abs(TnHH_dist);
        TnHH_dist = sort(TnHH_dist,1,"ascend","ComparisonMethod","abs");
        qHH = TnHH_dist(0.95*B,1);
        CI_HHboot(iter) = (b2sls(1)-qHH*se1(1) < b0)&(b0 < b2sls(1)+qHH*se1(1));
        
        TnMR_dist = Tn_dist(:,2);
        TnMR_dist = abs(TnMR_dist);
        TnMR_dist = sort(TnMR_dist,1,"ascend","ComparisonMethod","abs");
        qb = TnMR_dist(0.95*B,1);
        CI_MRboot(iter) = (b2sls(1)-qb*se2(1) < b0)&(b0 < b2sls(1)+qb*se2(1));
    
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    %     % F statistic
    %     pihat = (Z'*Z)\(Z'*morekids);
    %     Dfit = Z*pihat;
    %     uhat = morekids-Dfit;
    %     
    %     Qzz = (Z'*Z)/n;
    %     Zuhat=Z.*(uhat*ones(1,length(Z(1,:))));
    %     Vpi = Qzz\(Zuhat'*Zuhat/n)/Qzz;
    % 
    %     R = zeros(length(Z(1,:)),l1);
    %     R(1:l1,1:l1) = eye(l1);
    % 
    %     Wald = n*(R'*pihat)'/(R'*Vpi*R)*(R'*pihat);
    %     f = Wald/l1;
    %     F(iter) = f;
    % 
    %     % Concentration parameter estimate
    %     % Following Hansen, Hausman, and Newey (2009)
    %     concent(iter) = (length(X(:,1))-length(Z(:,1))+1)*(f-1);
        
        iter = iter + 1;
    end
    repAEsub = [b2sls_dist,SE1,SE2,CI1,CI2,CI_HHboot,CI_MRboot];

    return;
end