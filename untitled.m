tic
clear;

load pums80
clear res_1st_2zs res_2sls_weeks res_2sls_worked

data = [agefstm agem1 black boy1st boy2nd hispan hourswm incomem kidcount morekids multi2nd othrace samesex twoboys twogirls weeksm1 workedm];

n = 500;   % subsample size
r = 1;      % number of repetition

spec = 2;   % specification (see below)
dep = 1;    % dependent variable

if spec == 1
    l1 = 2;
elseif spec == 2
    l1 = 3;
elseif spec == 3
    l1 = 2;
end

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
    
    

    Mxz = (X'*Z)/n;
    Mzz = (Z'*Z)/n;
    ehat = Y-X*b2sls;

    Ze = repmat(ehat,1,length(Z(1,:))).*Z;
    Om = (Ze'*Ze)/n;
    Mze = mean(Ze)';

    Sig1 = (Mxz/Mzz*Mxz')\(Mxz/Mzz*Om/Mzz*Mxz')/(Mxz/Mzz*Mxz');
    % sec: conventional heteroskedasticity-robust s.e.
    se1 = sqrt(diag(Sig1/n));


%     J = n*(Mze'/Om*Mze);
%     J_pval(iter) = 1-chi2cdf(J,length(Z(1,:))-length(X(1,:)));
%     
%     sig2 = ehat'*ehat/n;
%     J_homo = n*(Mze'/(sig2*Mzz)*Mze);
%     J_homo_pval(iter) = 1-chi2cdf(J_homo,length(Z(1,:))-length(X(1,:)));

    Vn = 0;
    for i = 1:n
        xi_i = -Mxz/Mzz*(Ze(i,:)'-Mze)-(X(i,:)'*Z(i,:)-Mxz)/Mzz*Mze+Mxz/Mzz*(Z(i,:)'*Z(i,:)-Mzz)/Mzz*Mze;
        Vn = Vn + xi_i*xi_i';
    end
    Vn = Vn/n;

    Sig2 = (Mxz/Mzz*Mxz')\Vn/(Mxz/Mzz*Mxz');
    % ser: heteroskedasticity and multiple-LATEs robust s.e.
    se2 = sqrt(diag(Sig2/n));


        %%%%% Eunkyu Seong's Code %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Misspecification Robust Bootstrap %
    % Hall-Horowitz GMM Bootstrap %
    B = 5000; % number of bootstrap repetition

    TnMR_dist = zeros(B,1);
    TnHH_dist = zeros(B,1);
    for b = 1:B
        b
        
        boot_sample = randi(n,n,1);
        Yb = Y(boot_sample,1);
        Xb = X(boot_sample,:);
        Zb = Z(boot_sample,:);

        b_boot = ((Xb'*Zb)/(Zb'*Zb)*(Zb'*Xb))\((Xb'*Zb)/(Zb'*Zb)*(Zb'*Yb));
    
        if isnan(b_boot)==1
            continue
        end

        b_HH = ((Xb'*Zb)/(Zb'*Zb)*(Zb'*Xb))\((Xb'*Zb)/(Zb'*Zb)*(Zb'*Yb - Mze)); 
        % Hall-Horowitz GMM bootstrap : beta estimation with recentered 
        % moment condition : 
        % The last term Mze is the only difference with the original beta 
        % estimator
        
        if isnan(b_HH)==1
            continue
        end


        Mxzb = (Xb'*Zb)/n;
        Mzzb = (Zb'*Zb)/n;
        ehatb = Yb-Xb*b_boot;
    
        Zeb = repmat(ehatb,1,length(Zb(1,:))).*Zb;
        Omb = (Zeb'*Zeb)/n;
        Mzeb = mean(Zeb)';
        OmHH = Omb - Mze*Mzeb' - Mzeb*Mze' + Mze*Mze';

        SigHH = (Mxzb/Mzzb*Mxzb')\(Mxzb/Mzzb*OmHH/Mzzb*Mxzb')/(Mxzb/Mzzb*Mxzb');
        % seHH: conventional heteroskedasticity-robust s.e. with recentered
        % moment conditions
        seHH = sqrt(diag(SigHH/n));

        TnHH = (b_HH(1) - b2sls(1))/seHH(1);
        TnHH_dist(b,1) = TnHH;
        % TnHH is Hall-Horowitz GMM Bootstrap t-statistic %

        Vnb = 0;
        for i = 1:n
            xi_i = -Mxzb/Mzzb*(Zeb(i,:)'-Mzeb)-(Xb(i,:)'*Zb(i,:)-Mxzb)/Mzzb*Mzeb+Mxzb/Mzzb*(Zb(i,:)'*Zb(i,:)-Mzzb)/Mzzb*Mzeb;
            Vnb = Vnb + xi_i*xi_i';
        end
        Vnb = Vnb/n;

        Sig2b = (Mxzb/Mzzb*Mxzb')\Vnb/(Mxzb/Mzzb*Mxzb');
        % ser: heteroskedasticity and multiple-LATEs robust s.e.
        se2b = sqrt(diag(Sig2b/n));

        TnMR = (b_boot(1)-b2sls(1))/se2b(1);
 
        TnMR_dist(b,1) = TnMR;
        % TnMR is Misspecification Robust Bootstrap t-statistic %
    end

        TnHH_dist = abs(TnHH_dist);
    TnHH_dist = sort(TnHH_dist,1,"ascend","ComparisonMethod","abs");
    qHH = TnHH_dist(0.95*B,1);
    CI_HHboot = [b2sls(1)-qHH*se1(1),b2sls(1)+qHH*se1(1)];

    TnMR_dist = abs(TnMR_dist);
    TnMR_dist = sort(TnMR_dist,1,"ascend","ComparisonMethod","abs");
    qb = TnMR_dist(0.95*B,1);
    CI_MRboot = [b2sls(1)-qb*se2(1),b2sls(1)+qb*se2(1)];