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

function Tn_dist = GMMbootstrap(n,B,Y,X,Z,Mze,b2sls)
    TnMR_dist = zeros(B,1);
    TnHH_dist = zeros(B,1);
    for b = 1:B
        
        boot_sample = randi(n,n,1);
        Yb = Y(boot_sample,1);
        Xb = X(boot_sample,:);
        Zb = Z(boot_sample,:);

        b_boot = ((Xb'*Zb)/(Zb'*Zb)*(Zb'*Xb))\((Xb'*Zb)/(Zb'*Zb)*(Zb'*Yb));
    
        if isnan(b_boot)==1
            continue
        end

        b_HH = ((Xb'*Zb)/(Zb'*Zb)*(Zb'*Xb))\((Xb'*Zb)/(Zb'*Zb)*(Zb'*Yb - Mze)); 
        % Hall-Horowitz GMM bootstrap : beta estimation with recentered moment condition : 
        % The last term Mze is the only difference with the original beta estimator
        
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

        Vnb = VarMR(n,Xb,Zb,Zeb,Mxzb,Mzzb,Mze);

        Sig2b = (Mxzb/Mzzb*Mxzb')\Vnb/(Mxzb/Mzzb*Mxzb');
        % ser: heteroskedasticity and multiple-LATEs robust s.e.
        se2b = sqrt(diag(Sig2b/n));

        TnMR = (b_boot(1)-b2sls(1))/se2b(1);
 
        TnMR_dist(b,1) = TnMR;
        % TnMR is Misspecification Robust Bootstrap t-statistic %
    end
    Tn_dist = [TnHH_dist,TnMR_dist];
    return;
end