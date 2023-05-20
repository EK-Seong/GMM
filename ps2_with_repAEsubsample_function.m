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
tic
clear;

load pums80
clear res_1st_2zs res_2sls_weeks res_2sls_worked

data = [agefstm agem1 black boy1st boy2nd hispan hourswm incomem kidcount morekids multi2nd othrace samesex twoboys twogirls weeksm1 workedm];

n = 5000;    % subsample/bootstrap sample size
r = 1;    % number of repetition
B = 100;  % number of bootstrap repetition

spec = 2;   % specification (see below)
dep = 1;    % dependent variable

if spec == 1
    l1 = 2;
elseif spec == 2
    l1 = 3;
elseif spec == 3
    l1 = 2;
end


rng default;

repAEsub = repAEsubsample(n,r,B,data,dep,spec);
b2sls_dist = repAEsub(:,1);
SE1 = repAEsub(:,2);
SE2 = repAEsub(:,3);
CI1 = repAEsub(:,4);
CI2 = repAEsub(:,5);
CI_MRboot = repAEsub(:,6);
CI_HHboot = repAEsub(:,7);

%%%
SD = std(b2sls_dist);
%SDiv = std(biv_dist);
Mean2sls = mean(b2sls_dist);
%MeanIV = mean(biv_dist);
%Rej_J5 = mean(J_pval<0.05);
MeanSE1 = mean(SE1);
MeanSE2 = mean(SE2);
%MeanSEiv = mean(SEiv);
%MedianSE1 = median(SE1);
%MedianSE2 = median(SE2);
%MedianSEiv = median(SEiv);
%
%MeanT1 = mean(abs(t1) > 1.96);
%MeanT2 = mean(abs(t2) > 1.96);

MeanCI1 = mean(CI1);
MeanCI2 = mean(CI2);

%%%%% Eunkyu Seong's Code %%%%%

MeanCI_MRboot = mean(CI_MRboot);
MeanCI_HHboot = mean(CI_HHboot);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%DIF = 100*(SE2-SE1)./((SE1+SE2)/2);

toc


