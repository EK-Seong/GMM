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

function Vn = VarMR(n,X,Z,Ze,Mxz,Mzz,Mze)
    Vn = 0;
    for i = 1:n
        xi_i = -Mxz/Mzz*(Ze(i,:)'-Mze)-(X(i,:)'*Z(i,:)-Mxz)/Mzz*Mze+Mxz/Mzz*(Z(i,:)'*Z(i,:)-Mzz)/Mzz*Mze;
        Vn = Vn + xi_i*xi_i';
    end
    Vn = Vn/n;
    return;
end