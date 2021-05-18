%估计第二个方程CIR模型的参数，用极大似然法
%用SQRT V的方程估计
function [kappa,theta,sigma]=SQRT_CIR_esti(vReal,delta)
d1=0;
d2=0;
d3=0;
d4=0;
n=length(vReal);
for i=2:1:length(vReal)
    d1=d1+sqrt(vReal(i-1)*vReal(i));
    d2=d2+sqrt(vReal(i)/vReal(i-1));
    d3=d3+vReal(i-1);
    d4=d4+(1/vReal(i-1));
end
P = (d1-(1/(n))*d2*d3 )/(delta*(n)*0.5-delta/(2*(n))*d4*d3);
kappa = (1+ P *delta /(2*(n)) *d4 - d2/(n))*2/delta ;
d5=0;
for j = 2:1:n
%    d6=sqrt(vReal(j)) -sqrt(vReal(j-1)) - delta * (P - kappa * vReal(j-1))/(2*sqrt(vReal(j-1)));
    d5 = d5 +(sqrt(vReal(j)) -sqrt(vReal(j-1)) - delta * (P - kappa * vReal(j-1))/(2*sqrt(vReal(j-1))))^2;
end

% sigma2 = 4 * d5 /((n)*delta);
sigma2 = 4 * d5 /((n)*delta);
sigma=sqrt(sigma2);
theta = (P + 1/4*sigma2 )/kappa;

end