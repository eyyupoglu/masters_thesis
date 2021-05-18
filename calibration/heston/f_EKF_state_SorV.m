%%%%%构造非线性函数：通过滤波实现%%%%%%%%%%%%%
%一步滤波,已知参数的情况下，通过滤波求下一步估计
%对观测方程，即股票价格的噪声分解，而不是将波动率噪声的分解。
%通过滤波算法得到状态的估计值
%%%%%%%%%%对S的噪声进行分解时用P_design_old=2;%%%%%%%%%%
%对状态的噪声分解时用P_design_old=1;%%%%%%%%%%%%%%%%%%%
function [output_v,P]=f_EKF_state_SorV(para,z,choice,v0,var_s,var_v,P0,kappa_set,...
    kappa_theta_set,sigma_set,ro_set,mu_set,mu,delta)
%z是观测值（差分后的股票价格）
%var_s%生成数据量测的标准差
%var_v%生成数据噪声的标准差
%v是波动率初始值
%para为参数向量，y_m为下一步观测值
%choice=1为一致EKF，2为EKF
%P初始的P


P_design_old=2; %2是根据分解S方程得到新的方法
%1是分解V方向的噪声得到的估计


kappa=para(1);
ka_the=para(2);
sigma=para(3);
ro=para(4);


kappa_upper=max(abs(kappa_set));
kappa_theta_upper=max(abs(kappa_theta_set));
ro_upper=max(abs(ro_set));
sigma_upper=max(abs(sigma_set));
mu_upper=max(abs(mu_set));

kappa_lower=min(abs(kappa_set));
kappa_theta_lower=min(abs(kappa_theta_set));
ro_lower=min(abs(ro_set));
sigma_lower=min(abs(sigma_set));
mu_lower=min(abs(mu_set));
one_min_ro2_max=max(1-ro_set.^2);

%Q=var_v^2*diag([1,1]);%diag([var_v^2,0.00025*ones(1,6)]);%噪声方差阵，可调
Q=diag([var_s,var_v]).^2; %误差是；两维的

%进行滤波的过程
%滤波过程
%compute the state matrics
F=1-kappa*delta;
%     L=[sigma*ro*sqrt(abs(v0*delta)) sigma*sqrt(abs((1-ro^2)*v0*delta))];
L=[0 sigma*sqrt(max(v0*delta,0))];
if choice==1    %%一致EKF
    
    if P_design_old==1 %原来调整的算法
        P_bar=P0+delta^2*kappa_theta_upper^2*P0...
            +delta^2*(kappa_theta_upper-kappa_theta_lower)^2*v0^2....
            +delta^2*(kappa_theta_upper-kappa_theta_lower).^2....
            +(sigma_upper^2*max(eig(Q))*delta^2)*sqrt(2*P0+2*v0^2);
        
    else
        P_bar=P0*(1-kappa_lower*delta)^2+delta^2*kappa_theta_upper^2+sigma_upper^2*max(v0*delta,0)*Q(2,2);
%             P_bar=P0*(1-kappa*delta)^2+delta^2*ka_the^2+sigma^2*max(v0*delta,0)*Q(2,2);
    end
    
    v_bar=max(v0+ ka_the*delta-kappa*v0*delta,0);
    %compute the observation matrics
    H=-0.5*delta;
    M=[sqrt(max((1-ro^2)*v_bar*delta,0)), ro*sqrt(max(v_bar*delta,0))];
    %         M=[sqrt(abs(v_bar*delta)) 0];
    %     H=[-0.5*delta,0,0,0,delta,0];
    %     M=[sqrt(v_bar(1)*delta),0,0,0,0,0,0];
    %mesurement update
    K=(P_bar*H'+L*M')*(H*P_bar*H'+M*M'+H*L*M'+M*L'*H')^(-1);
    v=v_bar+K*(z- (mu-0.5*v_bar)*delta); %滤波值作为输出，xk的估计值
    
    if P_design_old==1
        P0=(1+K*0.5*delta)^2*P_bar+K^2*delta^2*(mu_upper-mu_lower)^2+...
            K^2*sqrt(2*P_bar+2*v_bar^2)*((sigma_upper*ro_upper)^2*max(eig(Q))*delta^2+sigma_upper^2*one_min_ro2_max*max(eig(Q))*delta^2)*max(eig(Q));    
    else
        P0=(1+K*0.5*delta)^2*P_bar+K^2*delta^2*(mu_upper-mu_lower)^2+...
            2*K^2*max(v_bar*delta,0)*((1-ro^2)*Q(1,1)+ro^2*Q(2,2));  %乘以2倍是因为用估计值代替真实值
    end
    
    
elseif choice==2   %%EKF
    
    P_bar=F*P0*F'+L*Q*L';
    
    v_bar=max(v0+ ka_the*delta-kappa*v0*delta,0);
    %compute the observation matrics
    H=-0.5*delta;
    %         M=[sqrt((v_bar*delta)) 0];
    M=[sqrt((1-ro^2)*v_bar*delta), ro*sqrt(v_bar*delta)];
    %mesurement update
%     K=(P_bar*H'+L*M')*(H*P_bar*H'+M*Q*M'+H*L*Q*M'+M*Q*L'*H')^(-1);
    K=(P_bar*H'+L*M')*(H*P_bar*H'+M*Q*M'+H*L*Q*M'+M*Q*L'*H')^(-1);
    v=v_bar+K*(z- (mu-0.5*v_bar)*delta); %xk的估计值
    
    %更新P阵
    %             P=(1-K*H)*P_bar*(1-K*H)'+K*(M*M'+H*L*M'+M*L'*H')*K';
    P0=P_bar-K*(H*P_bar+M*Q*L');
    
end
P=P0; %更新的P阵作为输出
output_v=max(v,0.00001);
% % output_v=abs(v);
% output_v=max(v,0);
% output_v=v;
%%%%%%%%%%%%%%选取参数的方式%%%%%%%%%%%%
%用后面的数据作为参数的极大似然函数，而不是全部的参数



end


