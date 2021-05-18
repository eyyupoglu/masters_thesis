
% clc; clear;
% 
% % Option features
% r = 0.05;           % Risk free rate
% q = 0;           % Dividend yield
% Mat = 10;          % Maturity in years
% S0 = 100;           % Spot price
% 
% % Heston parameters
% kappa =  5;       % Variance reversion speed
% theta =  0.2;    % Variance reversion level
% sigma =  0.4;     % Volatility of Variance
% v0    =  0.4;    % Initial variance
% lambda = 0;       % Risk parameter
% 
% % Simulation features
% N = 1;             % Number of stock price paths
% T = 2520;           % Number of time steps per path
% alpha = 0.5;       % Weight for explicit-implicit scheme
% negvar = 'T';      % Use the truncation scheme for negative variances
% rho   =  -0.25;      % Correlation between Brownian motions
% params = [kappa theta sigma v0 rho ];
% 
% 
% %% Simulate the processes and obtain the option prices
% schemeV = 'E';
% % [S V F Price] = EulerMilsteinPrice(schemeV,negvar,params,PutCall,S0,K,Mat,r,q,T,N,alpha);
% [S V F] = EulerMilsteinSim(schemeV,negvar,params,S0,Mat,r,q,T,N,alpha);
% figure;
% stackedplot([S V]);

%%
%这组设置的参数值千万不要动！！！
%递推式估计参数，用MLE方法分别估计四个参数
%主要是对kappa初值以及界的选取！！！
%实际数据，估计参数加跟踪波动率 
%无风险利率用的用的年libor的平均值
%% 比较实际波方差，和估计得到的V(t)的大小
%% 把CEKF和EKF方法都估计出来
tic;
close all
clear,clc
[data,textstr]=xlsread('SP_VIX.csv');
delta=1/252;%每天为时间间隔
S=data(:,1);
% S=S;
Y=log(S);  %股票价格的对数值
z_m=diff(Y); %对数股价做差分
VIX=data(:,2)*0.01;%市场隐含波动率
Variance=VIX.^2;%方差，与滤波值比较

longmean=mean(Variance); %平均值在0.0556左右
speed=[max(diff(Variance)./Variance(1:end-1)),min(diff(Variance)./Variance(1:end-1))]; %[1.25,-0.504]
% correlation=corr(Y,Variance); %-0.6074
correlation=-0.25; %-0.6074
%% 第一步： 生成100组原始数据
%mu=0.1
mu=0.005; %无风险收益率
r=mu;
V10=Variance(1); %初始值

var_s=1;  %噪声尽量选标准正态分布
var_v=1;  %
P0=0.5;  %初始的滤波P阵,可调
% P0=0.01;  %初始的滤波P阵,可调


%% 第二步：极大似然求参数估计

%%%%%%%%%%%%%第二组%%%%%%%%%%%%
 %kappa_ini=0.2;可行3
%  kappa_ini=0.5;%可行2
kappa_ini=1;
theta_ini=0.03;
kappa_theta_ini=kappa_ini*theta_ini;
sigma_ini=0.2;

% kappa_lb=0.8;可行1和可行2都可以用
kappa_lb=0.1;%可行3
kappa_ub=2;%可行2可行3
% kappa_ub=4;%可行2
%kappa_ub=5;%可行1
kappa_theta_lb=0.002; %theta均值为0.2左右
kappa_theta_ub=0.7;   %之前估计的好，后面没有上面估计的好
sigma_lb=0.1;
sigma_ub=0.6;
ro_lb=-0.6;  
ro_ub=0.6;
kappa_theta_set=[kappa_theta_lb,kappa_theta_ub];
kappa_set = [kappa_lb, kappa_ub];
sigma_set=[sigma_lb,sigma_ub];
ro_set=[ro_lb,ro_ub];
mu_set=[mu,mu];
step_set=20;  %设定参数变换开始时刻

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


choice1=1;  %一致EKF为1, EKF为2
choice2=2; 
v=V10;   %波动率滤波的初始估计值

lb=[kappa_lb,kappa_theta_lb,sigma_lb,ro_lb];
up=[kappa_ub,kappa_theta_ub,sigma_ub,ro_ub];

 %滤波只用到观测值，不用到真实状态值
ro_ini=0.2;
parax_final1=zeros(length(z_m),4);
parax_final2=zeros(length(z_m),4);
V_k1=zeros(length(z_m),1);
P_k1=zeros(length(z_m),1);
V_k2=zeros(length(z_m),1);
P_k2=zeros(length(z_m),1);
parax_final1(1,:)=[kappa_ini,kappa_theta_ini,sigma_ini,ro_ini];
parax_final2(1,:)=[kappa_ini,kappa_theta_ini,sigma_ini,ro_ini];
V_k1(1)=V10; %估计初值与真实初值选取不一样
P_k1(1)=P0;
V_k2(1)=V10;
P_k2(1)=P0;

%% 第三步：根据MLE 估计出来的参数，对波动率进行估计

for k=2:length(z_m)
   
    %%%%% CEKF %%%%%%%%%%%%%%%%%%%%用上一步的初值，参数值估计出状态值
    [V_k1(k),P_k1(k)]=f_EKF_state_SorV(parax_final1(k-1,:),z_m(k),choice1,V_k1(k-1),var_s,var_v,P_k1(k-1),kappa_set,...
        kappa_theta_set,sigma_set,ro_set,mu_set,mu,delta);
    %跟新参数
    if k<=step_set%若初值不是真值，可以一些步数
        %             if k<=length(z_m)
        parax_final1(k,:)=parax_final1(k-1,:);
    else
        %%%%%%%%%有些值偏大，会使估计的均值偏大%%%%%%%%
        %%%%%%%%%在估计参数的时候，应该去掉高估的波动率%%%%%%%%%%
        if V_k1(k)>0.2
            V_k1_ba = [V_k1(1:k-1);0.1];
            [kappa_ba_new1,theta_ba_new1,sigma_ba_new1]=SQRT_CIR_esti(V_k1_ba,delta); %用滤波值估计参数
            parax_final1(k,1)=kappa_ba_new1;
            parax_final1(k,2)=kappa_ba_new1*theta_ba_new1;
            parax_final1(k,3)=sigma_ba_new1;
            parax_final1(k,4)= ro_ini;
        else
            V_k1_ba = [V_k1(1:k-1);V_k1(k)];
            [kappa_ba_new1,theta_ba_new1,sigma_ba_new1]=SQRT_CIR_esti(V_k1_ba,delta); %用滤波值估计参数
            parax_final1(k,1)=kappa_ba_new1;
            parax_final1(k,2)=kappa_ba_new1*theta_ba_new1;
            parax_final1(k,3)=sigma_ba_new1;
            parax_final1(k,4)= ro_ini;
            
        end
    end
%%%%%%%%%%%%%EKF%%%%%%%%%%%%%%%%
               %用上一步的初值，参数值估计出状态值
%     [V_k2(k),P_k2(k)]=f_EKF_state_SorV(parax_final2(k-1,:),z_m(k),choice2,V_k2(k-1),var_s,var_v,P_k2(k-1),kappa_set,...
%         kappa_theta_set,sigma_set,ro_set,mu_set,mu,delta);
%     %跟新参数
% %     if k<=step_set
% % %                 if k<=length(z_m)
% %         parax_final2(k,:)=parax_final2(k-1,:);
% %     else
% %         [kappa_ba_new2,theta_ba_new2,sigma_ba2_new2]=CIR_esti(V_k2(1:k),delta); %用滤波值估计参数
% %         %         sigma_ba_new=sqrt(abs(sigma_ba2_new)); %防止出现复数
% %         sigma_ba_new2=sqrt(max(sigma_ba2_new2,0.0000001)); %防止出现复数
% % %         ro_ba_new=(mean(z_m(1:k))/delta-r-0.5*theta_ba_new2)/(0.5*sigma_ba_new2);
% %         parax_final2(k,1)=kappa_ba_new2;
% %         parax_final2(k,2)=kappa_ba_new2*theta_ba_new2;
% %         parax_final2(k,3)=sigma_ba_new2;
% %          obj_f=@(ro)(Rho_MLE_fminfun(ro,V_k2(1:k),z_m(1:k),r,delta,parax_final2(k,1:3)./([1 parax_final2(k,1) 1])));
% %         parax_final2(k,4)=fmincon(obj_f,ro_ini,[],[],[],[],-1,0);
% %      end
end

kappa_all_CEKF=parax_final1(:,1);
kapta_all_CEKF=parax_final1(:,2);
sigma_all_CEKF=parax_final1(:,3);
ro_all_CEKF=parax_final1(:,4);

% stackedplot(parax_final1);
% figure 
% yData = linspace(1,length(V_k1),length(V_k1));
% plot(yData, V_k1, 'r');
% plot(yData, V(2:end), 'k', yData, V_k1, 'r');

% kappa_all_EKF=parax_final2(:,1);
% kapta_all_EKF=parax_final2(:,2);
% sigma_all_EKF=parax_final2(:,3);
% ro_all_EKF=parax_final2(:,4);
% return;
%% 结果展示
[data2,textstr2]=xlsread('SP_VIX2005_2016.csv');
delta=1/252;%每天为时间间隔
SP2=data2(:,1);
Y2=log(SP2);  %股票价格的对数值
% VIX=data(:,2)*0.01;%波动率
% start = find(textstr2(:,1)=={'2006-12-1'})
start = find(strcmp(textstr2(:,1),'12/1/2006'));
Y2start = start-1;
%%%%%根据60天的数据计算每天的历史波动率，并年化%%%%
m = 30;
StandDevi = zeros(1,length(Y2));
for j = Y2start:1:length(Y2)
    HistoryReturn = Y2(j-m+1:j);
    Variance(j) = sum(HistoryReturn.^2)/m; 
    StandDevi(j) =std(HistoryReturn);
end
figure
StandDevi=  StandDevi(484:end)*sqrt(252);
StandDevi2=  StandDevi;
v_real=VIX;
error_v1=V_k1-v_real(1:length(V_k1));
error_v2=V_k2-v_real(1:length(V_k1));
%把图画在一起
yData = linspace(1,length(V_k1),length(V_k1));
for ii=1:length(V_k1)
    V(ii) = sqrt(V_k1(ii));
end
figure
plot(yData,v_real(1:length(V_k1)),'k',yData,V,'m',yData,StandDevi2(1:length(V_k1)))
plot(yData,v_real(1:length(V_k1)),'k',yData,V_k1,'m')
hold on
plot(yData,V_k2)
title('EKF ESTIMATION OF VOLATILITY')
legend('VIX^2','CEKF Variance','EKF Variance')
set(gca,'XTick',yData(1:200:end))
datetick('x','yyyy/mm/dd','keepticks') ;
%datetick('x','yyyy/mm','keepticks') ;
a=textstr(2:end,1);
set(gca,'XTick',yData(1:200:end))
set(gca,'XTickLabel',a(1:200:end))
%旋转横坐标
xtb = get(gca,'XTickLabel');% 获取横坐标轴标签句柄
xt = get(gca,'XTick');% 获取横坐标轴刻度句柄
yt = get(gca,'YTick'); % 获取纵坐标轴刻度句柄          
xtextp=xt;%每个标签放置位置的横坐标，这个自然应该和原来的一样了。                     
ytextp=yt(1)*ones(1,length(xt));
text(xtextp,ytextp,xtb,'HorizontalAlignment','right','rotation',45,'fontsize',8.5); 
set(gca,'xticklabel','');% 将原有的标签
%%%%%%%%画误差图%%%%%%%%%%%%%%%%
figure
plot(yData,error_v1.^2,'m')
hold on
plot(yData,error_v2.^2)
legend('CEKF error','EKF error')
set(gca,'XTick',yData(1:200:end))
datetick('x','yyyy/mm/dd','keepticks') ;
%datetick('x','yyyy/mm','keepticks') ;
set(gca,'XTick',yData(1:200:end))
set(gca,'XTickLabel',a(1:200:end))
%旋转横坐标
xtb = get(gca,'XTickLabel');% 获取横坐标轴标签句柄
xt = get(gca,'XTick');% 获取横坐标轴刻度句柄
yt = get(gca,'YTick'); % 获取纵坐标轴刻度句柄          
xtextp=xt;%每个标签放置位置的横坐标，这个自然应该和原来的一样了。                     
ytextp=yt(1)*ones(1,length(xt));
text(xtextp,ytextp,xtb,'HorizontalAlignment','right','rotation',45,'fontsize',8.5); 
set(gca,'xticklabel','');% 将原有的标签




%%%%%%%%%%%%%CEKF参数估计的图%%%%%%%%%%%%%%%%%
%参数估计的图
figure
subplot(4,1,1)
plot(1:length(z_m),kappa_all_CEKF,'m')
% hold on
% plot(1:length(z_m),kappa*ones(1,length(z_m)))
title('$\hat{\kappa}$','Interpreter','latex','FontSize',14)
% ylim([1,6])


subplot(4,1,2)
plot(1:length(z_m),kapta_all_CEKF./kappa_all_CEKF,'m')
% hold on
% plot(1:length(z_m),theta*ones(1,length(z_m)))
title('$\hat{\theta}$','Interpreter','latex','FontSize',14)
% ylim([0.24,0.28])

subplot(4,1,3)
plot(1:length(z_m),sigma_all_CEKF,'m')
% hold on 
% plot(1:length(z_m),sigma*ones(1,length(z_m)))
title('$\hat{\sigma}$','Interpreter','latex','FontSize',14)
% ylim([0,0.25])



subplot(4,1,4)
plot(1:length(z_m),ro_all_CEKF,'m')
% hold on 
% plot(1:length(z_m),ro*ones(1,length(z_m)))
title('$\hat{\rho}$','Interpreter','latex','FontSize',14)
% ylim([-1,1])
xlabel('k')
legend('esimation')

% ylim([-1,1])
sgtitle('CEKF Filtering value to estimate parameters')



%%%%%%%%%%%%%%%%%EKF参数估计的图%%%%%%%%%%%%%%%%%%
%参数估计的图
% figure
% subplot(4,1,1)
% plot(1:length(z_m),kappa_all_EKF,'m')
% % hold on
% % plot(1:length(z_m),kappa*ones(1,length(z_m)))
% title('$\hat{\kappa}$','Interpreter','latex','FontSize',14)
% % ylim([1,6])
% 
% 
% subplot(4,1,2)
% plot(1:length(z_m),kapta_all_EKF./kappa_all_EKF,'m')
% % hold on
% % plot(1:length(z_m),theta*ones(1,length(z_m)))
% title('$\hat{\theta}$','Interpreter','latex','FontSize',14)
% %  ylim([0.2,0.4])
% % ylim([0.24,0.28])
% subplot(4,1,3)
% plot(1:length(z_m),sigma_all_EKF,'m')
% % hold on 
% % plot(1:length(z_m),sigma*ones(1,length(z_m)))
% title('$\hat{\sigma}$','Interpreter','latex','FontSize',14)
% % ylim([-1,1])
% ylim([0,0.25])
% 
% subplot(4,1,4)
% plot(1:length(z_m),ro_all_EKF,'m')
% % hold on 
% % plot(1:length(z_m),ro*ones(1,length(z_m)))
% title('$\hat{\rho}$','Interpreter','latex','FontSize',14)
% % ylim([-1,1])
% xlabel('k')
% legend('esimation')
%  suptitle('EKF Filtering value to estimate parameters')

toc;





 