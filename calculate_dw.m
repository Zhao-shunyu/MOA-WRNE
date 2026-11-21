%%计算降水-蒸发+径流模拟
function dW= calculate_dw(T,RH,P,u2,Rs,Rain,z,lat,J)
%% 2 饱和水汽压 es
es = 0.6108 .* exp((17.27 .* T) ./ (T + 237.3));

%% 3 实际水汽压 ea
ea = es .* RH ./ 100;

%% 4 饱和水汽压斜率 Δ
Delta = 4098 .* es ./ ((T + 237.3).^2);

%% 5 干湿常数 γ
gamma = 0.665e-3 .* P;

%% 6 净辐射 Rn
% FAO常数
sigma = 4.903e-9;
% 太阳常数
Gsc = 0.0820;
% 地日距离
dr = 1 + 0.033*cos(2*pi/365 .* J);
% 太阳赤纬
delta = 0.409*sin(2*pi/365 .* J - 1.39);
% 日落时角
ws = acos(-tan(lat).*tan(delta));
% 大气外辐射
Ra = (24*60/pi) .* Gsc .* dr .* ...
    (ws.*sin(lat).*sin(delta) + cos(lat).*cos(delta).*sin(ws));
% 晴空辐射
Rso = (0.75 + 2e-5*z).*Ra;
% 短波净辐射
Rns = (1-0.08).*Rs;
% 温度K
T_k = T + 273.16;
% 长波辐射
Rnl = sigma .* (T_k.^4) .* ...
    (0.34 - 0.14.*sqrt(ea)) .* ...
    (1.35 .* (Rs./Rso) - 0.35);
% 净辐射
Rn = Rns - Rnl;

%% 7 土壤热通量 G (日尺度湖泊≈0)
G = 0;

%% 8 Penman–Monteith 蒸发量 ET0
ET0 = (0.408 .* Delta .* (Rn - G) + gamma .* (900 ./ (T + 273)) .* u2 .* (es - ea)) ...
     ./ (Delta + gamma .* (1 + 0.34 .* u2));

%% 9 Monte Carlo 径流模拟
alpha_min = 0.05;
alpha_max = 0.30;
N = 10000; % 蒙特卡洛模拟次数

n = length(Rain);  % 数据天数

% 水量变化模拟：使用 Monte Carlo 生成 10000 个α
alpha = alpha_min + (alpha_max-alpha_min)*rand(N,1);

% 初始化水量变化序列矩阵 W (每条模拟曲线对应一个列)
dW = zeros(n, N);  % 用来存储每一天的水量变化

A=5820000;%水域面积（单位平方米）
B=32.5*1000000; %北邻山集雨面积（单位平方米）
% 水量变化模拟
for i = 1:n
    P = Rain(i)*A/1000;  % 降水量
    E = ET0(i)*A/1000;   % 蒸发量
    % 每次使用随机生成的α值计算相应的径流入水量
    Qin = alpha .* Rain(i)*B/1000;  % 模拟的径流入水量
    dW(i,:) = P - E + Qin;  % 水量变化
end
end


