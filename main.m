clc; clear; close all;
rng(1);   % 固定随机种子，方便复现

%% =========================================================
% 1. 基本参数
% ==========================================================
start_date = datetime(2024,1,1);
end_date   = datetime(2025,12,31);

Qin_max  = 8e5;       % 最大进水量
Qout_max = 3.44736e6; % 最大排水量

% 风险阈值  
r_safe = 0.20;    % 可接受风险
r_zero = 0.001;   % 接近零风险
tol_alpha = 1e-3; % alpha 接近 0 或 1 时做截断

% 水量阈值（初始水量=0，增加为正，减少为负）
Hmin = -4.656e6;
Hmax =  1.746e6;

%% =========================================================
% 2. 读取气象数据
% ==========================================================
data_all = readtable('肇庆高要站点气象.csv','VariableNamingRule','preserve');
date_all = datetime(data_all.("年"), data_all.("月"), data_all.("日"));

idx = (date_all >= start_date) & (date_all <= end_date);
data = data_all(idx,:);

dates = datetime(data.("年"), data.("月"), data.("日"));

T    = data.("气温(℃)");
RH   = data.("湿度(%)");
P    = data.("气压(hpa)")/10;   % hPa -> kPa
u2   = data.("风速(m/s)");
Rs   = data.("辐射(MJ/m^2)");
Rain = data.("降水量（mm）");
z    = data.("海拔高度")(1);
lat  = deg2rad(data.("纬度")(1));
J    = day(dates,'dayofyear');

nDay = length(dates);

%% =========================================================
% 3. Monte Carlo 计算每日自然水量变化 dw_mc
%    注意：这里只表示“当天自然来水不确定性”
% ==========================================================
dw_mc = calculate_dw(T,RH,P,u2,Rs,Rain,z,lat,J);

%% ========================================================= 
% 4. 月尺度 TN / TP 预测
%    用前3个月气象 -> 预测第4个月
%    例如：2023.10-2023.12 -> 2024.01
% ==========================================================

% 只为月尺度预测单独往前补3个月
start_date_m = dateshift(start_date,'start','month') - calmonths(3);   % 2023-10-01
end_date_m   = end_date;

idx_m = (date_all >= start_date_m) & (date_all <= end_date_m);

dates_m = datetime(data_all.("年")(idx_m), data_all.("月")(idx_m), data_all.("日")(idx_m));
Rain_m  = data_all.("降水量（mm）")(idx_m);
Temp_m  = data_all.("气温(℃)")(idx_m);

Rain_daily = timetable(dates_m, Rain_m);
Temp_daily = timetable(dates_m, Temp_m);

Temp_month = retime(Temp_daily,'monthly','mean');
Rain_month = retime(Rain_daily,'monthly','sum');

Temp_month = Temp_month.Variables;
Rain_month = Rain_month.Variables;
X_month = [Rain_month, Temp_month];   % 2023-10 ~ 2025-12，共27个月

load("LSTM_net.mat");

Rain_sum  = X_month(:,1)';
Temp_mean = X_month(:,2)';

nPredMonth = size(X_month,1) - 3;    % 27-3=24，对应 2024-01 ~ 2025-12

Rain1 = zeros(3,nPredMonth);
Temp1 = zeros(3,nPredMonth);

for j = 1:3
    Rain1(j,:) = Rain_sum(j : j+nPredMonth-1);
    Temp1(j,:) = Temp_mean(j : j+nPredMonth-1);
end

X_month1 = [Rain1; Temp1];

X_input = num2cell(X_month1,1);
C = predict(LSTM_net, X_input);
C = cell2mat(C);
C = reshape(C, 8, length(X_input)).';

C_TN = C(:,2);   % 2024-01 ~ 2025-12
C_TP = C(:,7);   % 2024-01 ~ 2025-12

%% =========================================================
% 5. 把常量打包，后面函数直接用 params
% ==========================================================
params = struct();
params.dates     = dates;
params.Qin_max   = Qin_max;
params.Qout_max  = Qout_max;
params.r_safe    = r_safe;
params.r_zero    = r_zero;
params.tol_alpha = tol_alpha;
params.Hmin      = Hmin;
params.Hmax      = Hmax;
params.C_TN      = C_TN;
params.C_TP      = C_TP;

%% =========================================================
% 6. NSGA-II 优化
%
% 决策变量：alpha(t) ∈ [0,1]
% alpha=0 -> 只调到可接受风险
% alpha=1 -> 尽量调到接近零风险
% ==========================================================
nVar = nDay;     % 每天一个 alpha

lb = zeros(1,nDay);
ub = ones(1,nDay);

options = optimoptions('gamultiobj', ...
    'PopulationSize', 100, ...
    'MaxGenerations', 180, ...
    'ParetoFraction', 0.5, ...
    'Display', 'iter', ...
    'FunctionTolerance', 1e-6, ...
    'PlotFcn', {@gaplotpareto});

[x, fval] = gamultiobj(@(alpha) objective_alpha(alpha, dw_mc, params), ...
    nVar, [], [], [], [], lb, ub, options);

%% =========================================================
% 7. 从 Pareto 解中挑不同偏好的方案
% ==========================================================
[prefTable, prefIdx] = select_preference_solutions(x, dw_mc, params);

disp('================ Pareto代表解 =================');
disp(prefTable);

%% =========================================================
% 8. 选一个你最想看的方案（这里先默认平衡解）
% ==========================================================
chooseName = "MinRisk";%"MinRisk"; "MinPollution"; "MinGateCost"; "Balanced"
best_idx = prefIdx.(chooseName);
alpha_best = x(best_idx,:).';

sim_best = decode_alpha_solution(alpha_best, dw_mc, params);

disp('================ 选中的方案 =================');
disp(['方案类型: ', char(chooseName)]);
disp(['对应 Pareto 解编号: ', num2str(best_idx)]);
disp('alpha_best = ');
disp(alpha_best.');

disp(['最大日风险概率 = ', num2str(max(sim_best.Risk_prob))]);
disp(['累计日风险概率 = ', num2str(sum(sim_best.Risk_prob))]);
disp(['总污染排放 = ', num2str(sim_best.pollution_total)]);
disp(['总开闸成本 = ', num2str(sim_best.gate_cost)]);

%% =========================================================
% 9. 输出每日结果表
% ==========================================================
result_table = table(dates, sim_best.Qin, sim_best.Qout, sim_best.Risk_prob, sim_best.W_ref, ...
    'VariableNames', {'Date','Qin','Qout','RiskProb','Wref'});
disp(result_table);

%% =========================================================
% 10. 可视化
% ==========================================================
% (1) 单纯概率风险
figure;
plot(dates, sim_best.Risk_prob, 'k-', 'LineWidth', 1.8); hold on;
yline(r_safe, 'r--', 'LineWidth', 1.5);
xlabel('Date');
ylabel('Daily probability risk');
title(['Daily probability risk - ', char(chooseName)]);
legend('Risk probability','r\_safe = 0.2','Location','best');
ylim([0 1]);
grid on;

% (2) 风险 + 进排水
figure;
yyaxis left
plot(dates, sim_best.Risk_prob, 'k-', 'LineWidth', 1.8);
ylabel('Daily probability risk');
ylim([0 1]);

yyaxis right
bar(dates, [sim_best.Qin, -sim_best.Qout], 'grouped');
ylabel('Inflow (+) / Outflow (-)');

xlabel('Date');
title(['Risk and gate operation - ', char(chooseName)]);
legend('Risk probability','Qin','Qout','Location','best');
grid on;

% (3) 每月污染排放
years = year(dates);
months = month(dates);
year_month = years*100 + months;
unique_months = unique(year_month);

figure;
bar(1:length(sim_best.monthly_pollution), sim_best.monthly_pollution);
set(gca,'XTick',1:length(unique_months), ...
        'XTickLabel', string(unique_months));
xlabel('Month');
ylabel('Monthly pollution discharge');
title(['Monthly pollution discharge - ', char(chooseName)]);
grid on;

% (4) 参考水量过程
figure;
plot(dates, sim_best.W_ref, 'LineWidth', 1.8); hold on;
yline(Hmax, 'r--', 'LineWidth', 1.5);
yline(Hmin, 'b--', 'LineWidth', 1.5);
xlabel('Date');
ylabel('Reference water storage');
title(['Reference water storage - ', char(chooseName)]);
legend('Wref','Hmax','Hmin','Location','best');
grid on;

save('optimization_results_2024-2025.mat', ...
    'x', 'fval', ...
    'alpha_best', 'sim_best', 'best_idx', 'prefTable', 'prefIdx', ...
    'result_table');

