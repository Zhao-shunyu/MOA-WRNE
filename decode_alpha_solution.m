% 给定 alpha，解码出每日进/排水量、风险、污染
function sim = decode_alpha_solution(alpha, dw_mc, params)

alpha = alpha(:);
alpha(alpha < params.tol_alpha) = 0;
alpha(alpha > 1-params.tol_alpha) = 1;

nDay = length(params.dates);
nMC  = size(dw_mc,2);

% 结果变量
u         = zeros(nDay,1);   % 最终控制量：正=进水，负=排水
Qin       = zeros(nDay,1);   % 每日进水量
Qout      = zeros(nDay,1);   % 每日排水量
W_ref     = zeros(nDay,1);   % 参考状态（单值）
Risk_prob = zeros(nDay,1);   % 每日风险概率
Risk_mag  = zeros(nDay,1);   % 每日超限幅度风险

for t = 1:nDay

    % 当前参考状态：不再用 Monte Carlo 全程累计
    if t == 1
        W_prev = 0;
    else
        W_prev = W_ref(t-1);
    end

    % 当天不操作时的水量分布
    W0 = W_prev + dw_mc(t,:);

    % 计算当天不操作风险
    p0      = sum(W0 < params.Hmin | W0 > params.Hmax) / nMC;
    p_high0 = sum(W0 > params.Hmax) / nMC;
    p_low0  = sum(W0 < params.Hmin) / nMC;

    % 默认：不开闸
    u_t = 0;
    Qin_t = 0;
    Qout_t = 0;

    % 如果当天风险已经在可接受范围内，则不开闸
    if p0 > params.r_safe

        % -------------------------------------------------
        % 候选动作 1：排水
        % -------------------------------------------------
        best_p = p0;   % 当前最优风险（先设成不操作）
        if p_high0 >= p_low0

            % 理论最小必要排水量：把高水风险压到 r_safe
            q_req = quantile(W0, 1-params.r_safe) - params.Hmax;
            q_req = max(q_req, 0);

            % 更激进排水量：把高水风险压到接近零
            q_zero = quantile(W0, 1-params.r_zero) - params.Hmax;
            q_zero = max(q_zero, q_req);

            % 按 alpha 在 q_req 和 q_zero 之间插值
            q_try = q_req + alpha(t) * (min(q_zero, params.Qout_max) - q_req);
            q_try = min(q_try, params.Qout_max);

            % 计算该排水动作后的总风险
            W_try = W0 - q_try;
            p_try = sum(W_try < params.Hmin | W_try > params.Hmax) / nMC;

            % 如果按 alpha 的量仍未压到 r_safe，则尝试最大排水量
            if p_try > params.r_safe
                W_max = W0 - params.Qout_max;
                p_max = sum(W_max < params.Hmin | W_max > params.Hmax) / nMC;
                if p_max < p_try
                    q_try = params.Qout_max;
                    p_try = p_max;
                end
            end

            % 只有风险真的下降了，才接受排水
            if p_try < best_p
                best_p = p_try;
                u_t = -q_try;
                Qin_t = 0;
                Qout_t = q_try;
            end
        end

        % -------------------------------------------------
        % 候选动作 2：进水
        % -------------------------------------------------
        if p_low0 > p_high0

            q_req = params.Hmin - quantile(W0, params.r_safe);
            q_req = max(q_req, 0);

            q_zero = params.Hmin - quantile(W0, params.r_zero);
            q_zero = max(q_zero, q_req);

            q_try = q_req + alpha(t) * (min(q_zero, params.Qin_max) - q_req);
            q_try = min(q_try, params.Qin_max);

            W_try = W0 + q_try;
            p_try = sum(W_try < params.Hmin | W_try > params.Hmax) / nMC;

            if p_try > params.r_safe
                W_max = W0 + params.Qin_max;
                p_max = sum(W_max < params.Hmin | W_max > params.Hmax) / nMC;
                if p_max < p_try
                    q_try = params.Qin_max;
                    p_try = p_max;
                end
            end

            if p_try < best_p
                best_p = p_try;
                u_t = q_try;
                Qin_t = q_try;
                Qout_t = 0;
            end
        end
    end

    % 保存当天动作
    u(t)    = u_t;
    Qin(t)  = Qin_t;
    Qout(t) = Qout_t;

    % 当天执行后的风险分布
    W_today = W_prev + dw_mc(t,:) + u_t;

    % 每日风险
    Risk_prob(t) = sum(W_today < params.Hmin | W_today > params.Hmax) / nMC;

    % 超限幅度
    exceed_high = max(W_today - params.Hmax, 0);
    exceed_low  = max(params.Hmin - W_today, 0);
    Risk_mag(t) = mean(exceed_high + exceed_low);

    % 参考状态更新：只用均值推进，避免后期分布无限变散
    W_ref(t) = W_prev + mean(dw_mc(t,:)) + u_t;
end

% ---------------------------------------------------------
% 月污染排放
% ---------------------------------------------------------
years = year(params.dates);
months = month(params.dates);
year_month = years * 100 + months;
unique_months = unique(year_month);
nMonths = length(unique_months);

monthly_pollution = zeros(nMonths,1);

for m = 1:nMonths
    idx = (year_month == unique_months(m));
    TN_flux = sum(Qout(idx)) * params.C_TN(m);
    TP_flux = sum(Qout(idx)) * params.C_TP(m);
    monthly_pollution(m) = TN_flux + TP_flux;
end

pollution_total = sum(monthly_pollution);

% ---------------------------------------------------------
% 开闸成本
% ---------------------------------------------------------
open_flag = abs(u) > 1e-6;
gate_cost = ...
      0.1 * sum(abs(u)) ...
    + 100 * sum(open_flag) ...
    + 50  * sum(abs(diff(open_flag))) ...
    + 0.001 * sum(abs(diff(u)));

% 打包输出
sim = struct();
sim.alpha = alpha;
sim.u = u;
sim.Qin = Qin;
sim.Qout = Qout;
sim.W_ref = W_ref;
sim.Risk_prob = Risk_prob;
sim.Risk_mag = Risk_mag;
sim.monthly_pollution = monthly_pollution;
sim.pollution_total = pollution_total;
sim.gate_cost = gate_cost;
end