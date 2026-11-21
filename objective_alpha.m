% 多目标函数
function f = objective_alpha(alpha, dw_mc, params)

sim = decode_alpha_solution(alpha, dw_mc, params);

% 目标1：风险
% 思路：先压最大日风险，再压累计风险，再压超限幅度
risk_obj = 1e6 * max(sim.Risk_prob) + ...
           1e3 * sum(sim.Risk_prob) + ...
                 sum(sim.Risk_mag);

% 目标2：污染排放
pollution_obj = sim.pollution_total;

% 目标3：开闸成本
gate_obj = sim.gate_cost;

f = [risk_obj;
     pollution_obj;
     gate_obj];
end