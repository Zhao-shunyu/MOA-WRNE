%从 Pareto 解中挑不同偏好的代表方案
%
% 输出：
% prefTable：不同偏好的代表方案表
% prefIdx：对应的 Pareto 解编号
function [prefTable, prefIdx] = select_preference_solutions(x, dw_mc, params)

nSol = size(x,1);

maxRisk   = zeros(nSol,1);
sumRisk   = zeros(nSol,1);
pollution = zeros(nSol,1);
gateCost  = zeros(nSol,1);

for i = 1:nSol
    sim_i = decode_alpha_solution(x(i,:).', dw_mc, params);
    maxRisk(i)   = max(sim_i.Risk_prob);
    sumRisk(i)   = sum(sim_i.Risk_prob);
    pollution(i) = sim_i.pollution_total;
    gateCost(i)  = sim_i.gate_cost;
end

% 1. 风险优先
[~, idxRisk] = sortrows([maxRisk, sumRisk, pollution, gateCost], [1 2 3 4]);
idxRisk = idxRisk(1);

% 2. 排放优先
[~, idxPoll] = sortrows([pollution, maxRisk, gateCost], [1 2 3]);
idxPoll = idxPoll(1);

% 3. 开闸优先
[~, idxGate] = sortrows([gateCost, maxRisk, pollution], [1 2 3]);
idxGate = idxGate(1);

% 4. 平衡解：归一化后离原点最近
objNorm = [ ...
    (maxRisk   - min(maxRisk))   ./ max(max(maxRisk)-min(maxRisk), eps), ...
    (pollution - min(pollution)) ./ max(max(pollution)-min(pollution), eps), ...
    (gateCost  - min(gateCost))  ./ max(max(gateCost)-min(gateCost), eps) ];
dist = sqrt(sum(objNorm.^2, 2));
[~, idxBal] = min(dist);

prefIdx = struct();
prefIdx.MinRisk      = idxRisk;
prefIdx.MinPollution = idxPoll;
prefIdx.MinGateCost  = idxGate;
prefIdx.Balanced     = idxBal;

prefTable = table( ...
    ["MinRisk"; "MinPollution"; "MinGateCost"; "Balanced"], ...
    [idxRisk; idxPoll; idxGate; idxBal], ...
    [maxRisk(idxRisk); maxRisk(idxPoll); maxRisk(idxGate); maxRisk(idxBal)], ...
    [sumRisk(idxRisk); sumRisk(idxPoll); sumRisk(idxGate); sumRisk(idxBal)], ...
    [pollution(idxRisk); pollution(idxPoll); pollution(idxGate); pollution(idxBal)], ...
    [gateCost(idxRisk); gateCost(idxPoll); gateCost(idxGate); gateCost(idxBal)], ...
    'VariableNames', {'Preference','SolutionID','MaxDailyRisk','SumDailyRisk','PollutionTotal','GateCost'} );
end