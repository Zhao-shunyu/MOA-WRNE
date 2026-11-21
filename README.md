# A multi-objective optimization approach to managing water-level risk and nutrient export（MOA-WRNE）

MOA-WRNE is a multi-objective optimization method implemented in MATLAB for gate operations in lakes and reservoirs under climate change. It balances water-level risk, nitrogen and phosphorus export, and gate operation costs.

## Requirements

- MATLAB R2024a
- Global Optimization Toolbox
- Deep Learning Toolbox

## Installation

Download or clone this repository. Open MATLAB and set the repository folder as the current folder. To run the Xinghu Lake example, enter:

```matlab
result = run_moa_wrne;
```

## Description

MOA-WRNE combines relative water storage simulation, Monte Carlo analysis of runoff uncertainty, and LSTM-based prediction of monthly total nitrogen (TN) and total phosphorus (TP) concentrations. It uses MATLAB's `gamultiobj` solver, a variant of NSGA-II, to identify Pareto-optimal daily gate operation strategies.

The optimization minimizes three objectives:

- Cumulative probability of exceeding the upper or lower water-level limits.
- Total TN and TP loads exported through gate releases.
- Gate operation costs based on opening frequency and changes in daily flow.

Gate schedules are evaluated against daily inflow and outflow capacity limits, relative storage bounds, and a maximum daily exceedance probability of 20% in the Xinghu Lake example.

## Dataset

The repository includes daily meteorological observations for Xinghu Lake (`Weather_Xinghu.csv`) and a trained LSTM model (`LSTM_net.mat`). The default simulation covers January 2024 through December 2025.

To analyze another period or a future climate scenario, specify `WeatherFile`, `StartDate`, and `EndDate`. The weather file must use the same column names and units as the example and include the three full months preceding the first simulation month. Use bias-corrected daily climate inputs for future scenarios.

## Optimization Settings

| Setting | Default | Description |
| --- | ---: | --- |
| `PopulationSize` | 100 | Number of candidate schedules in each generation |
| `MaxGenerations` | 180 | Maximum number of optimization generations |
| `Seed` | 1 | Random seed for reproducibility |
| `OutputDir` | `moa_wrne_results` | Directory for the results |

For a quick run to check the workflow:

```matlab
result = run_moa_wrne(PopulationSize=8, MaxGenerations=2, OutputDir="quick_check");
```

## Results

The program reports four representative strategies: balanced operation, minimum water-level risk, minimum nutrient export, and minimum gate operation cost. It also evaluates a no-regulation baseline.

| File | Contents |
| --- | --- |
| `pareto.csv` | Feasible Pareto solutions and their three objective values |
| `strategies.csv` | Performance summary for the four representative strategies and the no-regulation baseline |
| `daily.csv` | Daily inflow, outflow, median relative storage, and exceedance probability |
| `monthly.csv` | Monthly TN and TP export loads in kilograms |
| `moa_wrne.mat` | MATLAB results, model parameters, and run settings |
