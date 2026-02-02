# CLAUDE.md - CAISO Price Forecast Project Guide

## Project Overview

This project forecasts California Independent System Operator (CAISO) wholesale electricity prices using time-series models. It compares three approaches — historic average (baseline), ARIMA, and LSTM neural networks — for 10-day hourly price prediction across CAISO's three main trading hubs: NP15 (Northern California), SP15 (Southern California), and ZP26 (Central Coast).

Key result: LSTM models achieve 40-50% lower RMSE than ARIMA across all hubs.

## Repository Structure

```
caiso-price-forecast/
├── Final.ipynb                  # Main notebook: multivariate LSTM analysis (active development)
├── README.md                    # Project documentation with findings
├── CAISO_Price_Forecast_MSell_vFINAL.pdf  # Final presentation/report
├── src/
│   ├── import_process_data.py   # Data pipeline: scraping, cleaning, merging
│   ├── eda.py                   # Exploratory data analysis & visualizations
│   ├── model.py                 # LSTM, ARIMA, and baseline model implementations
│   └── arima.py                 # ARIMA-specific analysis (decomposition, autocorrelation)
├── data/
│   ├── caiso_master.csv         # Merged master dataset (~11,500 hourly rows)
│   ├── caiso_lmp_nodes/         # 16 monthly LMP CSV files from CAISO OASIS
│   ├── np15_lmp_19_20.csv       # NP15 hub price curve
│   ├── sp15_lmp_19_20.csv       # SP15 hub price curve
│   ├── zp26_lmp_19_20.csv       # ZP26 hub price curve
│   ├── caiso_load_jan_19_may_20.csv      # Hourly consumption (MWh)
│   ├── caiso_gen_jan_19_may_20.csv       # Hourly generation by fuel type
│   ├── caiso_net_ex_jan_19_may_20.csv    # Hourly net exports (MW)
│   ├── natgas_jan_19_may_20.csv          # Daily Henry Hub natural gas prices
│   └── nat_gas.xls              # Natural gas source data
├── images/                      # Visualization outputs (PNG files)
├── preso/                       # Presentation materials
└── archive/                     # Historical/superseded notebooks
    ├── arima.ipynb
    ├── caiso-eda.ipynb
    ├── caiso-oasis-scrape.ipynb
    ├── data_process.ipynb
    ├── pres_readme_eda.ipynb
    └── rnn.ipynb
```

## Tech Stack

- **Python 3.11**
- **Data**: pandas, numpy
- **ML/Modeling**: tensorflow/keras (LSTM), statsmodels (ARIMA), scikit-learn (metrics)
- **Visualization**: matplotlib, seaborn
- **Data acquisition**: pyiso (CAISO OASIS scraper by WattTime)
- **Statistics**: scipy (signal processing, stats)

### Required packages

pandas, numpy, pyiso, matplotlib, tensorflow, seaborn, statsmodels, scikit-learn, scipy

## Data Pipeline

### Sources
1. **CAISO OASIS** — Locational Marginal Prices (LMPs), generation, load, net exports
2. **FRED** — Henry Hub daily natural gas spot prices

### Processing flow
```
CAISO OASIS / FRED CSVs
  → create_price_curves()           # Parse 16 monthly LMP files, extract hub data
  → scrape_process_caiso_*()        # Load, generation, net export via pyiso
  → obtain_format_hh_natgas_to_df() # Natural gas from FRED
  → create_caiso_master_df()        # Merge all sources, align to hourly
  → save_caiso_df_to_csv()          # Export caiso_master.csv
```

### Key data transformations
- 15-minute generation/load data aggregated to hourly via pivot tables
- Natural gas prices forward-filled for weekends/holidays (markets closed)
- GMT timestamps converted to Pacific Time (PT) with `timedelta(hours=7)`
- May 1-4, 2020 data error handled by forward-filling from April 30
- All datetime columns stripped of timezone info via `.replace(tzinfo=None)`

### Master dataset schema (caiso_master.csv)
- **Index**: `INTERVAL_START_PT` (hourly timestamps)
- **Time columns**: `INTERVAL_END_PT`, `OPR_DT_PT`, `OPR_HR_PT`, `OPR_INTERVAL`, `day_week`
- **Prices**: `$_MWH_np15`, `$_MWH_sp15`, `$_MWH_zp26` ($/MWh)
- **Generation**: `solar`, `wind`, `other`, `total_mw` (MWh)
- **Demand**: `load_MW` (MW)
- **Trade**: `net_exp_MW` (MW)
- **Fuel**: `HH_$_million_BTU_not_seasonal_adj` ($/MMBtu)

## Module Reference

### src/import_process_data.py
Data ETL pipeline. Key functions:
- `create_price_curves()` — Loads 16 monthly CSV files, filters by LMP type, extracts NP15/SP15/ZP26 nodes
- `scrape_process_caiso_load_data(iso_class, start, end)` — Hourly demand via pyiso
- `scrape_process_caiso_generation_data(iso_class, start, end)` — Generation by fuel type
- `scrape_process_caiso_net_ex_data(iso_class, start, end)` — Net exports/imports
- `obtain_format_hh_natgas_to_df()` — FRED natural gas prices
- `create_caiso_master_df(np15, sp15, zp26, gen, load, net_ex, natgas)` — Merge all datasets
- `import_caiso_dataset(version_name)` — Load CSV from `data/` directory

### src/model.py
Model implementations. Key functions:
- `calc_rmse(actual, pred)` — Root mean squared error
- `baseline_fcst(lmp_curve, n_periods_fcst)` — Historic average forecast
- `arima_uni_var_train_valid_split(lmp_curve, date_rng, train_split_idx)` — Train/validation split
- `arima_uni_var_fit(lmp_train, date_rng, p, d, q)` — Fit ARIMA model
- `arima_uni_var_predict(model, n_period_fcst)` — ARIMA forecast
- `windowize_data(data, n_prev)` — Create sliding sequences for LSTM
- `split_and_windowize(data, n_prev, fraction_valid)` — Split + create windows
- `compile_and_fit_lstm_uni_var(X_train, y_train, batch_size, n_nodes=32, n_epochs=20)` — 3-layer LSTM
- `plot_actual_arima_baselie_lstm(...)` — Comparative forecast visualization

### src/eda.py
Visualization functions:
- `import_process_data_for_eda()` — Prepares data with May 2020 error fix
- `plot_day_ahead_hourly_prices()` — Scatter plot of hourly prices
- `plot_price_curve_box_plot()` — Distribution by hour of day
- `draw_lag_plots()` — 24-hour lag autocorrelation plots
- `plot_prcnt_re_gen_moving_avg()` — Renewable generation trend

### src/arima.py
Statistical analysis:
- `fit_moving_average_trend()`, `fit_seasonal_trend()` — Trend fitting
- `plot_seasonal_decomposition()` — Trend/seasonal/residual decomposition
- `compute_autocorrelation()` — Lag correlation
- `plot_lmp_curve_autocorrelation()` — ACF visualization

## Model Architecture Details

### LSTM (primary model)
- 3-layer sequential LSTM with configurable nodes (default: 32 per layer)
- `return_sequences=True` for first two layers, `False` for third
- Dense output layer with linear activation
- Adam optimizer, MSE loss function
- MinMaxScaler preprocessing significantly improves RMSE
- Must call `inv_transform()` on predictions to reverse scaling

### ARIMA
- Order `(p, d, q)` tuned per hub
- Uses statsmodels `ARIMA` with hourly frequency (`freq='H'`)
- Performance degrades beyond ~5 days in forecast horizon

### Test period
- Last 10 days of May 2020 (240 hours)
- RMSE in $/MWh for direct comparison to electricity prices

## Naming Conventions

- Hub suffixes: `_np15`, `_sp15`, `_zp26`
- Price columns: `$_MWH` prefix (e.g., `$_MWH_np15`)
- Time columns: `OPR_DT_PT` (date), `OPR_HR_PT` (hour), `INTERVAL_START_PT` (timestamp)
- Generation: `solar`, `wind`, `other`, `total_mw`
- CAISO node IDs: `NP15SLAK_5_N001`, `SP26SLAK_5_N001`, `ZP26SLAK_5_N001`
- Functions use snake_case
- Docstrings follow numpy-style format with Parameters/Returns sections

## Active Development

Current focus is on **multivariate LSTM models** (see `Final.ipynb` and recent commits). This extends the univariate approach by incorporating supply/demand factors, fuel prices, and market dynamics as additional input features.

The `archive/` directory contains earlier notebook iterations that have been consolidated into the main `Final.ipynb`.

## Working with the Codebase

- Source modules in `src/` use relative paths (`../data/`) — they expect to be run from the `src/` directory or imported from notebooks at the project root
- The main notebook `Final.ipynb` is the primary workspace for experimentation
- `data/caiso_master.csv` is the pre-built merged dataset; regenerating it requires CAISO OASIS access via pyiso
- `model.py` contains a `%matplotlib inline` magic command at line 4, indicating it was developed in a notebook context
- Visualizations are saved to the `images/` directory for use in README and presentations
