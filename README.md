# Afficionado Coffee Roasters — Demand Forecasting

## Run the dashboard
```
pip install -r requirements.txt
streamlit run app.py
```
`app.py` only needs the CSV/JSON files in `data/` (forecasts and validation results are pre-computed).

## Re-run the pipeline (optional)
Place `Afficionado_Coffee_Roasters.xlsx` where `pipeline/prep.py` expects it (edit `SRC`), then from a folder containing `pipeline/*.py` and an empty `data/`:
```
pip install -r pipeline/requirements.txt
python prep.py && python run_daily.py && python run_hourly.py && python run_final.py && python analysis.py
```
| Script | Purpose |
|---|---|
| prep.py | Reconstruct dates, build hourly / daily / category panels |
| common.py | Daily models: Naive, Seasonal Naive, MA(7), Holt-Winters, SARIMA, Prophet, LightGBM |
| hourly_lib.py | Hourly LightGBM (lag 7/14/21 days, level-normalised, quantile intervals) |
| run_daily.py / run_hourly.py | Rolling-origin validation |
| run_final.py | July 2025 forecasts, intervals, category models |
| analysis.py | Metrics, KPIs, scenarios, figures |

## Key assumption
The source file has no date column. Dates were reconstructed from row order (time-of-day resets 180 times → 181 days) and anchored to 1 Jan 2025.
