import warnings, logging
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
logging.getLogger('cmdstanpy').setLevel(logging.ERROR); logging.getLogger('prophet').setLevel(logging.ERROR)

# ---------------- daily univariate models: f(train: Series, h) -> ndarray ----------------
def _early(idx): return np.asarray(idx.day <= 6, dtype=float).reshape(-1, 1)   # early-month regime flag (days 1-6)
def m_naive(y, h):  return np.repeat(y.iloc[-1], h)
def m_snaive(y, h):
    last = y.iloc[-7:].values; return np.array([last[i % 7] for i in range(h)])
def m_ma7(y, h):    return np.repeat(y.iloc[-7:].mean(), h)

def m_ets(y, h):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    f = ExponentialSmoothing(y.values, trend='add', damped_trend=True, seasonal='add',
                             seasonal_periods=7, initialization_method='estimated').fit()
    return f.forecast(h)

def m_sarima(y, h):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    fi = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=h)
    ex, exf = _early(y.index), _early(fi)
    f = SARIMAX(y.values, exog=ex, order=(1,1,1), seasonal_order=(1,0,0,7),
                enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=100)
    return f.forecast(h, exog=exf)

def m_prophet(y, h):
    from prophet import Prophet
    d = pd.DataFrame({'ds': y.index, 'y': y.values}); d['early'] = (d.ds.dt.day <= 6).astype(float)
    m = Prophet(weekly_seasonality=True, yearly_seasonality=False, daily_seasonality=False,
                n_changepoints=10, changepoint_prior_scale=0.3, interval_width=0.8)
    m.add_regressor('early', mode='multiplicative'); m.fit(d)
    fut = pd.DataFrame({'ds': pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=h)}); fut['early'] = (fut.ds.dt.day <= 6).astype(float)
    return m.predict(fut)['yhat'].values

# ---------------- pooled LightGBM, ratio-to-level target, recursive ----------------
def _feat(hist, t_date, sid_code):
    # hist: list/array of past values up to t-1 (len>=14)
    h = np.asarray(hist); m = h[-7:].mean()
    prev = h[-14:-7].mean()
    ne = sum((t_date - pd.Timedelta(days=k)).day <= 6 for k in range(1, 8))
    return [h[-1]/m, h[-7]/m, h[-3:].mean()/m, m/prev, t_date.dayofweek, sid_code, int(t_date.day <= 6), ne], m

def gbm_fit(series_by_store):
    import lightgbm as lgb
    X, Y = [], []
    for code, (sid, y) in enumerate(series_by_store.items()):
        v = y.values
        for t in range(14, len(v)):
            f, m = _feat(v[:t], y.index[t], code); X.append(f); Y.append(v[t]/m)
    mdl = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.05, num_leaves=6, min_child_samples=8,
                            subsample=0.8, subsample_freq=1, colsample_bytree=0.9, verbose=-1, random_state=0)
    mdl.fit(np.array(X), np.array(Y)); return mdl

def gbm_predict(mdl, series_by_store, h):
    out = {}
    for code, (sid, y) in enumerate(series_by_store.items()):
        hist = list(y.values); dates = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=h); p = []
        for d in dates:
            f, m = _feat(hist, d, code); v = float(mdl.predict(np.array([f]))[0]) * m
            p.append(v); hist.append(v)
        out[sid] = np.array(p)
    return out

UNI = {'Naive': m_naive, 'Seasonal Naive (7d)': m_snaive, 'Moving Average (7d)': m_ma7,
       'Holt-Winters (ETS)': m_ets, 'SARIMA': m_sarima, 'Prophet': m_prophet}
ALL_MODELS = list(UNI) + ['Gradient Boosting']

def forecast_all(train_by_store, h, models=ALL_MODELS):
    """returns {model: {store: array(h)}}"""
    res = {}
    for name in models:
        if name == 'Gradient Boosting':
            res[name] = gbm_predict(gbm_fit(train_by_store), train_by_store, h); continue
        res[name] = {}
        for sid, y in train_by_store.items():
            try: res[name][sid] = np.asarray(UNI[name](y, h), dtype=float)
            except Exception as e: res[name][sid] = m_naive(y, h)   # fallback, flagged below
    return res
