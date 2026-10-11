import numpy as np, pandas as pd, lightgbm as lgb, warnings
warnings.filterwarnings('ignore')
HRS = list(range(6, 21))   # operating window across all stores (06:00-20:59)

def load_arrays(target='transactions'):
    h = pd.read_csv('data/hourly_panel.csv', parse_dates=['ts','date'])
    stores = sorted(h.store_id.unique()); days = sorted(h.date.unique())
    A = {s: h[h.store_id == s].pivot(index='date', columns='hour', values=target).reindex(columns=range(24)).values for s in stores}
    loc = h[['store_id','store_location']].drop_duplicates().set_index('store_id').store_location.to_dict()
    # data-quality flag: store-days where the whole 07:00-10:59 block is empty (Astoria, days 1-6 of every month)
    global GD
    GD = {s: (A[s][:, 7:11].sum(axis=1) == 0) for s in stores}
    return A, pd.DatetimeIndex(days), loc

def feats(A, s, code, di, hr, dates, lags, lw_end):
    a = A[s]
    L = a[di-lw_end-6: di-lw_end+1].sum(axis=1).mean()
    lg = [np.nan if (7 <= hr <= 10 and GD[s][di-l]) else a[di-l, hr] for l in lags]   # mask lags that fall on gap days
    lm = np.nanmean(lg) if not np.all(np.isnan(lg)) else np.nan
    f = [x / L for x in lg] + [lm/L, hr, dates[di].dayofweek, code, int(dates[di].day <= 6)]
    return f, L

CFG = {'7d': dict(lags=[7,14,21], lw_end=7, min_di=21), '1d': dict(lags=[1,2,7], lw_end=1, min_di=14)}

def build(A, dates, rows_di, cfg, with_y=True):
    X, Y, meta = [], [], []
    for code, s in enumerate(sorted(A)):
        for di in rows_di:
            for hr in HRS:
                f, L = feats(A, s, code, di, hr, dates, cfg['lags'], cfg['lw_end'])
                X.append(f); meta.append((s, di, hr, L)); Y.append(A[s][di, hr] / L)
    return np.array(X), np.array(Y), meta

def fit_models(A, dates, o, cfg, quantiles=True):
    X, Y, _ = build(A, dates, range(cfg['min_di'], o), cfg)
    kw = dict(n_estimators=200, learning_rate=0.05, num_leaves=8, min_child_samples=15, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.9, verbose=-1, random_state=0)
    M = {'mean': lgb.LGBMRegressor(**kw).fit(X, Y)}
    if quantiles:
        M['lo'] = lgb.LGBMRegressor(objective='quantile', alpha=0.1, **kw).fit(X, Y)
        M['hi'] = lgb.LGBMRegressor(objective='quantile', alpha=0.9, **kw).fit(X, Y)
    return M

def predict_rows(M, A, dates, days_di, cfg):
    X, _, meta = build_pred(A, dates, days_di, cfg)
    out = {k: m.predict(X) for k, m in M.items()}
    res = pd.DataFrame(meta, columns=['store_id','di','hour','L'])
    for k, v in out.items(): res[k] = np.maximum(v, 0) * res.L
    if 'lo' in res: res['lo'] = np.minimum(res.lo, res['mean']); res['hi'] = np.maximum(res.hi, res['mean'])
    return res

def build_pred(A, dates, days_di, cfg):
    X, meta = [], []
    for code, s in enumerate(sorted(A)):
        for di in days_di:
            for hr in HRS:
                f, L = feats(A, s, code, di, hr, dates, cfg['lags'], cfg['lw_end']); X.append(f); meta.append((s, di, hr, L))
    return np.array(X), None, meta
