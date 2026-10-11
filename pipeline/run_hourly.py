"""Step 3b: hourly rolling-origin validation (transactions per store-hour): 7-day-ahead and next-day variants."""
import pandas as pd, numpy as np, time
from hourly_lib import *
A, dates, loc = load_arrays('transactions'); stores = sorted(A); rows = []; t0 = time.time()
def baselines(o, d, s, hr, kind):
    a = A[s]
    return {'Naive': a[o-1, hr] if kind == '7d' else a[d-1, hr],
            'Seasonal Naive (168h)': a[d-7, hr],
            'Moving Average (3-wk same hour)': np.mean([a[d-7, hr], a[d-14, hr], a[d-21, hr]])}
for kind, origins, H in [('7d', range(90, 175, 7), 7), ('1d', range(90, 181), 1)]:
    cfg = CFG[kind]
    for k, o in enumerate(origins):
        M = fit_models(A, dates, o, cfg)
        days = list(range(o, min(o+H, len(dates))))
        P = predict_rows(M, A, dates, days, cfg)
        thr = {s: np.percentile(np.concatenate([A[s][:o, hr] for hr in HRS]), 75) for s in stores}
        for r in P.itertuples():
            s, di, hr = r.store_id, r.di, r.hour; act = A[s][di, hr]
            preds = baselines(o, di, s, hr, kind); preds['Gradient Boosting'] = r.mean
            for m, p in preds.items():
                rows.append((kind, m, s, loc[s], k, o, di, di-o+1, dates[di], hr, act, p,
                             r.lo if m == 'Gradient Boosting' else np.nan, r.hi if m == 'Gradient Boosting' else np.nan, thr[s]))
    print(kind, 'done', round(time.time()-t0), 's', flush=True)
cv = pd.DataFrame(rows, columns=['horizon_type','model','store_id','store_location','fold','origin','di','step','date','hour','actual','pred','lo','hi','peak_thr'])
cv.to_csv('data/cv_hourly.csv', index=False); print(cv.shape)
