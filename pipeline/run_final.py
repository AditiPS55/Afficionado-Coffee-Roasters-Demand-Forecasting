"""Step 4: final forecasts (Jul 2025), category-level models, empirical prediction intervals, KPIs."""
import pandas as pd, numpy as np, json, time
from common import *
import hourly_lib as HL
from metrics import basic, peak
t0 = time.time()
daily = pd.read_csv('data/daily_panel.csv', parse_dates=['date'])
stores = daily[['store_id','store_location']].drop_duplicates().set_index('store_id').store_location.to_dict()
cvd = pd.read_csv('data/cv_daily.csv', parse_dates=['date'])
cvd['rel'] = cvd.actual / cvd.pred - 1

# ---------- (1) daily final forecasts, all models, 30 days, empirical 80% intervals ----------
def bucket(s): return np.select([s<=7, s<=14, s<=21], [1,2,3], 4)
cvd['b'] = bucket(cvd.step)
qs = cvd[cvd.horizon_type=='medium'].groupby(['target','model','b']).rel.quantile([0.1, 0.9]).unstack().rename(columns={0.1:'q10',0.9:'q90'}).reset_index()
rows = []
for target in ['revenue','qty']:
    S = {sid: daily[daily.store_id==sid].set_index('date')[target].asfreq('D') for sid in stores}
    fc = forecast_all(S, 30)
    for model, d in fc.items():
        for sid, p in d.items():
            dates = pd.date_range('2025-07-01', periods=30)
            for i in range(30):
                q = qs[(qs.target==target)&(qs.model==model)&(qs.b==bucket(np.array([i+1]))[0])].iloc[0]
                rows.append((target, model, sid, stores[sid], dates[i], i+1, p[i], p[i]*(1+q.q10), p[i]*(1+q.q90)))
fd = pd.DataFrame(rows, columns=['target','model','store_id','store_location','date','step','yhat','lo','hi'])
fd.to_csv('data/forecast_daily.csv', index=False); print('daily final', fd.shape, round(time.time()-t0),'s', flush=True)

# ---------- (2) hourly final forecasts Jul 1-7 (GBM + intervals), 3 targets ----------
hrows = []
for target in ['transactions','qty','revenue']:
    A, dates, loc = HL.load_arrays(target)
    dates_ext = dates.append(pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=7))
    cfg = HL.CFG['7d']; M = HL.fit_models(A, dates_ext, len(dates), cfg)
    P = HL.predict_rows(M, A, dates_ext, list(range(len(dates), len(dates)+7)), cfg)
    P['date'] = dates_ext[P.di]; P['target'] = target; P['store_location'] = P.store_id.map(loc)
    hrows.append(P.rename(columns={'mean':'yhat'})[['target','store_id','store_location','date','hour','yhat','lo','hi']])
fh = pd.concat(hrows); fh.to_csv('data/forecast_hourly.csv', index=False); print('hourly final', fh.shape, flush=True)

# ---------- (3) category-level daily demand (qty): CV + final ----------
cat = pd.read_csv('data/category_daily.csv', parse_dates=['date'])
top = ['Coffee','Tea','Bakery','Drinking Chocolate','Flavours']
CM = ['Naive','Seasonal Naive (7d)','Moving Average (7d)','Holt-Winters (ETS)','Prophet']
crow = []; frow = []
for c in top:
    S = {sid: cat[(cat.store_id==sid)&(cat.product_category==c)].set_index('date').qty.asfreq('D') for sid in stores}
    for k, o in enumerate(range(90, 175, 7)):
        fc = forecast_all({s: v.iloc[:o] for s, v in S.items()}, 7, CM)
        for m, d in fc.items():
            for sid, p in d.items():
                a = S[sid].iloc[o:o+7]
                for i in range(len(a)): crow.append((c, m, sid, stores[sid], k, i+1, a.iloc[i], p[i]))
ccv = pd.DataFrame(crow, columns=['category','model','store_id','store_location','fold','step','actual','pred'])
ccv.to_csv('data/cv_category.csv', index=False)
best = ccv.groupby(['category','model']).apply(lambda g: (g.pred-g.actual).abs().sum()/g.actual.sum()*100).rename('WAPE').reset_index()
bm = best.loc[best.groupby('category').WAPE.idxmin()].set_index('category').model.to_dict(); print('best category models', bm, flush=True)
for c in top:
    S = {sid: cat[(cat.store_id==sid)&(cat.product_category==c)].set_index('date').qty.asfreq('D') for sid in stores}
    fc = forecast_all(S, 30, [bm[c]])[bm[c]]
    for sid, p in fc.items():
        for i, d in enumerate(pd.date_range('2025-07-01', periods=30)): frow.append((c, bm[c], sid, stores[sid], d, i+1, max(p[i],0)))
pd.DataFrame(frow, columns=['category','model','store_id','store_location','date','step','yhat']).to_csv('data/forecast_category.csv', index=False)
print('done', round(time.time()-t0), 's')
