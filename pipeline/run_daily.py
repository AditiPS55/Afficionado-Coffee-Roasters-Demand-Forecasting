"""Step 3: rolling-origin validation for daily revenue & quantity per store (short 7d, medium 30d)."""
import sys, time, pandas as pd, numpy as np
from common import *
daily = pd.read_csv('data/daily_panel.csv', parse_dates=['date'])
stores = daily[['store_id','store_location']].drop_duplicates().set_index('store_id').store_location.to_dict()
rows = []; t0 = time.time()
CFG = {'short': dict(h=7, origins=range(90, 175, 7)), 'medium': dict(h=30, origins=range(100, 152, 10))}
for target in ['revenue', 'qty']:
    S = {sid: daily[daily.store_id == sid].set_index('date')[target].asfreq('D') for sid in stores}
    for hz, cfg in CFG.items():
        for k, o in enumerate(cfg['origins']):
            train = {sid: s.iloc[:o] for sid, s in S.items()}
            fc = forecast_all(train, cfg['h'])
            for model, d in fc.items():
                for sid, p in d.items():
                    act = S[sid].iloc[o:o+cfg['h']]
                    for i in range(len(act)):
                        rows.append((target, hz, model, sid, stores[sid], k, o, i+1, act.index[i], act.iloc[i], p[i]))
        print(target, hz, 'done', round(time.time()-t0), 's', flush=True)
cv = pd.DataFrame(rows, columns=['target','horizon_type','model','store_id','store_location','fold','origin','step','date','actual','pred'])
cv.to_csv('data/cv_daily.csv', index=False); print('saved', cv.shape)
