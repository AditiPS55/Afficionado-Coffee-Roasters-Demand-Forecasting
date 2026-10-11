import pandas as pd, numpy as np
def basic(g):
    e = g.pred - g.actual; nz = g.actual > 0
    return pd.Series({'MAE': e.abs().mean(), 'RMSE': np.sqrt((e**2).mean()),
                      'MAPE': (e.abs()[nz] / g.actual[nz]).mean()*100, 'WAPE': e.abs().sum()/g.actual.sum()*100,
                      'Bias%': e.sum()/g.actual.sum()*100})
def peak(g):
    """Peak hour = actual >= store's 75th-pct hourly volume (from training window)."""
    ap = g.actual >= g.peak_thr; pp = g.pred >= g.peak_thr
    cap = (pp & ap).sum() / max(ap.sum(), 1)
    prec = (pp & ap).sum() / max(pp.sum(), 1)
    # daily top-3 hour hit rate
    hits = []
    for _, d in g.groupby(['store_id','date']):
        d = d[d.actual > 0]
        if len(d) < 3: continue
        hits.append(len(set(d.nlargest(3,'actual').hour) & set(d.nlargest(3,'pred').hour)) / 3)
    return pd.Series({'Peak Capture Rate %': cap*100, 'Peak Error Rate %': (1-cap)*100, 'Peak Precision %': prec*100, 'Top-3 Hour Hit %': np.mean(hits)*100})
