import pandas as pd, numpy as np, json
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from metrics import basic, peak
plt.rcParams.update({'figure.dpi':130,'axes.spines.top':False,'axes.spines.right':False,'font.size':9})
COL = {'Astoria':'#2E6F95','Hell\'s Kitchen':'#C8553D','Lower Manhattan':'#588157'}
daily = pd.read_csv('data/daily_panel.csv', parse_dates=['date'])
cvd = pd.read_csv('data/cv_daily.csv', parse_dates=['date']); cvh = pd.read_csv('data/cv_hourly.csv', parse_dates=['date'])
fd = pd.read_csv('data/forecast_daily.csv', parse_dates=['date']); fh = pd.read_csv('data/forecast_hourly.csv', parse_dates=['date'])
tx = pd.read_csv('data/transactions_dated.csv', parse_dates=['date'])
out = {}

# ---- metrics tables ----
md = cvd.groupby(['target','horizon_type','model']).apply(basic).reset_index(); md.to_csv('data/metrics_daily.csv', index=False)
mds = cvd.groupby(['target','horizon_type','model','store_location']).apply(basic).reset_index(); mds.to_csv('data/metrics_daily_by_store.csv', index=False)
mh = cvh.groupby(['horizon_type','model']).apply(basic).join(cvh.groupby(['horizon_type','model']).apply(peak)).reset_index(); mh.to_csv('data/metrics_hourly.csv', index=False)
mhs = cvh.groupby(['horizon_type','model','store_location']).apply(basic).join(cvh.groupby(['horizon_type','model','store_location']).apply(peak)).reset_index(); mhs.to_csv('data/metrics_hourly_by_store.csv', index=False)
pd.set_option('display.width',200); pd.set_option('display.max_columns',20)
print(mh.round(2).to_string()); 

# ---- KPIs (daily revenue, best model per horizon) ----
best_s = md[(md.target=='revenue')&(md.horizon_type=='short')].sort_values('MAPE').iloc[0]
best_m = md[(md.target=='revenue')&(md.horizon_type=='medium')].sort_values('MAPE').iloc[0]
out['best_short_model'] = best_s.model; out['best_medium_model'] = best_m.model
out['forecast_accuracy_short_%'] = 100 - best_s.WAPE; out['forecast_accuracy_medium_%'] = 100 - best_m.WAPE
out['wape_short'] = best_s.WAPE; out['wape_medium'] = best_m.WAPE
out['mape_short'] = best_s.MAPE; out['mape_medium'] = best_m.MAPE
naive_s = md[(md.target=='revenue')&(md.horizon_type=='short')&(md.model=='Naive')].MAE.iloc[0]
out['mae_improvement_vs_naive_short_%'] = (1 - best_s.MAE/naive_s)*100
# revenue forecast error (cumulative window)
def cum_err(model, hz):
    g = cvd[(cvd.target=='revenue')&(cvd.horizon_type==hz)&(cvd.model==model)].groupby(['store_id','fold']).agg(a=('actual','sum'), p=('pred','sum'))
    e = (g.p-g.a)/g.a*100; return e.abs().mean(), e.mean()
out['rev_err_7d_abs%'], out['rev_err_7d_bias%'] = cum_err(best_s.model, 'short')
out['rev_err_30d_abs%'], out['rev_err_30d_bias%'] = cum_err(best_m.model, 'medium')
out['rev_err_30d_abs_MA7%'], out['rev_err_30d_bias_MA7%'] = cum_err('Moving Average (7d)', 'medium')
# lead time accuracy
lt = cvd[(cvd.target=='revenue')&(cvd.horizon_type=='medium')&(cvd.model==best_m.model)].assign(ape=lambda d:(d.pred-d.actual).abs()/d.actual*100)
lt['bucket'] = pd.cut(lt.step,[0,7,14,21,30],labels=['1-7','8-14','15-21','22-30']); out['lead_time_mape_prophet'] = lt.groupby('bucket').ape.mean().round(2).to_dict()
lts = cvd[(cvd.target=='revenue')&(cvd.horizon_type=='short')&(cvd.model==best_s.model)].assign(ape=lambda d:(d.pred-d.actual).abs()/d.actual*100)
out['lead_time_mape_short_by_day'] = lts.groupby('step').ape.mean().round(2).to_dict()
# store stability: std of fold-level WAPE per store
st = cvd[(cvd.target=='revenue')&(cvd.horizon_type=='short')&(cvd.model==best_s.model)].groupby(['store_location','fold']).apply(lambda g:(g.pred-g.actual).abs().sum()/g.actual.sum()*100).groupby('store_location').agg(['mean','std'])
out['store_stability_wape_mean_std'] = st.round(2).to_dict('index')
# peak KPIs for hourly GBM
g = mh[(mh.horizon_type=='7d')&(mh.model=='Gradient Boosting')].iloc[0]
out['peak_capture_%'] = g['Peak Capture Rate %']; out['peak_error_%'] = g['Peak Error Rate %']; out['top3_hit_%'] = g['Top-3 Hour Hit %']
gb = cvh[(cvh.model=='Gradient Boosting')&(cvh.horizon_type=='7d')]; out['hourly_80pct_coverage'] = ((gb.actual>=gb.lo)&(gb.actual<=gb.hi)).mean()*100
# daily interval coverage (calibration, in-sample of calibration folds)
f = fd[(fd.target=='revenue')]
cvm = cvd[(cvd.target=='revenue')&(cvd.horizon_type=='medium')].copy(); cvm['b']=np.select([cvm.step<=7,cvm.step<=14,cvm.step<=21],[1,2,3],4); cvm['rel']=cvm.actual/cvm.pred-1
q = cvm.groupby(['model','b']).rel.quantile([.1,.9]).unstack(); cvm=cvm.join(q,on=['model','b']); out['daily_interval_coverage_%'] = ((cvm.rel>=cvm[0.1])&(cvm.rel<=cvm[0.9])).mean()*100

# ---- July forecast summary + scenarios ----
bm = best_m.model
jul = fd[(fd.target=='revenue')&(fd.model==bm)].groupby('store_location').agg(rev=('yhat','sum'), lo=('lo','sum'), hi=('hi','sum'))
jun = daily[daily.date>='2025-06-01'].groupby('store_location').revenue.sum()
jul['june_actual'] = jun; jul['growth_%'] = (jul.rev/jul.june_actual-1)*100
cum = cvd[(cvd.target=='revenue')&(cvd.horizon_type=='medium')&(cvd.model==bm)].groupby(['store_location','fold']).apply(lambda g:(g.actual.sum()/g.pred.sum()-1))
jul['best_case'] = jul.rev*(1+cum.groupby('store_location').quantile(0.9)); jul['worst_case'] = jul.rev*(1+cum.groupby('store_location').quantile(0.1))
print(jul.round(0)); jul.round(1).to_csv('data/july_scenarios.csv'); out['july_total'] = jul.rev.sum(); out['june_total']=jun.sum()
# hourly peak profile
pk = fh[fh.target=='transactions'].groupby(['store_location','hour']).yhat.mean().reset_index()
top = pk.loc[pk.groupby('store_location').yhat.idxmax()]; out['forecast_peak_hour'] = top.set_index('store_location').to_dict('index')
json.dump(out, open('data/kpis.json','w'), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))

# ---- figures ----
# F1 monthly revenue by store
m = daily.assign(m=daily.date.dt.month).groupby(['m','store_location']).revenue.sum().unstack()
ax = (m/1000).plot(kind='bar', color=[COL[c] for c in m.columns], figsize=(6.5,3.2), width=.8); ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun'],rotation=0); ax.set_ylabel('Revenue ($K)'); ax.set_xlabel(''); ax.set_title('Monthly revenue by store (2025)'); ax.legend(frameon=False,fontsize=8)
plt.tight_layout(); plt.savefig('figs/f1_monthly.png'); plt.close()
# F2 hourly profile
hp = tx.groupby(['store_location','hour']).size().unstack(0)/181
fig, ax = plt.subplots(figsize=(6.5,3.2))
for c in hp: ax.plot(hp.index, hp[c], marker='o', ms=3, color=COL[c], label=c)
ax.set_xlabel('Hour of day'); ax.set_ylabel('Avg transactions per hour'); ax.set_title('Average hourly transactions by store'); ax.legend(frameon=False); plt.tight_layout(); plt.savefig('figs/f2_hourly.png'); plt.close()
# F3 heatmap hour x dow (all stores)
hh = pd.read_csv('data/hourly_panel.csv', parse_dates=['date']); hh['dow']=hh.date.dt.dayofweek
hm = hh[hh.hour.between(6,20)].groupby(['dow','hour']).transactions.mean().unstack()
fig, ax = plt.subplots(figsize=(6.5,3)); im = ax.imshow(hm.values, aspect='auto', cmap='YlOrBr'); ax.set_yticks(range(7)); ax.set_yticklabels(['Mon','Tue','Wed','Thu','Fri','Sat','Sun']); ax.set_xticks(range(15)); ax.set_xticklabels(range(6,21))
ax.set_title('Avg transactions per store-hour: day of week × hour'); plt.colorbar(im, ax=ax, shrink=.8); plt.tight_layout(); plt.savefig('figs/f3_heatmap.png'); plt.close()
# F4 category mix
cm = tx.groupby('product_category').revenue.sum().sort_values()/1000
fig, ax = plt.subplots(figsize=(6.5,3)); ax.barh(cm.index, cm.values, color='#6F4E37'); ax.set_xlabel('Revenue ($K, Jan–Jun 2025)'); ax.set_title('Revenue by product category'); plt.tight_layout(); plt.savefig('figs/f4_category.png'); plt.close()
# F5 model comparison (short & medium MAPE, revenue)
mm = md[md.target=='revenue'].pivot(index='model', columns='horizon_type', values='MAPE').sort_values('short')
ax = mm[['short','medium']].plot(kind='barh', color=['#2E6F95','#C8553D'], figsize=(6.5,3.4)); ax.set_xlabel('MAPE (%) — daily revenue, rolling-origin CV'); ax.set_ylabel(''); ax.set_title('Model comparison'); ax.legend(['1–7 day','30 day'],frameon=False); plt.tight_layout(); plt.savefig('figs/f5_models.png'); plt.close()
# F6 forecast with intervals per store (daily revenue)
fig, axs = plt.subplots(1,3,figsize=(9.5,3),sharey=True)
for ax,(c,g) in zip(axs, daily.groupby('store_location')):
    ax.plot(g.date[-60:], g.revenue[-60:], color='#444', lw=1, label='Actual')
    f = fd[(fd.target=='revenue')&(fd.model==bm)&(fd.store_location==c)]
    ax.plot(f.date, f.yhat, color=COL[c], lw=1.6, label=f'{bm} forecast'); ax.fill_between(f.date, f.lo, f.hi, color=COL[c], alpha=.2, label='80% interval'); ax.set_title(c, fontsize=9); ax.tick_params(axis='x', rotation=45)
axs[0].set_ylabel('Daily revenue ($)'); axs[0].legend(frameon=False, fontsize=7); plt.tight_layout(); plt.savefig('figs/f6_forecast.png'); plt.close()
# F7 forecast heatmap hourly (all stores avg), Jul 1-7
fp = fh[fh.target=='transactions']; fp = fp.assign(dow=fp.date.dt.day_name().str[:3]).groupby(['store_location','hour']).yhat.mean().unstack(0)
fig, ax = plt.subplots(figsize=(6.5,3.2))
for c in fp: ax.plot(fp.index, fp[c], marker='o', ms=3, color=COL[c], label=c)
ax.set_xlabel('Hour'); ax.set_ylabel('Forecast transactions/hour'); ax.set_title('Forecast hourly demand, 1–7 Jul 2025 (daily avg profile)'); ax.legend(frameon=False); plt.tight_layout(); plt.savefig('figs/f7_hourly_fc.png'); plt.close()
print('figs done')
