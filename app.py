"""Afficionado Coffee Roasters — Demand Forecasting Dashboard (Streamlit).
Run:  pip install -r requirements.txt && streamlit run app.py
"""
import json, numpy as np, pandas as pd, plotly.graph_objects as go, streamlit as st
from pathlib import Path

D = Path(__file__).parent / 'data'
st.set_page_config(page_title='Afficionado Demand Forecasting', page_icon='☕', layout='wide')
COL = {'Astoria': '#2E6F95', "Hell's Kitchen": '#C8553D', 'Lower Manhattan': '#588157', 'All stores': '#6F4E37'}

@st.cache_data
def load():
    r = lambda f, **k: pd.read_csv(D / f, **k)
    return dict(daily=r('daily_panel.csv', parse_dates=['date']), hourly=r('hourly_panel.csv', parse_dates=['date']),
                fd=r('forecast_daily.csv', parse_dates=['date']), fh=r('forecast_hourly.csv', parse_dates=['date']),
                fc=r('forecast_category.csv', parse_dates=['date']), cat=r('category_daily.csv', parse_dates=['date']),
                md=r('metrics_daily.csv'), mds=r('metrics_daily_by_store.csv'), mh=r('metrics_hourly.csv'),
                mhs=r('metrics_hourly_by_store.csv'), kpi=json.load(open(D / 'kpis.json')))
X = load(); fd, fh, kpi = X['fd'], X['fh'], X['kpi']
STORES = ['All stores'] + sorted(fd.store_location.unique())

# ------------------------------ sidebar ------------------------------
st.sidebar.title('☕ Controls')
store = st.sidebar.selectbox('Store', STORES)
metric = st.sidebar.radio('Metric', ['Revenue ($)', 'Quantity (units)'], horizontal=True)
tgt = 'revenue' if metric.startswith('Rev') else 'qty'; unit = '$' if tgt == 'revenue' else ''
horizon = st.sidebar.slider('Forecast horizon (days)', 1, 30, 14)
models = sorted(fd.model.unique())
choice = st.sidebar.selectbox('Model', ['Auto (best by validation)'] + models)
auto = 'Gradient Boosting' if horizon <= 7 else 'Prophet'
model = auto if choice.startswith('Auto') else choice
show_ci = st.sidebar.checkbox('Show 80% prediction interval', True)
st.sidebar.caption(f'Active model: **{model}**' + (' (auto: best CV score for this horizon)' if choice.startswith('Auto') else ''))

def sel(df):  return df if store == 'All stores' else df[df.store_location == store]
def series(df, col):  # aggregate across stores if needed
    return df.groupby('date')[col].sum()

st.title('Afficionado Coffee Roasters — Demand Forecasting')
st.caption('Jan–Jun 2025 transactions · 3 stores · forecasts for 1–30 July 2025 · rolling-origin validated')
t1, t2, t3, t4, t5, t6 = st.tabs(['📈 Sales forecast', '🔥 Hourly demand', '🏆 Model comparison', '🎯 Scenarios & risk', '🥐 Category demand', 'ℹ️ Data notes'])

# ------------------------------ tab 1 ------------------------------
with t1:
    hist = series(sel(X['daily']), tgt); f = sel(fd[(fd.target == tgt) & (fd.model == model) & (fd.step <= horizon)])
    fs = f.groupby('date')[['yhat', 'lo', 'hi']].sum()
    fig = go.Figure(); h60 = hist.iloc[-60:]
    fig.add_scatter(x=h60.index, y=h60.values, name='Actual', line=dict(color='#444', width=1.5))
    if show_ci:
        fig.add_scatter(x=fs.index, y=fs.hi, line=dict(width=0), showlegend=False, hoverinfo='skip')
        fig.add_scatter(x=fs.index, y=fs.lo, fill='tonexty', fillcolor='rgba(200,85,61,.18)', line=dict(width=0), name='80% interval')
    fig.add_scatter(x=fs.index, y=fs.yhat, name=f'{model} forecast', line=dict(color=COL[store], width=2.5))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), yaxis_title=metric, title=f'Daily {metric.split(" ")[0].lower()} — {store}', hovermode='x unified', legend=dict(orientation='h'))
    st.plotly_chart(fig, width='stretch')
    last7 = hist.iloc[-7:].mean(); c = st.columns(4)
    c[0].metric(f'Forecast total ({horizon}d)', f'{unit}{fs.yhat.sum():,.0f}')
    c[1].metric('Avg per day', f'{unit}{fs.yhat.mean():,.0f}', f'{(fs.yhat.mean()/last7-1)*100:+.1f}% vs last 7d')
    c[2].metric('Interval (sum of daily 10th–90th)', f'{unit}{fs.lo.sum():,.0f} – {unit}{fs.hi.sum():,.0f}')
    hz = 'short' if horizon <= 7 else 'medium'
    m = X['md'][(X['md'].target == tgt) & (X['md'].horizon_type == hz) & (X['md'].model == model)].iloc[0]
    c[3].metric(f'Validated MAPE ({"1–7d" if hz=="short" else "30d"})', f'{m.MAPE:.1f}%', help='Rolling-origin cross-validation, all stores pooled')
    st.caption('Interval bounds are summed day by day, so the total range is conservative (daily errors partly offset). See *Scenarios & risk* for a calibrated 30-day total range.')
    with st.expander('Forecast table'):
        t = f.pivot_table(index='date', columns='store_location', values='yhat').round(0); t['All stores'] = t.sum(axis=1); st.dataframe(t, width='stretch')

# ------------------------------ tab 2 ------------------------------
with t2:
    hm_t = st.radio('Heatmap measure', ['Transactions', 'Quantity', 'Revenue'], horizontal=True, index=0)
    tt = {'Transactions': 'transactions', 'Quantity': 'qty', 'Revenue': 'revenue'}[hm_t]
    g = sel(fh[fh.target == tt]); z = g.pivot_table(index='date', columns='hour', values='yhat', aggfunc='sum')
    z.index = [d.strftime('%a %d %b') for d in z.index]
    fig = go.Figure(go.Heatmap(z=z.values, x=[f'{h:02d}:00' for h in z.columns], y=z.index, colorscale='YlOrBr', colorbar=dict(title=hm_t)))
    fig.update_yaxes(autorange='reversed'); fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10), title=f'Forecast hourly demand, 1–7 Jul 2025 — {store}')
    st.plotly_chart(fig, width='stretch')
    base = sel(fh[fh.target == 'transactions']).groupby(['date', 'hour']).agg(yhat=('yhat', 'sum'), lo=('lo', 'sum'), hi=('hi', 'sum')).reset_index()
    pk = base.loc[base.groupby('date').yhat.idxmax()]; pk['peak'] = pk.hour.map(lambda h: f'{h:02d}:00')
    st.subheader('Peak hour by day (transactions)'); st.dataframe(pk.assign(day=pk.date.dt.strftime('%a %d %b'))[['day', 'peak', 'yhat', 'lo', 'hi']].rename(columns={'yhat': 'forecast', 'lo': 'P10', 'hi': 'P90'}).round(0).set_index('day'), width='stretch')
    d0 = st.selectbox('Hourly interval for day', sorted(base.date.dt.strftime('%a %d %b').unique()))
    b = base[base.date.dt.strftime('%a %d %b') == d0]
    fig = go.Figure(); fig.add_scatter(x=b.hour, y=b.hi, line=dict(width=0), showlegend=False)
    fig.add_scatter(x=b.hour, y=b.lo, fill='tonexty', fillcolor='rgba(46,111,149,.2)', line=dict(width=0), name='80% interval'); fig.add_scatter(x=b.hour, y=b.yhat, name='Forecast', line=dict(color='#2E6F95', width=2.5))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), xaxis_title='Hour', yaxis_title='Transactions / hour'); st.plotly_chart(fig, width='stretch')
    st.info('Astoria shows no 07:00–10:59 sales on days 1–6 of every month in the historical data (and weaker mornings at the other stores on those days). The model reproduces this recurring pattern for 1–6 July. See *Data notes*.')

# ------------------------------ tab 3 ------------------------------
with t3:
    L, R = st.columns(2)
    with L:
        st.subheader('Daily forecasts (rolling-origin CV)')
        h = st.radio('Horizon', ['short', 'medium'], format_func=lambda x: '1–7 days' if x == 'short' else '30 days', horizontal=True)
        md = X['md'][(X['md'].target == tgt) & (X['md'].horizon_type == h)].set_index('model')[['MAE', 'RMSE', 'MAPE', 'WAPE', 'Bias%']].sort_values('MAPE')
        st.dataframe(md.round(2).style.highlight_min(subset=['MAE', 'RMSE', 'MAPE', 'WAPE'], color='#cfe8cf'), width='stretch')
        fig = go.Figure(go.Bar(x=md.MAPE, y=md.index, orientation='h', marker_color='#6F4E37')); fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), xaxis_title='MAPE %'); fig.update_yaxes(autorange='reversed'); st.plotly_chart(fig, width='stretch')
        s = X['mds'][(X['mds'].target == tgt) & (X['mds'].horizon_type == h)].pivot(index='model', columns='store_location', values='MAPE').round(1); st.caption('MAPE % by store'); st.dataframe(s, width='stretch')
    with R:
        st.subheader('Hourly transaction volume')
        hh = st.radio('Hourly horizon', ['7d', '1d'], format_func=lambda x: '7 days ahead' if x == '7d' else 'Next day', horizontal=True)
        mh = X['mh'][X['mh'].horizon_type == hh].set_index('model').drop(columns='horizon_type').sort_values('MAE')
        st.dataframe(mh.round(1).style.highlight_min(subset=['MAE', 'RMSE', 'MAPE', 'WAPE', 'Peak Error Rate %'], color='#cfe8cf').highlight_max(subset=['Peak Capture Rate %', 'Peak Precision %', 'Top-3 Hour Hit %'], color='#cfe8cf'), width='stretch')
        st.caption('Peak hour = hourly volume ≥ the store’s 75th percentile. Peak Error Rate = share of true peak hours the model failed to flag. MAPE uses non-zero hours only.')

# ------------------------------ tab 4 ------------------------------
with t4:
    st.subheader('Best-case / worst-case demand and sensitivity to spikes')
    sc = pd.read_csv(D / 'july_scenarios.csv').set_index('store_location')
    shock = st.slider('Demand shock vs. forecast (%)', -30, 30, 0, 5)
    sc2 = sc[['june_actual', 'worst_case', 'rev', 'best_case']].copy(); sc2.columns = ['June actual', 'Worst case (P10)', 'Base forecast', 'Best case (P90)']
    sc2.loc['All stores'] = sc2.sum(); sc2['With shock applied'] = sc2['Base forecast'] * (1 + shock/100)
    st.dataframe(sc2.round(0).style.format('${:,.0f}'), width='stretch')
    st.caption('July revenue (30 days). Worst/best case = 10th/90th percentile of historical 30-day total forecast error (Prophet, 18 store-folds) — indicative given the short history.')
    st.subheader('Staffing / capacity stress test (next 7 days)')
    cap = st.number_input('Max transactions per hour a store can handle comfortably', 20, 120, 45, 5)
    tx = fh[fh.target == 'transactions'].copy(); tx['shocked'] = tx.yhat * (1 + shock/100); tx['shocked_hi'] = tx.hi * (1 + shock/100)
    r = tx.groupby('store_location').apply(lambda d: pd.Series({'Peak-hour forecast': d.shocked.max(), 'Hours > capacity (forecast)': int((d.shocked > cap).sum()), 'Hours > capacity (P90 case)': int((d.shocked_hi > cap).sum()), 'Open store-hours': int((d.yhat > 1).sum())}))
    st.dataframe(r.round(1), width='stretch')
    st.caption(f'KPIs: hourly 80% interval coverage {kpi["hourly_80pct_coverage"]:.0f}% · daily interval coverage {kpi["daily_interval_coverage_%"]:.0f}% (both calibrated to ≈80% on validation data).')

# ------------------------------ tab 5 ------------------------------
with t5:
    cats = sorted(X['fc'].product_category.unique() if 'product_category' in X['fc'] else X['fc'].category.unique())
    cs = st.multiselect('Categories', cats, default=cats[:3]); fc = sel(X['fc']); ch = sel(X['cat'])
    fig = go.Figure()
    for i, c in enumerate(cs):
        hh_ = ch[ch.product_category == c].groupby('date').qty.sum().iloc[-45:]; ff = fc[(fc.category == c) & (fc.step <= horizon)].groupby('date').yhat.sum()
        col = ['#6F4E37', '#2E6F95', '#588157', '#C8553D', '#E0A458'][i % 5]
        fig.add_scatter(x=hh_.index, y=hh_.values, name=f'{c} actual', line=dict(color=col, width=1), opacity=.6); fig.add_scatter(x=ff.index, y=ff.values, name=f'{c} forecast', line=dict(color=col, width=2.5, dash='dash'))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), yaxis_title='Units per day', title=f'Daily category demand — {store}'); st.plotly_chart(fig, width='stretch')
    tot = fc[fc.step <= horizon].groupby(['category', 'model']).yhat.sum().round(0).reset_index().rename(columns={'yhat': f'Units next {horizon}d', 'model': 'Model used'}); st.dataframe(tot, width='stretch', hide_index=True)
    st.caption('Category model chosen per category by lowest 7-day WAPE in rolling validation.')

# ------------------------------ tab 6 ------------------------------
with t6:
    st.markdown(f"""
**Headline validation KPIs** (rolling-origin, no shuffling)
- Daily revenue forecast accuracy: **{kpi['forecast_accuracy_short_%']:.0f}%** (1–7 d, {kpi['best_short_model']}) · **{kpi['forecast_accuracy_medium_%']:.0f}%** (30 d, {kpi['best_medium_model']})
- Peak demand capture rate (hourly, 7-day ahead): **{kpi['peak_capture_%']:.0f}%** · top-3 hour hit rate **{kpi['top3_hit_%']:.0f}%**
- 30-day cumulative revenue error: **{kpi['rev_err_30d_abs%']:.1f}%** average absolute

**Important data notes**
1. The source file has **no date column**. Calendar dates were reconstructed from row order (time of day resets 180 times → 181 days) and anchored to 1 Jan 2025. The reconstruction is corroborated by recurring patterns aligning exactly with month starts and by the total revenue ($698,812).
2. **Days 1–6 of every month behave differently**: Astoria has zero sales 07:00–10:59, and the other stores’ mornings are roughly halved. Daily revenue is 4–19 % lower on those days. Cause unknown (operational vs. data-capture) — please verify with the business.
3. Only six months of history: no yearly seasonality can be learned, and the strong Jan→Jun growth trend may not persist. Re-train as new data arrives.
4. Hours outside 06:00–20:59 have no sales and are excluded from hourly modelling.
""")
