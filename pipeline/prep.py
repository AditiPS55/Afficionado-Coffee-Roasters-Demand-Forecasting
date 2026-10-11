"""Step 1: data preparation. Reconstructs dates, builds continuous hourly/daily panels."""
import pandas as pd, numpy as np

SRC = '/mnt/user-data/uploads/Afficionado_Coffee_Roasters.xlsx'
START = pd.Timestamp('2025-01-01')  # ASSUMPTION: file has no date column; rows are chronological, 181 day-resets

df = pd.read_excel(SRC)
sec = pd.to_timedelta(df.transaction_time.astype(str)).dt.total_seconds()
df['day_idx'] = (sec.diff() < 0).cumsum()           # time goes backwards => new day
df['date'] = START + pd.to_timedelta(df.day_idx, unit='D')
df['hour'] = (sec // 3600).astype(int)
df['revenue'] = df.transaction_qty * df.unit_price
df['ts'] = df.date + pd.to_timedelta(df.hour, unit='h')
df.drop(columns=['year']).to_csv('data/transactions_dated.csv', index=False)

stores = df[['store_id','store_location']].drop_duplicates().sort_values('store_id')
full_h = pd.date_range(df.date.min(), df.date.max() + pd.Timedelta(hours=23), freq='h')
rows = []
for sid, loc in stores.itertuples(index=False):
    s = df[df.store_id == sid]
    g = s.groupby('ts').agg(transactions=('transaction_id','count'), qty=('transaction_qty','sum'), revenue=('revenue','sum'))
    g = g.reindex(full_h, fill_value=0)       # missing intervals = zero sales (store closed / no sales)
    g.index.name = 'ts'; g = g.reset_index()
    g['store_id'] = sid; g['store_location'] = loc
    rows.append(g)
hourly = pd.concat(rows, ignore_index=True)
hourly['date'] = hourly.ts.dt.normalize(); hourly['hour'] = hourly.ts.dt.hour
hourly.to_csv('data/hourly_panel.csv', index=False)

daily = hourly.groupby(['store_id','store_location','date'], as_index=False)[['transactions','qty','revenue']].sum()
daily.to_csv('data/daily_panel.csv', index=False)

cat = df.groupby(['store_id','store_location','date','product_category'], as_index=False).agg(qty=('transaction_qty','sum'), revenue=('revenue','sum'))
# complete grid with zeros
idx = pd.MultiIndex.from_product([stores.store_id, pd.date_range(df.date.min(), df.date.max()), sorted(df.product_category.unique())], names=['store_id','date','product_category'])
cat = cat.set_index(['store_id','date','product_category']).reindex(idx).drop(columns='store_location').fillna(0).reset_index()
cat = cat.merge(stores, on='store_id')
cat.to_csv('data/category_daily.csv', index=False)

print('dates', df.date.min().date(), '->', df.date.max().date(), '| days', df.date.nunique())
print('hourly panel', hourly.shape, '| daily', daily.shape, '| category', cat.shape)
print('total revenue', round(df.revenue.sum(),2), '| zero-sale hours (all 24h):', (hourly.transactions==0).mean().round(3))
print(daily.groupby('store_location')[['transactions','revenue']].mean().round(1))
d = daily.assign(dow=daily.date.dt.day_name())
print(d.groupby('dow').revenue.mean().round(0).reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']))
m = daily.assign(m=daily.date.dt.month).groupby('m').revenue.sum().round(0); print(m)
print('price outliers (unit_price>=20):', (df.unit_price>=20).sum(), '| qty>=5:', (df.transaction_qty>=5).sum())
