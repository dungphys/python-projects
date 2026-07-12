"""
event_study_utils.py
---------------------
Reusable engine for the Russia-sanctions event study.

Methodology
-----------
1. Individual sanctions actions that occur within CLUSTER_GAP trading days of
   each other are merged into a single "event cluster" (e.g. the Feb-Mar 2022
   wave of invasion + SWIFT expulsion + CBR asset freeze). This avoids treating
   overlapping shocks as independent events.
2. For every cluster we define:
     - an estimation window  [t0 + EST_START, t0 + EST_END]   (normal-return model)
     - an event window       [t0 + EVT_PRE,   t1 + EVT_POST]  (abnormal-return window)
3. A cluster is flagged CONTAMINATED if any other cluster's anchor date falls
   inside its combined estimation+event span. Additionally, the estimation
   window itself is "cleaned" by dropping any trading day that falls inside
   ANOTHER cluster's event window, so the normal-return model is never fit on
   days that are themselves a different shock.
4. Abnormal/cumulative abnormal returns are computed on price/level SERIES
   (log MOEX index, log USD/RUB, level Urals-Brent spread) rather than by
   summing daily returns. This makes CAR robust to missing/halted trading
   days (MOEX was halted 2022-02-28 to 2022-03-23) - the endpoint simply
   rolls forward to the first price available once trading resumes.
5. Standardized CAR (SCAR = CAR / (sigma_est * sqrt(L))) is used for per-event
   t-tests, and the Boehmer-Musumeci-Poulsen (1991) cross-sectional test is
   used to aggregate across clusters (robust to event-induced variance
   increases, unlike a naive pooled t-test).
"""
import numpy as np
import pandas as pd
from scipy import stats

EST_START, EST_END = -120, -21          # estimation window offset (trading days rel. to cluster anchor)
EVT_PRE, EVT_POST = -2, 10            # event window offset (trading days rel. to cluster anchor)
CLUSTER_GAP = EVT_POST - EVT_PRE + 1       # trading days: gap <= this => same cluster

def load_market_data(commodities_csv, equity_csv):
    # load equity and commodities data, merge on date/year
    eq = pd.read_csv(equity_csv, parse_dates=['date']) 
    co = pd.read_csv(commodities_csv, parse_dates=['date'])
    mkt = eq.merge(co, on=['date', 'year'], how='inner').sort_values('date').reset_index(drop=True)
    # add interger trading-day counter and compute log returns and level changes
    mkt['t'] = np.arange(len(mkt))
    mkt['ln_moex'] = np.log(mkt['moex_index'])
    mkt['ln_rub'] = np.log(mkt['usd_rub'])
    mkt['moex_ret'] = mkt['ln_moex'].diff()
    mkt['rub_ret'] = mkt['ln_rub'].diff()
    mkt['urals_spread_chg'] = mkt['urals_brent_spread_usd'].diff()
    return mkt

# lookup dictionaries so the rest of the code can loop over "all three assets" generically
# asset key -> (level column used for the CAR endpoint calc, return column used for mu/sigma)
ASSET_COLS = {
    'moex_ret': ('ln_moex', 'moex_ret'),
    'rub_ret': ('ln_rub', 'rub_ret'),
    'urals_spread_chg': ('urals_brent_spread_usd', 'urals_spread_chg'),
}
ASSET_LABELS = {
    'moex_ret': 'MOEX (equity, log return)',
    'rub_ret': 'USD/RUB (FX, log return; `+` = RUB depreciation)', # note: USD/RUB is quoted as RUB per USD, so a positive log return means the RUB has depreciated
    'urals_spread_chg': 'Urals-Brent discount (level change, USD/bbl)',
}


def cluster_events(sanctions_csv, mkt):
    # load sanctions events
    ev = pd.read_csv(sanctions_csv, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    mkt_dates = mkt['date'].values

    # maps a sanctions-event date to the nearest available trading day in the market data
    def nearest_trading_t(d):
        idx = np.searchsorted(mkt_dates, np.datetime64(d), side='left')
        idx = min(idx, len(mkt_dates) - 1)
        return mkt.loc[idx, 't']
    ev['t'] = ev['date'].apply(nearest_trading_t)
    ev = ev.sort_values('t').reset_index(drop=True)
    
    # run-length clustering: cluster events that are close together
    cluster_id = [0]
    for i in range(1, len(ev)):
        gap = ev.loc[i, 't'] - ev.loc[i - 1, 't']
        cluster_id.append(cluster_id[-1] + (1 if gap > CLUSTER_GAP else 0))
    ev['cluster_id'] = cluster_id

    # aggregate per-cluster metadata
    clusters = ev.groupby('cluster_id').agg(
        t0=('t', 'min'), t1=('t', 'max'),
        date0=('date', 'min'), date1=('date', 'max'),
        n_events=('event_id', 'count'), severity_sum=('severity_score', 'sum'),
        severity_max=('severity_score', 'max'),
        waves=('wave', lambda x: '|'.join(sorted(set(x)))),
        jurisdictions=('jurisdiction', lambda x: '|'.join(sorted(set(x)))),
        is_energy_any=('is_energy', 'max'), is_financial_any=('is_financial', 'max'),
        event_ids=('event_id', lambda x: ','.join(x)),
    ).reset_index()
    return ev, clusters


def build_window_metadata(clusters):
    """Estimation/event windows + contamination flag + cleaned estimation days per cluster."""
    max_t = clusters['t0'].max() + 10000  # placeholder, replaced by caller with mkt max
    evt_hi_of = lambda r: r['t1'] + EVT_POST
    rows = []
    for _, row in clusters.iterrows(): # run for each cluster, compute its estimation/event windows and contamination status
        t0, t1 = row['t0'], row['t1']
        est_lo, est_hi = t0 + EST_START, t0 + EST_END # estimation window (trading days relative to cluster anchor)
        evt_lo, evt_hi = t0 + EVT_PRE, evt_hi_of(row)  # event window (trading days relative to cluster anchor)

        # contamination check: is there any other cluster's anchor date that falls inside this cluster's estimation+event span?
        # i.e., another sanctions shock happened close enough to interfere with either the baseline estimate or the event measurement
        other_t0 = clusters.loc[clusters['cluster_id'] != row['cluster_id'], 't0']
        contaminated = bool(other_t0.between(est_lo, evt_hi).any())
        
        # clean estimation days: drop any trading day that falls inside another cluster's event window, 
        # so the normal-return model is never fit on days that are themselves a different shock
        est_days = np.arange(est_lo, est_hi + 1) # number of trading days in the estimation window
        other_evt = [(r['t0'] + EVT_PRE, evt_hi_of(r))
                     for _, r in clusters.iterrows() if r['cluster_id'] != row['cluster_id']]
        clean_days = np.array([d for d in est_days if not any(lo <= d <= hi for lo, hi in other_evt)])

        rows.append(dict(
            cluster_id=row['cluster_id'], t0=t0, t1=t1, date0=row['date0'], date1=row['date1'],
            n_events=row['n_events'], severity_sum=row['severity_sum'], severity_max=row['severity_max'],
            waves=row['waves'], jurisdictions=row['jurisdictions'],
            is_energy_any=row['is_energy_any'], is_financial_any=row['is_financial_any'],
            est_lo=est_lo, est_hi=est_hi, evt_lo=evt_lo, evt_hi=evt_hi,
            contaminated=contaminated, clean_days=clean_days,
        ))
    return pd.DataFrame(rows)

# helper function to find the last available value of a column before a given date
def _last_avail(mkt, col, t):
    s = mkt.loc[mkt['t'] <= t, col].dropna()
    return s.iloc[-1] if len(s) else np.nan

# helper function to find the first available value of a column on or after a given date
def _first_avail_on_or_after(mkt, col, t):
    s = mkt.loc[mkt['t'] >= t, col].dropna()
    return s.iloc[0] if len(s) else np.nan

# compute CAR and SCAR for a given cluster and asset
def compute_car_for_cluster(mkt, mrow, asset_key, min_est_obs=20):
    lvl_col, ret_col = ASSET_COLS[asset_key]  # level column used for the CAR endpoint calc, return column used for mu/sigma
    est_vals = mkt.loc[mkt['t'].isin(mrow['clean_days']), ret_col].dropna() # estimation-period returns for this asset
    # if there are too few estimation-period observations, return None
    L1 = len(est_vals)
    if len(est_vals) < min_est_obs:
        print(f"Skipping cluster {mrow['cluster_id']}, asset {asset_key}: only {len(est_vals)} estimation-period observations (min required = {min_est_obs})")
        return None
    # Computes mu (expected daily change) and sigma (its volatility) from the cleaned estimation period
    # if sigma is zero or NaN, return None
    mu, sigma = est_vals.mean(), est_vals.std(ddof=1) 
    if sigma == 0 or np.isnan(sigma):
        print(f"Skipping cluster {mrow['cluster_id']}, asset {asset_key}: sigma = {sigma} (estimation-period volatility)")
        return None
    # compute length of the event window in trading days 
    L2 = mrow['evt_hi'] - mrow['evt_lo'] + 1

    p_start = _last_avail(mkt, lvl_col, mrow['evt_lo'] - 1) # price/level at the last available trading day **BEFORE** the event window
    # compute price/level at the first available trading day on or after the event window end
    exact_end = mkt.loc[mkt['t'] == mrow['evt_hi'], lvl_col] 
    if len(exact_end) and not exact_end.isna().all(): # if the exact end date is available, use it
        p_end = exact_end.iloc[0]
    else: # otherwise, roll forward to the first available trading day
        p_end = _first_avail_on_or_after(mkt, lvl_col, mrow['evt_hi'])
    if pd.isna(p_start) or pd.isna(p_end):
        return None
    # compute raw cumulative change, CAR, and SCAR
    raw_cum = p_end - p_start # raw cumulative change over the event window
    car = raw_cum - mu * L2 # car = actual cumulative change minus expected cumulative change over L days
    scar = car / (sigma * np.sqrt(L2 * (1 + L2 / L1))) # scar = car standardized by the estimation-period volatility and the square root of the event window length
    # check if there are any missing returns in the event window (i.e., a gap in trading)
    had_gap = bool(mkt.loc[(mkt['t'] >= mrow['evt_lo']) & (mkt['t'] <= mrow['evt_hi']), ret_col].isna().any())
    return dict(mu=mu, sigma=sigma, car=car, L1=L1,
                L2=L2, scar=scar, had_gap=had_gap)

# build a table of CAR and SCAR for each cluster and asset
def build_car_table(mkt, meta):
    """One row per (cluster, asset) with CAR and standardized CAR (SCAR)."""
    rows = []
    for _, mrow in meta.iterrows():
        for key in ASSET_COLS:
            r = compute_car_for_cluster(mkt, mrow, key)
            if r is None:
                continue
            rows.append(dict(
                cluster_id=mrow['cluster_id'], date0=mrow['date0'], date1=mrow['date1'],
                asset=key, asset_label=ASSET_LABELS[key],
                n_events=mrow['n_events'], severity_sum=mrow['severity_sum'],
                waves=mrow['waves'], jurisdictions=mrow['jurisdictions'],
                is_energy_any=mrow['is_energy_any'], is_financial_any=mrow['is_financial_any'],
                contaminated=mrow['contaminated'], had_gap=r['had_gap'],
                car=r['car'], scar=r['scar'], sigma_est=r['sigma'], L1=r['L1'], L2=r['L2'],
            ))
    return pd.DataFrame(rows)


def bmp_test(scar_series):
    """Boehmer-Musumeci-Poulsen (1991) cross-sectional standardized test.
    Input: array of per-event standardized CARs (SCAR_i). Robust to
    event-induced variance changes because each event is pre-standardized
    by its own estimation-period sigma before cross-sectional averaging."""
    # convert to numpy array, drop NaNs
    x = np.asarray(scar_series, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    # if there are fewer than 2 events, return NaN for mean, t-statistic, and p-value
    if n < 2:
        return dict(n=n, mean_scar=np.nan, bmp_t=np.nan, p_value=np.nan)
    # compute mean, std, t-statistic, and p-value
    mean_scar = x.mean()
    sd = x.std(ddof=1)
    bmp_t = np.sqrt(n) * mean_scar / sd if sd > 0 else np.nan # t-statistic for the null hypothesis that the mean SCAR is zero
    # p-value for a two-sided t-test with n-1 degrees of freedom
    p_value = 2 * (1 - stats.t.cdf(abs(bmp_t), df=n - 1)) if not np.isnan(bmp_t) else np.nan
    return dict(n=n, mean_scar=mean_scar, bmp_t=bmp_t, p_value=p_value)
