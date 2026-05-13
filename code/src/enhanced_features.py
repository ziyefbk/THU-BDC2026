import pandas as pd
import numpy as np


def engineer_cross_sectional_features(df):
    """
    Compute cross-sectional rank features and market-wide features per date.
    These are computed AFTER stock-level feature engineering (e.g. after engineer_features_158plus39).
    This function takes the processed DataFrame with features already computed per stock.
    """
    df = df.copy()
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in
                    ['股票代码', '日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额',
                     '振幅', '涨跌额', '换手率', '涨跌幅', 'open_t1', 'open_t5', 'label', 'instrument']]

    print(f"  Computing cross-sectional ranks for {len(feature_cols)} features...")

    rank_features = {}
    for col in feature_cols:
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            ranked = df.groupby('日期')[col].rank(pct=True, na_option='keep')
            rank_features[f'rank_{col}'] = ranked.values

    rank_df = pd.DataFrame(rank_features, index=df.index)
    df = pd.concat([df, rank_df], axis=1)

    print(f"  Added {len(rank_features)} cross-sectional rank features")

    return df


def compute_market_features(df):
    """
    Compute market-wide features per date: equal-weighted market return, market volume, breadth.
    """
    df = df.copy()
    df['日期'] = pd.to_datetime(df['日期'])

    open_ = df['开盘'].astype(float)
    close = df['收盘'].astype(float)
    volume = df['成交量'].astype(float)

    daily_ret = close.groupby(df['日期']).mean().pct_change()

    df['market_return'] = df['日期'].map(daily_ret)
    df['market_return'] = df.groupby('日期')['market_return'].transform(
        lambda x: x.fillna(x.mean())
    )

    market_vol = volume.groupby(df['日期']).mean()
    df['market_avg_volume'] = df['日期'].map(market_vol)
    df['volume_to_market'] = volume / (df['market_avg_volume'] + 1e-12)

    rising = (close > open_).groupby(df['日期']).sum()
    total = close.groupby(df['日期']).count()
    df['market_breadth'] = df['日期'].map(rising / total)

    print("  Added market-wide features: market_return, volume_to_market, market_breadth")

    return df


def compute_risk_features(df):
    """
    Compute risk features per stock: beta proxy, idiosyncratic volatility, drawdown.
    These are time-series features computed per stock, not cross-sectional.
    """
    df = df.copy()
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    close = df['收盘'].astype(float)
    volume = df['成交量'].astype(float)

    df['daily_return'] = close.groupby(df['股票代码']).pct_change()

    market_ret = df.groupby('日期')['daily_return'].transform('mean')

    df['stock_market_cov'] = df.groupby('股票代码')['daily_return'].transform(
        lambda x: x.rolling(20, min_periods=10).cov(market_ret.loc[x.index])
    )
    df['market_var'] = market_ret.groupby(df['日期']).transform(
        lambda x: x.rolling(20, min_periods=10).var()
    )
    df['beta'] = df['stock_market_cov'] / (df['market_var'] + 1e-12)
    df['beta'] = df['beta'].fillna(0).clip(-10, 10)

    df['rolling_std_20'] = df.groupby('股票代码')['daily_return'].transform(
        lambda x: x.rolling(20, min_periods=5).std()
    )
    df['idiosyncratic_vol'] = df['rolling_std_20']

    cum_ret = df.groupby('股票代码')['daily_return'].transform(
        lambda x: (1 + x.fillna(0)).cumprod()
    )
    running_max = cum_ret.groupby(df['股票代码']).cummax()
    df['drawdown'] = (cum_ret - running_max) / (running_max + 1e-12)

    df['volume_std_20'] = df.groupby('股票代码')['成交量'].transform(
        lambda x: x.rolling(20, min_periods=5).std()
    )
    df['volume_to_avg'] = volume / (df['成交量'].groupby(df['股票代码']).transform('mean') + 1e-12)

    print("  Added risk features: beta, idiosyncratic_vol, drawdown, volume_to_avg")

    return df


def compute_momentum_features(df):
    """
    Compute momentum acceleration and divergence features per stock.
    """
    df = df.copy()
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    close = df['收盘'].astype(float)
    volume = df['成交量'].astype(float)

    ret_1d = close.groupby(df['股票代码']).pct_change(1)
    ret_5d = close.groupby(df['股票代码']).pct_change(5)
    ret_20d = close.groupby(df['股票代码']).pct_change(20)

    df['momentum_accel'] = ret_1d - ret_5d
    df['short_long_mom'] = ret_5d - ret_20d

    vol_ret_corr = df.groupby('股票代码').apply(
        lambda g: g['成交量'].astype(float).rolling(20, min_periods=10).corr(
            close.loc[g.index].astype(float)
        )
    ).reset_index(level=0, drop=True)
    df['vol_price_corr'] = vol_ret_corr

    vol_change = volume.groupby(df['股票代码']).pct_change(5)
    price_change = ret_5d
    df['vol_price_div'] = vol_change - price_change

    print("  Added momentum features: momentum_accel, short_long_mom, vol_price_corr, vol_price_div")

    return df


def engineer_enhanced_features(df):
    """
    Top-level wrapper: applies all enhanced feature engineering on top of
    the existing engineer_features_158plus39 output.
    Call this AFTER engineer_features_158plus39(df).
    """
    df = df.copy()

    print("Computing enhanced cross-sectional features...")
    df = engineer_cross_sectional_features(df)

    print("Computing market-wide features...")
    df = compute_market_features(df)

    print("Computing risk features...")
    df = compute_risk_features(df)

    print("Computing momentum features...")
    df = compute_momentum_features(df)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    print(f"Enhanced feature engineering complete. Total columns: {len(df.columns)}")
    return df
