"""
Technical indicators for trading feature engineering.
All indicators are computed without lookahead bias.

Notes
- Indicators are computed using only past/current information (no lookahead).
- We drop warm-up rows with NaNs at the end of add_indicators().
"""

import numpy as np
import pandas as pd


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to an OHLCV DataFrame."""
    df = df.copy()

    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    # ── Moving Averages ──────────────────────────────────────────
    df['sma_20'] = close.rolling(20).mean()
    df['sma_50'] = close.rolling(50).mean()
    df['ema_12'] = close.ewm(span=12, adjust=False).mean()
    df['ema_26'] = close.ewm(span=26, adjust=False).mean()

    # ── MACD ─────────────────────────────────────────────────────
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # ── RSI ──────────────────────────────────────────────────────
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100.0 - (100.0 / (1.0 + rs))

    # ── Bollinger Bands ──────────────────────────────────────────
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df['bb_upper'] = bb_mid + 2 * bb_std
    df['bb_lower'] = bb_mid - 2 * bb_std
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / bb_mid

    # ── True Range / ATR ─────────────────────────────────────────
    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Simple ATR (rolling mean). Kept for backwards-compat with existing results.
    df['atr'] = tr.rolling(14).mean()

    # ── ADX (Average Directional Index) ───────────────────────────
    # Classic Wilder's smoothing via EMA(alpha=1/n).
    n = 14
    up_move = high.diff()
    down_move = -low.diff()  # prev_low - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr_w = tr.ewm(alpha=1 / n, adjust=False).mean()
    plus_dm_w = pd.Series(plus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean()
    minus_dm_w = pd.Series(minus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean()

    tr_w_safe = tr_w.replace(0, np.nan)
    plus_di = 100.0 * (plus_dm_w / tr_w_safe)
    minus_di = 100.0 * (minus_dm_w / tr_w_safe)

    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    df['adx'] = dx.ewm(alpha=1 / n, adjust=False).mean()

    # ── Volume features ──────────────────────────────────────────
    df['volume_sma'] = volume.rolling(20).mean()
    df['volume_ratio'] = volume / df['volume_sma'].replace(0, np.nan)

    # ── Returns ──────────────────────────────────────────────────
    df['returns'] = close.pct_change()
    df['log_returns'] = np.log(close / close.shift(1))

    # Drop rows where indicators are still warming up
    df = df.dropna().reset_index(drop=True)
    return df


def normalize_features(df: pd.DataFrame, feature_cols: list, window: int = 100) -> pd.DataFrame:
    """
    Rolling z-score normalisation (avoids lookahead bias).

    Price-based / unbounded columns get rolling z-score normalisation.
    Bounded columns (e.g. RSI/ADX) get simple scaling to [0, 1] ranges.
    """
    df = df.copy()

    rolling_norm_cols = [
        'open', 'high', 'low', 'close',
        'sma_20', 'sma_50',
        'macd', 'macd_signal', 'macd_hist',
        'atr',
    ]

    for col in rolling_norm_cols:
        if col in df.columns:
            rm = df[col].rolling(window, min_periods=1).mean()
            rs = df[col].rolling(window, min_periods=1).std().replace(0, 1)
            df[f'{col}_norm'] = (df[col] - rm) / rs

    # Bounded / clipped features
    if 'rsi' in df.columns:
        df['rsi_norm'] = df['rsi'].clip(0, 100) / 100.0
    if 'adx' in df.columns:
        df['adx_norm'] = df['adx'].clip(0, 100) / 100.0
    if 'volume_ratio' in df.columns:
        df['volume_ratio_norm'] = df['volume_ratio'].clip(0, 5) / 5.0
    if 'bb_width' in df.columns:
        df['bb_width_norm'] = df['bb_width'].clip(0, 0.3) / 0.3

    # Drop the initial normalisation warm-up window (same behaviour as before).
    df = df.iloc[window:].reset_index(drop=True)
    return df
