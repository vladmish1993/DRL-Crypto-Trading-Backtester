"""
Technical indicators for trading feature engineering.
All indicators are computed without lookahead bias.
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

    # ── ATR (Average True Range) ─────────────────────────────────
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()

    # ── Volume features ──────────────────────────────────────────
    df['volume_sma'] = volume.rolling(20).mean()
    df['volume_ratio'] = volume / df['volume_sma'].replace(0, np.nan)

    # ── Returns ──────────────────────────────────────────────────
    df['returns'] = close.pct_change()
    df['log_returns'] = np.log(close / close.shift(1))

    # Drop rows where indicators are still warming up
    df = df.dropna().reset_index(drop=True)
    return df


def normalize_features(df: pd.DataFrame, feature_cols: list,
                       window: int = 100) -> pd.DataFrame:
    """
    Rolling z-score normalisation (avoids lookahead bias).
    Price-based columns get rolling normalisation;
    bounded columns (rsi, volume_ratio, bb_width) get simpler scaling.
    """
    df = df.copy()

    rolling_norm_cols = [
        'open', 'high', 'low', 'close',
        'sma_20', 'sma_50', 'macd', 'macd_signal', 'macd_hist', 'atr',
    ]

    for col in rolling_norm_cols:
        if col in df.columns:
            rm = df[col].rolling(window, min_periods=1).mean()
            rs = df[col].rolling(window, min_periods=1).std().replace(0, 1)
            df[f'{col}_norm'] = (df[col] - rm) / rs

    # Bounded features
    if 'rsi' in df.columns:
        df['rsi_norm'] = df['rsi'] / 100.0
    if 'volume_ratio' in df.columns:
        df['volume_ratio_norm'] = df['volume_ratio'].clip(0, 5) / 5.0
    if 'bb_width' in df.columns:
        df['bb_width_norm'] = df['bb_width'].clip(0, 0.3) / 0.3

    df = df.iloc[window:].reset_index(drop=True)
    return df
