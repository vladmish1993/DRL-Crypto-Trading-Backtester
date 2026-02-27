#!/usr/bin/env python3
"""plot_trades.py

Run a rule config and plot price with trade markers.

Key features:
- Accepts a sweep `config_key` (recommended) or manual params.
- Can plot markers either at the trade price ("price") or above the chart ("top").
- Figure width can scale with the number of candles (mm per candle), with a sensible cap.
Outputs a PNG (default: results/trade_plots/<config>_<split>.png)
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# allow imports from scripts/
sys.path.insert(0, os.path.dirname(__file__))

from indicators import add_indicators, normalize_features
from rule_backtest import EnvParams, FastBacktester, RsiTrendStrategy


# Must match your pipeline normalisation warm-up.
FEATURES = [
    'close_norm', 'open_norm', 'high_norm', 'low_norm',
    'sma_20_norm', 'sma_50_norm',
    'rsi_norm', 'macd_norm', 'macd_signal_norm', 'macd_hist_norm',
    'bb_width_norm', 'atr_norm', 'volume_ratio_norm', 'returns',
    'adx_norm',
]


def _tag_to_float(s: str) -> float:
    """Convert tag form like '30p0' or '0p005' to float."""
    s = s.strip()
    if 'p' in s:
        left, right = s.split('p', 1)
        if left in ('', '+'):
            left = '0'
        if left == '-':
            left = '-0'
        return float(f"{left}.{right}")
    return float(s)


def _float_to_tag(x: float, max_decimals: int = 6) -> str:
    """Convert float to tag form used in config keys.
    Examples: 30.0 -> '30p0', 0.005 -> '0p005', 0.1 -> '0p1', 0.02 -> '0p02'
    """
    x = float(x)
    # Avoid scientific notation
    s = f"{x:.{max_decimals}f}"
    # Trim trailing zeros but keep at least one decimal digit
    if '.' in s:
        s = s.rstrip('0').rstrip('.')
    if '.' not in s:
        s = s + '.0'
    left, right = s.split('.', 1)
    if right == '':
        right = '0'
    return f"{left}p{right}"


def parse_config_key(config_key: str) -> Dict:
    """Parse keys like:
    re30p0_rx80p0_lb5_sma200_sl0p0_tp0p005_mp0p1_mh32_cd4_sh1_adx28p0
    """
    out: Dict = {}
    for tok in str(config_key).strip().split('_'):
        if tok.startswith('re'):
            out['rsi_entry'] = _tag_to_float(tok[2:])
        elif tok.startswith('rx'):
            out['rsi_exit'] = _tag_to_float(tok[2:])
        elif tok.startswith('lb'):
            out['trend_lookback'] = int(tok[2:])
        elif tok.startswith('sma'):
            out['sma_period'] = int(tok[3:])
        elif tok.startswith('sl'):
            out['sl'] = _tag_to_float(tok[2:])
        elif tok.startswith('tp'):
            out['tp'] = _tag_to_float(tok[2:])
        elif tok.startswith('mp'):
            out['max_pos'] = _tag_to_float(tok[2:])
        elif tok.startswith('mh'):
            out['min_hold'] = int(tok[2:])
        elif tok.startswith('cd'):
            out['cooldown'] = int(tok[2:])
        elif tok.startswith('sh'):
            out['allow_short'] = bool(int(tok[2:]))
        elif tok.startswith('adx'):
            out['adx_threshold'] = _tag_to_float(tok[3:])

    out.setdefault('allow_short', False)
    out.setdefault('adx_threshold', 0.0)
    return out


def load_split(csv_path: str, split: str, train_ratio: float, val_ratio: float) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    df = add_indicators(df)
    df = normalize_features(df, FEATURES, window=100)

    if split == 'full':
        return df.reset_index(drop=True)

    split1 = int(len(df) * train_ratio)
    split2 = int(len(df) * (train_ratio + val_ratio))

    train_df = df.iloc[:split1]
    val_df = df.iloc[split1:split2] if val_ratio > 0 else None
    test_df = df.iloc[split2:] if val_ratio > 0 else df.iloc[split1:]

    if split == 'train':
        return train_df.reset_index(drop=True)
    if split == 'val':
        if val_df is None:
            raise ValueError('val_ratio is 0, cannot use split=val')
        return val_df.reset_index(drop=True)
    if split == 'test':
        return test_df.reset_index(drop=True)

    raise ValueError(f"Unknown split: {split}")


class RsiTrendStrategyADX(RsiTrendStrategy):
    """Wrap the existing RSI strategy but block entries when ADX is above a threshold."""

    def __init__(self, df: pd.DataFrame, params, adx_threshold: float = 0.0):
        super().__init__(df, params)
        self.adx_threshold = float(adx_threshold or 0.0)
        self._adxs = df['adx'].values.astype(np.float64) if 'adx' in df.columns else None

    def decide(self, bt: FastBacktester) -> int:
        action = super().decide(bt)

        # Entry gating only (never blocks exits)
        if bt.position == 0 and action in (bt.LONG, bt.SHORT) and self._adxs is not None and self.adx_threshold > 0:
            a = self._adxs[bt.step_idx]
            if (not np.isfinite(a)) or (a > self.adx_threshold):
                return bt.HOLD

        return action


def run_and_collect(df_eval: pd.DataFrame, strat_params: Dict, env_params: Dict) -> Tuple[Dict, list, list]:
    from rule_backtest import StrategyParams

    sp = StrategyParams(
        rsi_entry=float(strat_params.get('rsi_entry', 30.0)),
        rsi_exit=float(strat_params.get('rsi_exit', 70.0)),
        trend_lookback=int(strat_params.get('trend_lookback', 5)),
        slope_threshold=float(strat_params.get('slope_threshold', 0.0)),
        sma_period=int(strat_params.get('sma_period', 50)),
        allow_short=bool(strat_params.get('allow_short', False)),
    )

    ep = EnvParams(
        initial_balance=float(env_params.get('initial_balance', 10_000.0)),
        leverage=int(env_params.get('leverage', 1)),
        fee_rate=float(env_params.get('fee', 0.0004)),
        max_position_frac=float(env_params.get('max_pos', 0.10)),
        stop_loss_pct=float(env_params.get('sl', 0.0)),
        take_profit_pct=float(env_params.get('tp', 0.0)),
        min_hold_steps=int(env_params.get('min_hold', 16)),
        cooldown_steps=int(env_params.get('cooldown', 4)),
    )

    bt = FastBacktester(df_eval, ep, capture_equity=True, capture_trades=True)
    st = RsiTrendStrategyADX(bt.df, sp, adx_threshold=float(strat_params.get('adx_threshold', 0.0)))
    st.reset()

    done = False
    while not done:
        action = st.decide(bt)
        done = bt.step(action)

    metrics = bt.get_metrics()
    return metrics, bt.trades, bt.equity_curve


def _figsize_from_candles(n_candles: int, mm_per_candle: float, min_width_in: float, max_width_in: float, height_in: float) -> Tuple[float, float]:
    """Compute figure size. We cap width so huge datasets don't create enormous images."""
    mm_per_candle = float(mm_per_candle)
    width_in = (n_candles * mm_per_candle) / 25.4  # mm -> inches
    width_in = max(min_width_in, min(max_width_in, width_in))
    return float(width_in), float(height_in)


def plot_trades(
    df_eval: pd.DataFrame,
    trades: list,
    out_png: str,
    title: str,
    mode: str = 'simple',
    markers: str = 'top',
    marker_size: int = 70,
    marker_zorder: int = 20,
    line_zorder: int = 1,
    marker_edge: str = 'white',
    marker_lw: float = 1.2,
    long_open_color: str = 'lime',
    long_close_color: str = 'green',
    short_open_color: str = 'red',
    short_close_color: str = 'darkred',
    buy_color: str = 'lime',
    sell_color: str = 'red',
    figsize: Tuple[float, float] = (12, 7),
    legend_outside: bool = True,
):
    if 'timestamp' in df_eval.columns:
        x = df_eval['timestamp']
    else:
        x = np.arange(len(df_eval))

    close = df_eval['close'].values.astype(float)

    # buckets
    buy_x, buy_y = [], []
    sell_x, sell_y = [], []

    long_open_x, long_open_y = [], []
    long_close_x, long_close_y = [], []
    short_open_x, short_open_y = [], []
    short_close_x, short_close_y = [], []

    pos = 0  # 0 flat, 1 long, -1 short

    for t in trades:
        step = int(t.get('step', -1))
        if step < 0 or step >= len(df_eval):
            continue

        ts = x.iloc[step] if hasattr(x, 'iloc') else x[step]
        price = float(t.get('price', close[step]))
        action = str(t.get('action', '')).upper()

        if action == 'LONG':
            if mode == 'detailed':
                long_open_x.append(ts); long_open_y.append(price)
            buy_x.append(ts); buy_y.append(price)
            pos = 1

        elif action == 'SHORT':
            if mode == 'detailed':
                short_open_x.append(ts); short_open_y.append(price)
            sell_x.append(ts); sell_y.append(price)
            pos = -1

        elif action in ('CLOSE', 'SL', 'TP'):
            if pos == 1:
                if mode == 'detailed':
                    long_close_x.append(ts); long_close_y.append(price)
                sell_x.append(ts); sell_y.append(price)
            elif pos == -1:
                if mode == 'detailed':
                    short_close_x.append(ts); short_close_y.append(price)
                buy_x.append(ts); buy_y.append(price)
            pos = 0

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, close, label='Close Price')

    # If requested, draw markers above the chart at fixed y-levels.
    if markers == 'top':
        y_min = float(np.nanmin(close))
        y_max = float(np.nanmax(close))
        y_rng = max(1e-9, y_max - y_min)
        base = y_max + 0.02 * y_rng
        step = 0.03 * y_rng

        def _const_y(n: int, level: int) -> list:
            return [base + level * step] * n

        if mode == 'detailed':
            if long_open_x:
                ax.scatter(long_open_x, _const_y(len(long_open_x), 0), marker='^', s=marker_size, label='Long Open', color=long_open_color, clip_on=False, zorder=marker_zorder, linewidths=marker_lw)
            if long_close_x:
                ax.scatter(long_close_x, _const_y(len(long_close_x), 1), marker='v', s=marker_size, label='Long Close', color=long_close_color, clip_on=False, zorder=marker_zorder, linewidths=marker_lw)
            if short_open_x:
                ax.scatter(short_open_x, _const_y(len(short_open_x), 2), marker='v', s=marker_size, label='Short Open', color=short_open_color, clip_on=False, zorder=marker_zorder, linewidths=marker_lw)
            if short_close_x:
                ax.scatter(short_close_x, _const_y(len(short_close_x), 3), marker='^', s=marker_size, label='Short Close', color=short_close_color, clip_on=False, zorder=marker_zorder, linewidths=marker_lw)
        else:
            if buy_x:
                ax.scatter(buy_x, _const_y(len(buy_x), 0), marker='^', s=marker_size, label='Buy Signal', color=buy_color, clip_on=False, zorder=marker_zorder, linewidths=marker_lw)
            if sell_x:
                ax.scatter(sell_x, _const_y(len(sell_x), 1), marker='v', s=marker_size, label='Sell Signal', color=sell_color, clip_on=False, zorder=marker_zorder, linewidths=marker_lw)

        # Extend ylim so the above-markers are visible.
        ax.set_ylim(y_min, base + 4.5 * step)

    else:
        # Plot markers at the trade price levels.
        if mode == 'detailed':
            if long_open_x:
                ax.scatter(long_open_x, long_open_y, marker='^', s=marker_size, label='Long Open', color=long_open_color, zorder=marker_zorder, linewidths=marker_lw)
            if long_close_x:
                ax.scatter(long_close_x, long_close_y, marker='v', s=marker_size, label='Long Close', color=long_close_color, zorder=marker_zorder, linewidths=marker_lw)
            if short_open_x:
                ax.scatter(short_open_x, short_open_y, marker='v', s=marker_size, label='Short Open', color=short_open_color, zorder=marker_zorder, linewidths=marker_lw)
            if short_close_x:
                ax.scatter(short_close_x, short_close_y, marker='^', s=marker_size, label='Short Close', color=short_close_color, zorder=marker_zorder, linewidths=marker_lw)
        else:
            if buy_x:
                ax.scatter(buy_x, buy_y, marker='^', s=marker_size, label='Buy Signal', color=buy_color, zorder=marker_zorder, linewidths=marker_lw)
            if sell_x:
                ax.scatter(sell_x, sell_y, marker='v', s=marker_size, label='Sell Signal', color=sell_color, zorder=marker_zorder, linewidths=marker_lw)

    ax.set_title(title, y=1.02 if markers == 'top' else None)
    ax.set_xlabel('Time')
    ax.set_ylabel('Close Price')
    ax.grid(True, alpha=0.2)

    if legend_outside:
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
        fig.tight_layout(rect=[0, 0, 0.84, 1])
    else:
        ax.legend()
        fig.tight_layout()

    fig.autofmt_xdate()

    os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved plot -> {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join('data', 'SOL_USDT_15m.csv'))
    ap.add_argument('--split', choices=['train', 'val', 'test', 'full'], default='val')
    ap.add_argument('--train_ratio', type=float, default=0.6)
    ap.add_argument('--val_ratio', type=float, default=0.2)

    ap.add_argument('--config_key', type=str, default=None, help='Config key from sweeps, e.g. re30p0_rx80p0_..._adx28p0')

    # Manual params (used if no --config_key)
    ap.add_argument('--rsi_entry', type=float, default=30.0)
    ap.add_argument('--rsi_exit', type=float, default=70.0)
    ap.add_argument('--trend_lookback', type=int, default=5)
    ap.add_argument('--slope_threshold', type=float, default=0.0)
    ap.add_argument('--sma_period', type=int, default=50)
    ap.add_argument('--allow_short', action='store_true')
    ap.add_argument('--adx_threshold', type=float, default=0.0)

    ap.add_argument('--fee', type=float, default=0.0004)
    ap.add_argument('--max_pos', type=float, default=0.10)
    ap.add_argument('--sl', type=float, default=0.0)
    ap.add_argument('--tp', type=float, default=0.0)
    ap.add_argument('--min_hold', type=int, default=16)
    ap.add_argument('--cooldown', type=int, default=4)
    ap.add_argument('--initial_balance', type=float, default=10_000.0)
    ap.add_argument('--leverage', type=int, default=1)

    # Plot controls
    ap.add_argument('--mode', choices=['simple', 'detailed'], default='simple')
    ap.add_argument('--markers', choices=['top', 'price'], default='price', help='Place markers above the chart (top) or at trade price (price).')
    ap.add_argument('--marker_size', type=int, default=70)
    ap.add_argument('--marker_zorder', type=int, default=20, help='Render markers above the price line (higher = on top).')
    ap.add_argument('--line_zorder', type=int, default=1, help='Z-order for the price line (lower = behind markers).')
    ap.add_argument('--marker_edge', type=str, default='white', help='Marker outline colour for visibility.')
    ap.add_argument('--marker_lw', type=float, default=1.2, help='Marker outline width.')

    # Marker colours (matplotlib colour names or hex, e.g. 'lime' or '#00ff00')
    ap.add_argument('--long_open_color', type=str, default='lime', help='Colour for Long Open markers.')
    ap.add_argument('--long_close_color', type=str, default='green', help='Colour for Long Close markers.')
    ap.add_argument('--short_open_color', type=str, default='red', help='Colour for Short Open markers.')
    ap.add_argument('--short_close_color', type=str, default='darkred', help='Colour for Short Close markers.')
    ap.add_argument('--buy_color', type=str, default='lime', help='Colour for Buy markers (simple mode).')
    ap.add_argument('--sell_color', type=str, default='red', help='Colour for Sell markers (simple mode).')
    ap.add_argument('--mm_per_candle', type=float, default=1.0, help='Figure width scaling: millimetres per candle.')
    ap.add_argument('--min_width_in', type=float, default=12.0)
    ap.add_argument('--max_width_in', type=float, default=30.0)
    ap.add_argument('--height_in', type=float, default=7.0)
    ap.add_argument('--legend_outside', action='store_true', help='Put legend outside the plot (right side).')
    ap.add_argument('--legend_inside', action='store_true', help='Force legend inside the plot.')

    ap.add_argument('--out_png', type=str, default=None)
    ap.add_argument('--title', type=str, default=None)

    args = ap.parse_args()

    if args.config_key:
        p = parse_config_key(args.config_key)
        strat_params = {
            'rsi_entry': p.get('rsi_entry', args.rsi_entry),
            'rsi_exit': p.get('rsi_exit', args.rsi_exit),
            'trend_lookback': p.get('trend_lookback', args.trend_lookback),
            'slope_threshold': args.slope_threshold,
            'sma_period': p.get('sma_period', args.sma_period),
            'allow_short': p.get('allow_short', args.allow_short),
            'adx_threshold': p.get('adx_threshold', args.adx_threshold),
        }
        env_params = {
            'fee': args.fee,
            'max_pos': p.get('max_pos', args.max_pos),
            'sl': p.get('sl', args.sl),
            'tp': p.get('tp', args.tp),
            'min_hold': p.get('min_hold', args.min_hold),
            'cooldown': p.get('cooldown', args.cooldown),
            'initial_balance': args.initial_balance,
            'leverage': args.leverage,
        }
        config_name = args.config_key
    else:
        strat_params = {
            'rsi_entry': args.rsi_entry,
            'rsi_exit': args.rsi_exit,
            'trend_lookback': args.trend_lookback,
            'slope_threshold': args.slope_threshold,
            'sma_period': args.sma_period,
            'allow_short': bool(args.allow_short),
            'adx_threshold': float(args.adx_threshold),
        }
        env_params = {
            'fee': args.fee,
            'max_pos': args.max_pos,
            'sl': args.sl,
            'tp': args.tp,
            'min_hold': args.min_hold,
            'cooldown': args.cooldown,
            'initial_balance': args.initial_balance,
            'leverage': args.leverage,
        }
        config_name = (f"re{_float_to_tag(args.rsi_entry)}_rx{_float_to_tag(args.rsi_exit)}_"f"lb{args.trend_lookback}_sma{args.sma_period}_"f"sl{_float_to_tag(args.sl)}_tp{_float_to_tag(args.tp)}_mp{_float_to_tag(args.max_pos)}_"f"mh{args.min_hold}_cd{args.cooldown}_sh{1 if args.allow_short else 0}_"f"adx{_float_to_tag(args.adx_threshold)}")

    df_eval = load_split(args.data, args.split, args.train_ratio, args.val_ratio)
    metrics, trades, _eq = run_and_collect(df_eval, strat_params, env_params)

    if args.out_png is None:
        out_png = os.path.join('results', 'trade_plots', f"{config_name}_st{_float_to_tag(args.slope_threshold)}_fee{_float_to_tag(args.fee)}_tr{_float_to_tag(args.train_ratio)}_vr{_float_to_tag(args.val_ratio)}_split{args.split}.png")
    else:
        out_png = args.out_png

    if args.title is None:
        title = (
            f"Trade signals ({args.split}) | Sharpe {metrics['sharpe_ratio']:+.2f} | "
            f"Ret {metrics['total_return']:+.2f}% | DD {metrics['max_drawdown']:.2f}% | "
            f"Trades {metrics['total_trades']}"
        )
    else:
        title = args.title

    fig_w, fig_h = _figsize_from_candles(
        n_candles=len(df_eval),
        mm_per_candle=args.mm_per_candle,
        min_width_in=args.min_width_in,
        max_width_in=args.max_width_in,
        height_in=args.height_in,
    )

    legend_outside = True
    if args.legend_inside:
        legend_outside = False
    if args.legend_outside:
        legend_outside = True

    plot_trades(
        df_eval,
        trades,
        out_png=out_png,
        title=title,
        mode=args.mode,
        markers=args.markers,
        marker_size=args.marker_size,
        marker_zorder=args.marker_zorder,
        line_zorder=args.line_zorder,
        marker_edge=args.marker_edge,
        marker_lw=args.marker_lw,
        long_open_color=args.long_open_color,
        long_close_color=args.long_close_color,
        short_open_color=args.short_open_color,
        short_close_color=args.short_close_color,
        buy_color=args.buy_color,
        sell_color=args.sell_color,
        figsize=(fig_w, fig_h),
        legend_outside=legend_outside,
    )


if __name__ == '__main__':
    main()