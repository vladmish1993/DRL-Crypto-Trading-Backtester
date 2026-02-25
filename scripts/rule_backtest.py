#!/usr/bin/env python3
"""
Rule-based backtest: RSI trend + optional SMA filter + SL/TP + fees.

Design goals
- Deterministic, no ML.
- Same data pipeline as train_all_window.py: add_indicators() + normalize_features(window=100)
- Execution model mirrors CryptoFuturesEnv mechanics (long/short/flat, fees, SL/TP intrabar, min-hold, cooldown).

Usage
  python scripts/rule_backtest.py --split val --rsi_entry 30 --rsi_exit 70 --sma_period 50 --sl 0.02 --tp 0.04
  python scripts/rule_backtest.py --split test --allow_short --sma_period 200 --trend_lookback 7

Output
- JSON with metrics for "RSI Rule" plus "Buy & Hold" baseline.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# allow imports from scripts/
sys.path.insert(0, os.path.dirname(__file__))

from indicators import add_indicators, normalize_features


# Must match train_all_window.py feature pipeline (normalisation warm-up window is 100)
FEATURES = [
    'close_norm', 'open_norm', 'high_norm', 'low_norm',
    'sma_20_norm', 'sma_50_norm',
    'rsi_norm', 'macd_norm', 'macd_signal_norm', 'macd_hist_norm',
    'bb_width_norm', 'atr_norm', 'volume_ratio_norm', 'returns',
]

# Annualisation factor for 15-minute bars: 4 * 24 * 365
ANN_FACTOR = math.sqrt(4 * 24 * 365)


def load_data(csv_path: str, train_ratio: float = 0.6, val_ratio: float = 0.2):
    """
    Same split logic as train_all_window.py:
      df -> add_indicators -> normalize_features(window=100) -> train/val/test by ratio.
    """
    df = pd.read_csv(csv_path)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    df = add_indicators(df)
    df = normalize_features(df, FEATURES, window=100)

    split1 = int(len(df) * train_ratio)
    split2 = int(len(df) * (train_ratio + val_ratio))

    train = df.iloc[:split1].reset_index(drop=True)
    val = df.iloc[split1:split2].reset_index(drop=True) if val_ratio > 0 else None
    test = df.iloc[split2:].reset_index(drop=True) if val_ratio > 0 else df.iloc[split1:].reset_index(drop=True)
    return train, val, test


def _ensure_sma(df: pd.DataFrame, period: int) -> str:
    """
    Ensure df has a column 'sma_{period}' computed from close with a simple rolling mean.
    Returns the column name. Does not drop rows; early values will be NaN.
    """
    col = f'sma_{int(period)}'
    if period <= 0:
        return col
    if col not in df.columns:
        df[col] = df['close'].rolling(int(period), min_periods=int(period)).mean()
    return col


@dataclass(frozen=True)
class StrategyParams:
    rsi_entry: float = 30.0
    rsi_exit: float = 70.0
    trend_lookback: int = 5
    slope_threshold: float = 0.0
    sma_period: int = 50  # 0 disables
    allow_short: bool = False


@dataclass(frozen=True)
class EnvParams:
    initial_balance: float = 10_000.0
    leverage: int = 1
    fee_rate: float = 0.0004
    max_position_frac: float = 0.2
    stop_loss_pct: float = 0.0
    take_profit_pct: float = 0.0
    min_hold_steps: int = 16
    cooldown_steps: int = 4


class FastBacktester:
    """
    Minimal backtest engine mirroring CryptoFuturesEnv step mechanics and metrics.
    """

    HOLD, LONG, SHORT, CLOSE = 0, 1, 2, 3

    def __init__(self, df: pd.DataFrame, env: EnvParams, capture_equity: bool = True, capture_trades: bool = True):
        self.df = df.reset_index(drop=True)
        self.env = env
        self.capture_equity = bool(capture_equity)
        self.capture_trades = bool(capture_trades)

        # Pre-extract numpy arrays for fast access (avoids pandas iloc in hot loop)
        self._closes = self.df['close'].values.astype(np.float64)
        self._highs = self.df['high'].values.astype(np.float64)
        self._lows = self.df['low'].values.astype(np.float64)
        self._n_bars = len(self._closes)

        self.step_idx: int = 0
        self.balance: float = env.initial_balance
        self.position: int = 0
        self.position_size: float = 0.0
        self.entry_price: float = 0.0
        self.entry_step: int = 0
        self.cooldown: int = 0

        self.equity_curve: List[float] = [env.initial_balance] if self.capture_equity else []
        self.trades: List[dict] = [] if self.capture_trades else []

        self.sl_hits: int = 0
        self.tp_hits: int = 0

        # online stats for Sharpe when capture_equity=False
        self._ret_n = 0
        self._ret_mean = 0.0
        self._ret_M2 = 0.0
        self._peak_eq = env.initial_balance
        self._max_dd = 0.0
        self._prev_eq = env.initial_balance

        # closed-trade stats
        self._closed_n = 0
        self._wins = 0
        self._pnl_sum = 0.0

    # ---------------------------- helpers
    def _price(self) -> float:
        return self._closes[self.step_idx]

    def _equity(self) -> float:
        p = self._price()
        upnl = self.position * self.position_size * (p - self.entry_price) if self.position else 0.0
        return self.balance + upnl

    def _record_equity(self, eq: float):
        if self.capture_equity:
            self.equity_curve.append(eq)

        # online Sharpe + DD updates
        r = (eq - self._prev_eq) / self._prev_eq if self._prev_eq > 0 else 0.0
        self._ret_n += 1
        delta = r - self._ret_mean
        self._ret_mean += delta / self._ret_n
        delta2 = r - self._ret_mean
        self._ret_M2 += delta * delta2

        self._peak_eq = max(self._peak_eq, eq)
        dd = (self._peak_eq - eq) / self._peak_eq if self._peak_eq > 0 else 0.0
        self._max_dd = max(self._max_dd, dd)

        self._prev_eq = eq

    def _open(self, direction: int):
        price = self._price()
        notional = self.balance * self.env.max_position_frac * self.env.leverage
        fee = notional * self.env.fee_rate
        self.position = direction
        self.position_size = notional / price if price > 0 else 0.0
        self.entry_price = price
        self.entry_step = self.step_idx
        self.balance -= fee

        if self.capture_trades:
            self.trades.append(dict(
                step=self.step_idx,
                timestamp=str(self.df.iloc[self.step_idx].get('timestamp', '')),
                action='LONG' if direction == 1 else 'SHORT',
                price=round(price, 6),
                size=round(self.position_size, 8),
                fee=round(fee, 4),
            ))

    def _close_at(self, fill_price: float, reason: str = 'CLOSE'):
        if not self.position:
            return 0.0
        pnl = self.position * self.position_size * (fill_price - self.entry_price)
        fee = self.position_size * fill_price * self.env.fee_rate
        self.balance += pnl - fee

        # closed-trade stats
        self._closed_n += 1
        self._pnl_sum += pnl
        if pnl > 0:
            self._wins += 1

        if reason == 'SL':
            self.sl_hits += 1
        elif reason == 'TP':
            self.tp_hits += 1

        if self.capture_trades:
            self.trades.append(dict(
                step=self.step_idx,
                timestamp=str(self.df.iloc[self.step_idx].get('timestamp', '')),
                action=reason,
                price=round(float(fill_price), 6),
                size=round(self.position_size, 8),
                pnl=round(float(pnl), 4),
                fee=round(float(fee), 4),
            ))

        self.position = 0
        self.position_size = 0.0
        self.entry_price = 0.0
        return pnl

    def _close(self, reason: str = 'CLOSE'):
        return self._close_at(self._price(), reason=reason)

    def _check_sl_tp(self) -> bool:
        if not self.position or (not self.env.stop_loss_pct and not self.env.take_profit_pct):
            return False

        high = self._highs[self.step_idx]
        low = self._lows[self.step_idx]

        if self.position == 1:
            if self.env.stop_loss_pct and low <= self.entry_price * (1 - self.env.stop_loss_pct):
                fill = self.entry_price * (1 - self.env.stop_loss_pct)
                self._close_at(fill, reason='SL')
                return True
            if self.env.take_profit_pct and high >= self.entry_price * (1 + self.env.take_profit_pct):
                fill = self.entry_price * (1 + self.env.take_profit_pct)
                self._close_at(fill, reason='TP')
                return True

        if self.position == -1:
            if self.env.stop_loss_pct and high >= self.entry_price * (1 + self.env.stop_loss_pct):
                fill = self.entry_price * (1 + self.env.stop_loss_pct)
                self._close_at(fill, reason='SL')
                return True
            if self.env.take_profit_pct and low <= self.entry_price * (1 - self.env.take_profit_pct):
                fill = self.entry_price * (1 - self.env.take_profit_pct)
                self._close_at(fill, reason='TP')
                return True

        return False

    # ---------------------------- step loop
    def step(self, action: int) -> bool:
        """
        Executes one bar. Returns done bool.
        """
        if self.step_idx >= self._n_bars - 1:
            return True

        # SL/TP fires before discretionary action (matches CryptoFuturesEnv)
        sl_tp_closed = self._check_sl_tp()
        if sl_tp_closed and self.env.cooldown_steps:
            self.cooldown = max(self.cooldown, self.env.cooldown_steps)

        trades_this_step = 0

        # cooldown tick
        if self.cooldown > 0:
            self.cooldown -= 1

        hold_steps = (self.step_idx - self.entry_step) if self.position else 0
        can_close = (not self.position) or (hold_steps >= self.env.min_hold_steps)
        can_trade = (self.cooldown == 0) and (not sl_tp_closed)

        # execute with anti-churn gates (matches CryptoFuturesEnv)
        if action == self.LONG and self.position <= 0 and can_trade:
            if self.position == -1:
                if can_close:
                    self._close(); trades_this_step += 1
                else:
                    action = self.HOLD
            if self.position == 0:
                self._open(+1); trades_this_step += 1
                self.cooldown = self.env.cooldown_steps

        elif action == self.SHORT and self.position >= 0 and can_trade:
            if self.position == +1:
                if can_close:
                    self._close(); trades_this_step += 1
                else:
                    action = self.HOLD
            if self.position == 0:
                self._open(-1); trades_this_step += 1
                self.cooldown = self.env.cooldown_steps

        elif action == self.CLOSE and self.position != 0:
            if can_close and can_trade:
                self._close(); trades_this_step += 1
                self.cooldown = self.env.cooldown_steps
            else:
                action = self.HOLD

        self.step_idx += 1

        done = self.step_idx >= self._n_bars - 1
        if done and self.position:
            self._close()
            trades_this_step += 1

        eq = self._equity()
        self._record_equity(eq)
        return done

    # ---------------------------- metrics
    def get_metrics(self) -> dict:
        final_eq = self._prev_eq
        total_return = (final_eq / self.env.initial_balance - 1) * 100.0

        if self._ret_n > 0:
            var = self._ret_M2 / self._ret_n
            std = math.sqrt(var)
            sharpe = (self._ret_mean / std * ANN_FACTOR) if std > 0 else 0.0
        else:
            sharpe = 0.0

        max_dd = self._max_dd * 100.0
        win_rate = (self._wins / self._closed_n * 100.0) if self._closed_n else 0.0
        avg_pnl = (self._pnl_sum / self._closed_n) if self._closed_n else 0.0

        return dict(
            total_return=round(total_return, 2),
            sharpe_ratio=round(sharpe, 2),
            max_drawdown=round(max_dd, 2),
            win_rate=round(win_rate, 2),
            total_trades=int(self._closed_n),
            avg_trade_pnl=round(float(avg_pnl), 4),
            final_balance=round(float(final_eq), 2),
            sl_hits=int(self.sl_hits),
            tp_hits=int(self.tp_hits),
        )


class RsiTrendStrategy:
    """
    RSI reversal-ish entry:
      - Long when RSI <= entry and RSI slope over N bars is positive.
      - Optional trend filter: close > SMA(period) for long.
      - Exit long when RSI >= exit OR RSI slope flips negative.

    Optional shorts:
      - Short when RSI >= (100-entry) and RSI slope over N bars is negative.
      - Optional trend filter: close < SMA(period) for short.
      - Exit short when RSI <= (100-exit) OR slope flips positive.
    """

    def __init__(self, df: pd.DataFrame, params: StrategyParams):
        self.df = df
        self.p = params
        self.rsi_hist: Deque[float] = deque(maxlen=max(2, int(params.trend_lookback)))
        self.sma_col = _ensure_sma(self.df, int(params.sma_period))

        # Pre-extract numpy arrays for fast access
        self._rsis = df['rsi'].values.astype(np.float64) if 'rsi' in df.columns else None
        self._closes = df['close'].values.astype(np.float64)
        if int(params.sma_period) > 0 and self.sma_col in df.columns:
            self._smas = df[self.sma_col].values.astype(np.float64)
        else:
            self._smas = None

        n = max(2, int(params.trend_lookback))
        x = np.arange(n, dtype=float)
        self._n = n
        self._sum_x = float(x.sum())
        self._sum_x2 = float((x * x).sum())
        self._den = (n * self._sum_x2 - self._sum_x * self._sum_x) or 1.0

    def reset(self):
        self.rsi_hist.clear()

    def _slope(self) -> Optional[float]:
        if len(self.rsi_hist) < self._n:
            return None
        y = np.asarray(self.rsi_hist, dtype=float)
        x = np.arange(self._n, dtype=float)
        sum_y = float(y.sum())
        sum_xy = float((x * y).sum())
        slope = (self._n * sum_xy - self._sum_x * sum_y) / self._den
        return float(slope)

    def decide(self, bt: FastBacktester) -> int:
        idx = bt.step_idx

        if self._rsis is None:
            return bt.HOLD
        rsi = self._rsis[idx]
        close = self._closes[idx]

        if not np.isfinite(rsi) or not np.isfinite(close):
            return bt.HOLD

        self.rsi_hist.append(rsi)
        slope = self._slope()
        if slope is None:
            return bt.HOLD

        slope_thr = float(self.p.slope_threshold)

        # trend filter
        trend_ok_long = True
        trend_ok_short = True
        if self._smas is not None:
            sma = self._smas[idx]
            if not np.isfinite(sma):
                trend_ok_long = False
                trend_ok_short = False
            else:
                trend_ok_long = close > sma
                trend_ok_short = close < sma

        # exit signals
        if bt.position == 1:
            if (rsi >= self.p.rsi_exit) or (slope < -slope_thr):
                return bt.CLOSE
            return bt.HOLD

        if bt.position == -1:
            short_exit_rsi = 100.0 - float(self.p.rsi_exit)
            if (rsi <= short_exit_rsi) or (slope > slope_thr):
                return bt.CLOSE
            return bt.HOLD

        # entry signals (flat)
        if bt.position == 0:
            if (rsi <= self.p.rsi_entry) and (slope > slope_thr) and trend_ok_long:
                return bt.LONG

            if self.p.allow_short:
                short_entry_rsi = 100.0 - float(self.p.rsi_entry)
                if (rsi >= short_entry_rsi) and (slope < -slope_thr) and trend_ok_short:
                    return bt.SHORT

        return bt.HOLD


def buy_and_hold_baseline(df_eval: pd.DataFrame, initial: float = 10_000.0) -> dict:
    prices = df_eval['close'].values.astype(float)
    equity = initial * prices / prices[0]
    ret = (equity[-1] / initial - 1) * 100
    rets = np.diff(equity) / equity[:-1]
    sharpe = (rets.mean() / rets.std() * ANN_FACTOR) if rets.std() > 0 else 0.0
    peak = np.maximum.accumulate(equity)
    mdd = ((peak - equity) / peak).max() * 100

    step = max(1, len(equity) // 2000)
    return dict(
        algorithm='Buy & Hold',
        total_return=round(float(ret), 2),
        sharpe_ratio=round(float(sharpe), 2),
        max_drawdown=round(float(mdd), 2),
        win_rate=0.0,
        total_trades=1,
        avg_trade_pnl=0.0,
        final_balance=round(float(equity[-1]), 2),
        equity_curve=[round(float(equity[i]), 2) for i in range(0, len(equity), step)],
        trades=[],
    )


def run_rule_backtest(df_eval: pd.DataFrame, strat: StrategyParams, env: EnvParams,
                      capture: bool = True) -> dict:
    bt = FastBacktester(df_eval, env, capture_equity=True, capture_trades=bool(capture))
    st = RsiTrendStrategy(bt.df, strat)
    st.reset()

    done = False
    while not done:
        action = st.decide(bt)
        done = bt.step(action)

    m = bt.get_metrics()
    m['algorithm'] = 'RSI Rule'
    m['params'] = {
        'rsi_entry': strat.rsi_entry,
        'rsi_exit': strat.rsi_exit,
        'trend_lookback': strat.trend_lookback,
        'slope_threshold': strat.slope_threshold,
        'sma_period': strat.sma_period,
        'allow_short': strat.allow_short,
        'fee_rate': env.fee_rate,
        'max_position_frac': env.max_position_frac,
        'stop_loss_pct': env.stop_loss_pct,
        'take_profit_pct': env.take_profit_pct,
        'min_hold_steps': env.min_hold_steps,
        'cooldown_steps': env.cooldown_steps,
        'leverage': env.leverage,
        'initial_balance': env.initial_balance,
    }

    if capture:
        eq = bt.equity_curve
        step = max(1, len(eq) // 2000)
        m['equity_curve'] = [round(float(eq[i]), 2) for i in range(0, len(eq), step)]
        m['trades'] = bt.trades[-50:]
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join('data', 'SOL_USDT_15m.csv'))
    ap.add_argument('--output', default=os.path.join('results', 'rule_backtest_results.json'))
    ap.add_argument('--split', choices=['val', 'test', 'both'], default='val')
    ap.add_argument('--train_ratio', type=float, default=0.6)
    ap.add_argument('--val_ratio', type=float, default=0.2)

    # Strategy params
    ap.add_argument('--rsi_entry', type=float, default=30.0)
    ap.add_argument('--rsi_exit', type=float, default=70.0)
    ap.add_argument('--trend_lookback', type=int, default=5)
    ap.add_argument('--slope_threshold', type=float, default=0.0)
    ap.add_argument('--sma_period', type=int, default=50, help='0 disables SMA filter')
    ap.add_argument('--allow_short', action='store_true')

    # Execution params
    ap.add_argument('--initial_balance', type=float, default=10_000.0)
    ap.add_argument('--leverage', type=int, default=1)
    ap.add_argument('--fee', type=float, default=0.0004)
    ap.add_argument('--max_pos', type=float, default=0.2)
    ap.add_argument('--sl', type=float, default=0.0)
    ap.add_argument('--tp', type=float, default=0.0)
    ap.add_argument('--min_hold', type=int, default=16)
    ap.add_argument('--cooldown', type=int, default=4)

    # Control output shape
    ap.add_argument('--no_public_copy', action='store_true')
    args = ap.parse_args()

    if args.trend_lookback < 2:
        raise ValueError("--trend_lookback must be >= 2")

    train_df, val_df, test_df = load_data(args.data, train_ratio=args.train_ratio, val_ratio=args.val_ratio)

    eval_sets: List[Tuple[str, pd.DataFrame]] = []
    if args.split == 'val':
        if val_df is None:
            raise ValueError("val_ratio is 0, cannot evaluate on validation set")
        eval_sets = [('val', val_df)]
    elif args.split == 'test':
        eval_sets = [('test', test_df)]
    else:
        if val_df is None:
            raise ValueError("val_ratio is 0, cannot evaluate on validation set")
        eval_sets = [('val', val_df), ('test', test_df)]

    strat = StrategyParams(
        rsi_entry=args.rsi_entry,
        rsi_exit=args.rsi_exit,
        trend_lookback=args.trend_lookback,
        slope_threshold=args.slope_threshold,
        sma_period=args.sma_period,
        allow_short=bool(args.allow_short),
    )

    env = EnvParams(
        initial_balance=args.initial_balance,
        leverage=args.leverage,
        fee_rate=args.fee,
        max_position_frac=args.max_pos,
        stop_loss_pct=args.sl,
        take_profit_pct=args.tp,
        min_hold_steps=args.min_hold,
        cooldown_steps=args.cooldown,
    )

    results: Dict[str, Dict[str, dict]] = {}
    for split_name, df_eval in eval_sets:
        split_results: Dict[str, dict] = {}
        m = run_rule_backtest(df_eval, strat, env, capture=True)
        split_results[m['algorithm']] = m

        bh = buy_and_hold_baseline(df_eval, initial=env.initial_balance)
        split_results['Buy & Hold'] = bh

        print(f"\n[{split_name}] RSI Rule  return={m['total_return']:+.2f}%  sharpe={m['sharpe_ratio']:+.2f}  "
              f"maxDD={m['max_drawdown']:.2f}%  trades={m['total_trades']}  winR={m['win_rate']:.1f}%")
        print(f"[{split_name}] Buy & Hold return={bh['total_return']:+.2f}%  sharpe={bh['sharpe_ratio']:+.2f}  "
              f"maxDD={bh['max_drawdown']:.2f}%")

        results[split_name] = split_results

    # Backwards-compatible shape (single split -> flat dict)
    out_obj = results
    if len(results) == 1:
        out_obj = next(iter(results.values()))

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(out_obj, f, indent=2, default=str)
    print(f"\nSaved -> {args.output}")

    if not args.no_public_copy:
        os.makedirs('public', exist_ok=True)
        pub = os.path.join('public', 'rule_backtest_results.json')
        with open(pub, 'w', encoding='utf-8') as f:
            json.dump(out_obj, f, indent=2, default=str)
        print(f"Copied -> {pub}")


if __name__ == '__main__':
    main()