"""
Gymnasium trading environment for cryptocurrency futures.

Supports long / short / flat positions with configurable leverage,
transaction fees, and realistic position management.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class CryptoFuturesEnv(gym.Env):
    """
    Crypto futures trading env.

    Actions
    -------
    0 = Hold   – do nothing
    1 = Long   – close any short, open long
    2 = Short  – close any long, open short
    3 = Close  – flatten position

    Observation
    -----------
    [feature_columns...] + [position_direction, unrealised_pnl%, hold_time]
    """

    metadata = {"render_modes": ["human"]}
    HOLD, LONG, SHORT, CLOSE = 0, 1, 2, 3

    def __init__(
            self,
            df,
            feature_columns,
            initial_balance: float = 10_000.0,
            leverage: int = 1,
            fee_rate: float = 0.0004,  # Binance futures taker 0.04 %
            max_position_frac: float = 1.0,  # fraction of equity per trade
            stop_loss_pct: float = 0.0,  # 0 = disabled, e.g. 0.02 = 2%
            take_profit_pct: float = 0.0,  # 0 = disabled, e.g. 0.03 = 3%
            min_hold_steps: int = 0,  # 0 = disabled, e.g. 16 = 4h on 15m bars
            cooldown_steps: int = 0,  # 0 = disabled, bars to wait after a trade
            trade_penalty: float = 0.0,  # reward penalty per trade (fraction of initial_balance)
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.feature_columns = list(feature_columns)
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.fee_rate = fee_rate
        self.max_position_frac = max_position_frac
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.min_hold_steps = int(min_hold_steps)
        self.cooldown_steps = int(cooldown_steps)
        self.trade_penalty = float(trade_penalty)
        self.cooldown = 0

        self.action_space = spaces.Discrete(4)
        n_obs = len(self.feature_columns) + 3
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n_obs,), dtype=np.float32)

        self.reset()

    # ------------------------------------------------------------------ reset
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_idx = 0
        self.balance = self.initial_balance
        self.position = 0  # +1 long, -1 short, 0 flat
        self.position_size = 0.0  # units of base asset
        self.entry_price = 0.0
        self.entry_step = 0
        self.trades: list[dict] = []
        self.equity_curve = [self.initial_balance]
        self.cooldown = 0
        return self._obs(), {}

    # -------------------------------------------------------------- helpers
    def _price(self):
        return float(self.df.iloc[self.step_idx]['close'])

    def _equity(self):
        p = self._price()
        upnl = self.position * self.position_size * (p - self.entry_price) if self.position else 0
        return self.balance + upnl

    def _obs(self):
        if self.step_idx >= len(self.df):
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        row = self.df.iloc[self.step_idx]
        feats = row[self.feature_columns].values.astype(np.float32)

        # position meta-features
        p = self._price()
        upnl_pct = (self.position * (p - self.entry_price) / self.entry_price
                    if self.position and self.entry_price else 0.0)
        hold_dur = min((self.step_idx - self.entry_step) / 200.0, 1.0) if self.position else 0.0

        extra = np.array([float(self.position), upnl_pct, hold_dur], dtype=np.float32)
        return np.concatenate([feats, extra])

    # ----------------------------------------------------------- open / close
    def _open(self, direction: int):
        price = self._price()
        notional = self.balance * self.max_position_frac * self.leverage
        fee = notional * self.fee_rate
        self.position = direction
        self.position_size = notional / price
        self.entry_price = price
        self.entry_step = self.step_idx
        self.balance -= fee

        self.trades.append(dict(
            step=self.step_idx,
            timestamp=str(self.df.iloc[self.step_idx].get('timestamp', '')),
            action='LONG' if direction == 1 else 'SHORT',
            price=price, size=self.position_size, fee=fee,
        ))

    def _close(self):
        if not self.position:
            return 0.0
        price = self._price()
        pnl = self.position * self.position_size * (price - self.entry_price)
        fee = self.position_size * price * self.fee_rate
        self.balance += pnl - fee

        self.trades.append(dict(
            step=self.step_idx,
            timestamp=str(self.df.iloc[self.step_idx].get('timestamp', '')),
            action='CLOSE', price=price, size=self.position_size,
            pnl=round(pnl, 4), fee=round(fee, 4),
        ))
        self.position = 0
        self.position_size = 0.0
        self.entry_price = 0.0
        return pnl

    # --------------------------------------------------------- SL / TP check
    def _check_sl_tp(self):
        """
        Check if the current candle's high/low triggers a stop-loss or
        take-profit.  Uses intra-bar extremes for realistic fills.
        Returns True if position was closed.
        """
        if not self.position or (not self.stop_loss_pct and not self.take_profit_pct):
            return False

        row = self.df.iloc[self.step_idx]
        high = float(row['high'])
        low = float(row['low'])

        if self.position == 1:  # LONG
            if self.stop_loss_pct and low <= self.entry_price * (1 - self.stop_loss_pct):
                fill = self.entry_price * (1 - self.stop_loss_pct)
                self._close_at(fill, reason='SL')
                return True
            if self.take_profit_pct and high >= self.entry_price * (1 + self.take_profit_pct):
                fill = self.entry_price * (1 + self.take_profit_pct)
                self._close_at(fill, reason='TP')
                return True
        elif self.position == -1:  # SHORT
            if self.stop_loss_pct and high >= self.entry_price * (1 + self.stop_loss_pct):
                fill = self.entry_price * (1 + self.stop_loss_pct)
                self._close_at(fill, reason='SL')
                return True
            if self.take_profit_pct and low <= self.entry_price * (1 - self.take_profit_pct):
                fill = self.entry_price * (1 - self.take_profit_pct)
                self._close_at(fill, reason='TP')
                return True
        return False

    def _close_at(self, fill_price: float, reason: str = 'CLOSE'):
        """Close position at a specific fill price (used by SL/TP)."""
        if not self.position:
            return 0.0
        pnl = self.position * self.position_size * (fill_price - self.entry_price)
        fee = self.position_size * fill_price * self.fee_rate
        self.balance += pnl - fee

        self.trades.append(dict(
            step=self.step_idx,
            timestamp=str(self.df.iloc[self.step_idx].get('timestamp', '')),
            action=reason, price=round(fill_price, 6), size=self.position_size,
            pnl=round(pnl, 4), fee=round(fee, 4),
        ))
        self.position = 0
        self.position_size = 0.0
        self.entry_price = 0.0
        return pnl

    # ------------------------------------------------------------------ step
    def step(self, action):
        assert self.action_space.contains(action)
        if self.step_idx >= len(self.df) - 1:
            return self._obs(), 0.0, True, False, {}

        prev_equity = self._equity()

        # ── SL / TP fires before agent action ──
        sl_tp_closed = self._check_sl_tp()
        if sl_tp_closed and self.cooldown_steps:
            # If SL/TP closed intrabar, do not allow discretionary re-entry this same bar
            self.cooldown = max(self.cooldown, self.cooldown_steps)

        trades_this_step = 0

        # Cooldown tick
        if self.cooldown > 0:
            self.cooldown -= 1

        # Minimum hold time before a voluntary CLOSE or flip
        hold_steps = (self.step_idx - self.entry_step) if self.position else 0
        can_close = (not self.position) or (hold_steps >= self.min_hold_steps)
        can_trade = (self.cooldown == 0) and (not sl_tp_closed)

        # execute (with anti-churn gates)
        if action == self.LONG and self.position <= 0 and can_trade:
            if self.position == -1:
                if can_close:
                    self._close(); trades_this_step += 1
                else:
                    action = self.HOLD
            if self.position == 0:
                self._open(+1); trades_this_step += 1
                self.cooldown = self.cooldown_steps

        elif action == self.SHORT and self.position >= 0 and can_trade:
            if self.position == +1:
                if can_close:
                    self._close(); trades_this_step += 1
                else:
                    action = self.HOLD
            if self.position == 0:
                self._open(-1); trades_this_step += 1
                self.cooldown = self.cooldown_steps

        elif action == self.CLOSE and self.position != 0:
            if can_close and can_trade:
                self._close(); trades_this_step += 1
                self.cooldown = self.cooldown_steps
            else:
                action = self.HOLD

        self.step_idx += 1

        done = self.step_idx >= len(self.df) - 1
        if done and self.position:
            self._close()
            trades_this_step += 1

        cur_equity = self._equity()
        reward = (cur_equity - prev_equity) / self.initial_balance
        reward -= self.trade_penalty * trades_this_step
        self.equity_curve.append(cur_equity)

        return self._obs(), reward, done, False, {
            'equity': cur_equity,
            'balance': self.balance,
            'position': self.position,
        }

    # ------------------------------------------------------------ metrics
    def get_metrics(self) -> dict:
        eq = np.array(self.equity_curve)
        rets = np.diff(eq) / eq[:-1]

        total_return = (eq[-1] / self.initial_balance - 1) * 100

        # Sharpe – annualised for 15-min bars (4 × 24 × 365 = 35 040)
        ann = np.sqrt(4 * 24 * 365)
        sharpe = (rets.mean() / rets.std() * ann) if rets.std() > 0 else 0.0

        # Max drawdown
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / peak
        max_dd = dd.max() * 100

        closed = [t for t in self.trades if t['action'] in ('CLOSE', 'SL', 'TP') and 'pnl' in t]
        wins = sum(1 for t in closed if t['pnl'] > 0)
        win_rate = (wins / len(closed) * 100) if closed else 0.0

        avg_pnl = float(np.mean([t['pnl'] for t in closed])) if closed else 0.0
        sl_count = sum(1 for t in self.trades if t['action'] == 'SL')
        tp_count = sum(1 for t in self.trades if t['action'] == 'TP')

        return dict(
            total_return=round(total_return, 2),
            sharpe_ratio=round(sharpe, 2),
            max_drawdown=round(max_dd, 2),
            win_rate=round(win_rate, 2),
            total_trades=len(closed),
            avg_trade_pnl=round(avg_pnl, 4),
            final_balance=round(eq[-1], 2),
            sl_hits=sl_count,
            tp_hits=tp_count,
        )
