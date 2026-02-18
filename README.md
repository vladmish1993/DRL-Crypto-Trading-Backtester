# DRL Crypto Trading Backtester

A working deep reinforcement learning framework for cryptocurrency futures trading.
Trains and compares **4 DRL algorithms** on **SOL/USDT 15m** Binance Futures data,
then visualises results in an interactive Next.js dashboard.

Based on [A comparative study of Bitcoin and Ripple cryptocurrencies trading using Deep Reinforcement Learning algorithms](https://arxiv.org/abs/2505.07660) (Fangnon et al., 2025).

---

## Algorithms

| # | Algorithm | Key Idea |
|---|-----------|----------|
| 1 | **DQN** | Vanilla Deep Q-Network with target network |
| 2 | **Double DQN** | Online network selects actions, target evaluates — reduces overestimation |
| 3 | **Dueling DQN** | Separate Value V(s) and Advantage A(s,a) streams |
| 4 | **A2C** | Advantage Actor-Critic with shared feature trunk |

All implemented from scratch in PyTorch (no SB3 dependency for the core agents).

## Quick Start

```bash
# 1. Install dependencies
pnpm install
pip install -r requirements.txt

# 2a. Download real SOL/USDT data from Binance
python scripts/fetch_data.py --symbol SOL/USDT --timeframe 15m --start 2023-06-01

# 2b. Or generate synthetic test data (no API needed)
python scripts/generate_sample_data.py

# 3. Train all 4 models + run backtests
python scripts/train_all.py --episodes 50

# 4. Launch the dashboard
pnpm dev
# Open http://localhost:3000
```

Or use the convenience script:
```bash
chmod +x run.sh
./run.sh 50     # 50 episodes
```

## Project Structure

```
scripts/
  fetch_data.py            # Download SOL/USDT from Binance Futures (ccxt)
  generate_sample_data.py  # Generate synthetic data for testing
  indicators.py            # Technical indicators (SMA, RSI, MACD, BB, ATR)
  trading_env.py           # Gymnasium futures trading environment
  models.py                # All 4 DRL agents (PyTorch)
  train_all.py             # Training + backtesting pipeline
data/
  SOL_USDT_15m.csv         # OHLCV data (created by fetch or generate scripts)
models/                    # Saved model weights (.pt files)
results/
  backtest_results.json    # Full backtest output
app/
  page.tsx                 # Next.js dashboard (reads real results)
  api/results/route.ts     # API endpoint for results
```

## Trading Environment

The `CryptoFuturesEnv` implements a realistic futures trading simulation:

- **Actions**: Hold, Open Long, Open Short, Close Position
- **Fees**: 0.04% taker fee (Binance Futures default)
- **Leverage**: Configurable (default 1×)
- **Observations**: 14 normalised features + 3 position meta-features
- **Reward**: Step-wise equity change / initial balance
- **Indicators**: SMA(20,50), RSI(14), MACD, Bollinger Bands, ATR(14), Volume Ratio

## Configuration

Edit hyperparameters in `scripts/train_all.py`:

```python
hp = dict(
    lr=1e-4,            # Learning rate
    gamma=0.99,         # Discount factor
    eps_start=1.0,      # Initial exploration
    eps_end=0.05,       # Final exploration
    eps_decay=0.998,    # Decay per step
    buffer_size=50_000, # Replay buffer
    batch_size=64,      # Mini-batch size
    target_update=1000, # Target network sync interval
)
```

## Tech Stack

- **Models**: PyTorch (custom DQN/Double DQN/Dueling DQN/A2C)
- **Environment**: Gymnasium
- **Data**: ccxt (Binance Futures) + pandas
- **Frontend**: Next.js 15, React 19, Recharts, Tailwind CSS, shadcn/ui
- **Indicators**: NumPy / pandas

## References

1. Fangnon, D. et al. (2025). "A comparative study of Bitcoin and Ripple cryptocurrencies trading using Deep Reinforcement Learning algorithms." arXiv:2505.07660.
2. Wang, Z. et al. (2016). "Dueling Network Architectures for Deep Reinforcement Learning." ICML.
3. van Hasselt, H. et al. (2016). "Deep Reinforcement Learning with Double Q-learning." AAAI.
4. Mnih, V. et al. (2016). "Asynchronous Methods for Deep Reinforcement Learning." ICML.

## License

MIT
