#!/usr/bin/env bash
set -e

echo "═══════════════════════════════════════════════════"
echo "  DRL Crypto Trading Backtester — Full Pipeline"
echo "═══════════════════════════════════════════════════"
echo ""

EPISODES=${1:-50}
DATA="data/SOL_USDT_15m.csv"

# Step 1: Check / generate data
if [ ! -f "$DATA" ]; then
    echo "→ No data found. Generating sample data..."
    python scripts/generate_sample_data.py
    echo ""
fi

# Step 2: Train + backtest
echo "→ Training all 4 DRL models ($EPISODES episodes each)..."
echo ""
python scripts/train_all.py --data "$DATA" --episodes "$EPISODES"
echo ""

# Step 3: Stop-loss sweep
echo "→ Running stop-loss sweep (0.5% – 5.0%) ..."
echo ""
python scripts/sl_sweep.py --data "$DATA"
echo ""

# Step 4: Start frontend
echo "═══════════════════════════════════════════════════"
echo "  Done! Start the dashboard with: pnpm dev"
echo "  Then open http://localhost:3000"
echo "═══════════════════════════════════════════════════"