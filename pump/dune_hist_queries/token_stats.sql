WITH
  -- SINGLE SCAN: entire query derives from this one table hit
  raw_trades AS (
    SELECT
      block_time,
      trader_id                 AS wallet,
      token_bought_mint_address AS token_address,
      'buy'                     AS action,
      token_bought_amount       AS token_amount,
      amount_usd
    FROM dex_solana.trades
    WHERE project = 'pumpdotfun'
      AND block_time >= NOW() - INTERVAL '7' DAY
      AND token_bought_mint_address IS NOT NULL
      AND amount_usd > 0

    UNION ALL

    SELECT
      block_time,
      trader_id                AS wallet,
      token_sold_mint_address  AS token_address,
      'sell'                   AS action,
      token_sold_amount        AS token_amount,
      amount_usd
    FROM dex_solana.trades
    WHERE project = 'pumpdotfun'
      AND block_time >= NOW() - INTERVAL '7' DAY
      AND token_sold_mint_address IS NOT NULL
      AND amount_usd > 0
  ),

  launch_times AS (
    SELECT
      token_address,
      MIN(block_time) AS launch_time
    FROM raw_trades
    GROUP BY token_address
  ),

  trades_with_launch AS (
    SELECT
      rt.block_time,
      rt.wallet,
      rt.token_address,
      rt.action,
      rt.token_amount,
      rt.amount_usd,
      lt.launch_time,
      date_diff('second', lt.launch_time, rt.block_time) AS seconds_from_launch,
      rt.amount_usd / NULLIF(rt.token_amount, 0)         AS trade_price
    FROM raw_trades rt
    JOIN launch_times lt ON rt.token_address = lt.token_address
  ),

  wallet_volumes AS (
    SELECT
      token_address,
      wallet,
      SUM(CASE WHEN action = 'buy'  THEN amount_usd    ELSE 0 END) AS total_bought_usd,
      SUM(CASE WHEN action = 'sell' THEN amount_usd    ELSE 0 END) AS total_sold_usd,
      SUM(CASE WHEN action = 'buy'  THEN token_amount  ELSE 0 END)
        - SUM(CASE WHEN action = 'sell' THEN token_amount ELSE 0 END) AS net_token_balance
    FROM trades_with_launch
    GROUP BY token_address, wallet
  ),

  wallet_ranks AS (
    SELECT
      token_address,
      wallet,
      total_bought_usd,
      total_sold_usd,
      net_token_balance,
      ROW_NUMBER() OVER (
        PARTITION BY token_address
        ORDER BY net_token_balance DESC
      ) AS holder_rank
    FROM wallet_volumes
  ),

  token_supply AS (
    SELECT
      token_address,
      SUM(CASE WHEN net_token_balance > 0 THEN net_token_balance ELSE 0 END) AS approx_supply,
      SUM(total_bought_usd) AS total_buy_volume,
      SUM(total_sold_usd)   AS total_sell_volume
    FROM wallet_ranks
    GROUP BY token_address
  ),

  concentration AS (
    SELECT
      wr.token_address,
      ROUND(
        SUM(CASE WHEN wr.holder_rank <= 5  AND wr.net_token_balance > 0 THEN wr.net_token_balance ELSE 0 END)
        * 100.0 / NULLIF(MAX(ts.approx_supply), 0), 2
      ) AS top5_holder_pct,
      ROUND(
        SUM(CASE WHEN wr.holder_rank <= 10 AND wr.net_token_balance > 0 THEN wr.net_token_balance ELSE 0 END)
        * 100.0 / NULLIF(MAX(ts.approx_supply), 0), 2
      ) AS top10_holder_pct,
      ROUND(
        SUM(CASE WHEN wr.holder_rank <= 20 AND wr.net_token_balance > 0 THEN wr.net_token_balance ELSE 0 END)
        * 100.0 / NULLIF(MAX(ts.approx_supply), 0), 2
      ) AS top20_holder_pct,
      ROUND(
        SUM(CASE WHEN wr.holder_rank > 10
          THEN wr.total_bought_usd - wr.total_sold_usd ELSE 0 END), 2
      ) AS net_flow_excl_top10,
      ROUND(
        SUM(CASE WHEN wr.holder_rank <= 10
          THEN wr.total_bought_usd + wr.total_sold_usd ELSE 0 END)
        * 100.0 / NULLIF(MAX(ts.total_buy_volume) + MAX(ts.total_sell_volume), 0), 2
      ) AS top10_volume_pct
    FROM wallet_ranks wr
    JOIN token_supply ts ON wr.token_address = ts.token_address
    GROUP BY wr.token_address
  ),

  buy_windows AS (
    SELECT
      token_address,
      wallet,
      seconds_from_launch,
      amount_usd,
      FLOOR(seconds_from_launch / 10) AS window_10s_bucket
    FROM trades_with_launch
    WHERE action = 'buy'
  ),

  window_counts AS (
    SELECT
      token_address,
      window_10s_bucket,
      COUNT(DISTINCT wallet) AS wallets_in_window
    FROM buy_windows
    GROUP BY token_address, window_10s_bucket
  ),

  bundler_metrics AS (
    SELECT
      bw.token_address,
      COUNT(DISTINCT CASE WHEN bw.seconds_from_launch <= 10  THEN bw.wallet END) AS bundler_wallets_10s,
      COUNT(DISTINCT CASE WHEN bw.seconds_from_launch <= 30  THEN bw.wallet END) AS bundler_wallets_30s,
      COUNT(DISTINCT CASE WHEN bw.seconds_from_launch <= 60  THEN bw.wallet END) AS bundler_wallets_60s,
      COUNT(DISTINCT CASE WHEN bw.seconds_from_launch <= 300 THEN bw.wallet END) AS bundler_wallets_5m
    FROM buy_windows bw
    JOIN window_counts wc ON bw.token_address = wc.token_address
                          AND bw.window_10s_bucket = wc.window_10s_bucket
    WHERE wc.wallets_in_window >= 3
    GROUP BY bw.token_address
  ),

  wallet_first_appearance AS (
    SELECT
      token_address,
      wallet,
      MIN(seconds_from_launch) AS first_seen_seconds
    FROM trades_with_launch
    WHERE action = 'buy'
    GROUP BY token_address, wallet
  ),

  repeated_wallet_metrics AS (
    SELECT
      token_address,
      COUNT(DISTINCT CASE WHEN first_seen_seconds <= 60  THEN wallet END) AS early_buyers,
      COUNT(DISTINCT CASE WHEN first_seen_seconds > 300  THEN wallet END) AS late_buyers,
      ROUND(
        COUNT(DISTINCT CASE WHEN first_seen_seconds > 300  THEN wallet END) * 1.0
        / NULLIF(COUNT(DISTINCT CASE WHEN first_seen_seconds <= 60 THEN wallet END), 0), 2
      ) AS late_to_early_ratio
    FROM wallet_first_appearance
    GROUP BY token_address
  ),

  price_metrics AS (
    SELECT
      token_address,
      MIN(CASE WHEN seconds_from_launch <= 10 THEN trade_price END)       AS price_at_launch,
      MAX(CASE WHEN seconds_from_launch <= 300 THEN trade_price END)       AS peak_price_5m,
      ROUND(STDDEV(CASE WHEN seconds_from_launch <= 300 THEN trade_price END), 8) AS price_stddev_5m,
      ROUND(
        SUM(CASE WHEN action = 'buy'  AND seconds_from_launch <= 300 THEN amount_usd ELSE 0 END)
        - SUM(CASE WHEN action = 'sell' AND seconds_from_launch <= 300 THEN amount_usd ELSE 0 END), 2
      ) AS net_buy_pressure_5m,
      ROUND(
        MAX(CASE WHEN seconds_from_launch <= 300 THEN trade_price END)
        / NULLIF(MIN(CASE WHEN seconds_from_launch <= 60 THEN trade_price END), 0), 2
      ) AS upside_burst_5m
    FROM trades_with_launch
    GROUP BY token_address
  ),

  token_metrics AS (
    SELECT
      token_address,
      MIN(launch_time) AS launch_time,

      -- 30 SECOND
      COUNT(DISTINCT CASE WHEN action = 'buy'  AND seconds_from_launch <= 30 THEN wallet END) AS buyers_30s,
      COUNT(DISTINCT CASE WHEN action = 'sell' AND seconds_from_launch <= 30 THEN wallet END) AS sellers_30s,
      SUM(CASE WHEN seconds_from_launch <= 30 THEN amount_usd END)                            AS volume_30s,
      COUNT(CASE WHEN action = 'buy'  AND seconds_from_launch <= 30 THEN 1 END)               AS buy_txns_30s,
      COUNT(CASE WHEN action = 'sell' AND seconds_from_launch <= 30 THEN 1 END)               AS sell_txns_30s,

      -- 1 MINUTE
      COUNT(DISTINCT CASE WHEN action = 'buy'  AND seconds_from_launch <= 60 THEN wallet END) AS buyers_1m,
      COUNT(DISTINCT CASE WHEN action = 'sell' AND seconds_from_launch <= 60 THEN wallet END) AS sellers_1m,
      SUM(CASE WHEN seconds_from_launch <= 60 THEN amount_usd END)                            AS volume_1m,
      COUNT(CASE WHEN action = 'buy'  AND seconds_from_launch <= 60 THEN 1 END)               AS buy_txns_1m,
      COUNT(CASE WHEN action = 'sell' AND seconds_from_launch <= 60 THEN 1 END)               AS sell_txns_1m,
      ROUND(APPROX_PERCENTILE(CASE WHEN action = 'buy' AND seconds_from_launch <= 60 THEN amount_usd END, 0.25), 2) AS buy_size_p25_1m,
      ROUND(APPROX_PERCENTILE(CASE WHEN action = 'buy' AND seconds_from_launch <= 60 THEN amount_usd END, 0.50), 2) AS buy_size_p50_1m,
      ROUND(APPROX_PERCENTILE(CASE WHEN action = 'buy' AND seconds_from_launch <= 60 THEN amount_usd END, 0.75), 2) AS buy_size_p75_1m,
      ROUND(APPROX_PERCENTILE(CASE WHEN action = 'buy' AND seconds_from_launch <= 60 THEN amount_usd END, 0.95), 2) AS buy_size_p95_1m,

      -- 3 MINUTE
      COUNT(DISTINCT CASE WHEN action = 'buy'  AND seconds_from_launch <= 180 THEN wallet END) AS buyers_3m,
      COUNT(DISTINCT CASE WHEN action = 'sell' AND seconds_from_launch <= 180 THEN wallet END) AS sellers_3m,
      SUM(CASE WHEN seconds_from_launch <= 180 THEN amount_usd END)                            AS volume_3m,

      -- 5 MINUTE
      COUNT(DISTINCT CASE WHEN action = 'buy'  AND seconds_from_launch <= 300 THEN wallet END) AS buyers_5m,
      COUNT(DISTINCT CASE WHEN action = 'sell' AND seconds_from_launch <= 300 THEN wallet END) AS sellers_5m,
      SUM(CASE WHEN seconds_from_launch <= 300 THEN amount_usd END)                            AS volume_5m,
      COUNT(DISTINCT CASE WHEN seconds_from_launch <= 300 THEN wallet END)                     AS unique_wallets_5m,
      COUNT(CASE WHEN action = 'buy'  AND seconds_from_launch <= 300 THEN 1 END)               AS buy_txns_5m,
      COUNT(CASE WHEN action = 'sell' AND seconds_from_launch <= 300 THEN 1 END)               AS sell_txns_5m,
      ROUND(APPROX_PERCENTILE(CASE WHEN action = 'buy' AND seconds_from_launch <= 300 THEN amount_usd END, 0.25), 2) AS buy_size_p25_5m,
      ROUND(APPROX_PERCENTILE(CASE WHEN action = 'buy' AND seconds_from_launch <= 300 THEN amount_usd END, 0.50), 2) AS buy_size_p50_5m,
      ROUND(APPROX_PERCENTILE(CASE WHEN action = 'buy' AND seconds_from_launch <= 300 THEN amount_usd END, 0.75), 2) AS buy_size_p75_5m,
      ROUND(APPROX_PERCENTILE(CASE WHEN action = 'buy' AND seconds_from_launch <= 300 THEN amount_usd END, 0.95), 2) AS buy_size_p95_5m,

      -- 30 MINUTE
      COUNT(DISTINCT CASE WHEN action = 'buy'  AND seconds_from_launch <= 1800 THEN wallet END) AS buyers_30m,
      COUNT(DISTINCT CASE WHEN action = 'sell' AND seconds_from_launch <= 1800 THEN wallet END) AS sellers_30m,
      SUM(CASE WHEN seconds_from_launch <= 1800 THEN amount_usd END)                            AS volume_30m,
      COUNT(DISTINCT CASE WHEN seconds_from_launch <= 1800 THEN wallet END)                     AS unique_wallets_30m,

      -- HOLDER GROWTH
      COUNT(DISTINCT CASE WHEN action = 'buy' AND seconds_from_launch <= 60   THEN wallet END) AS holders_at_1m,
      COUNT(DISTINCT CASE WHEN action = 'buy' AND seconds_from_launch <= 300  THEN wallet END) AS holders_at_5m,
      COUNT(DISTINCT CASE WHEN action = 'buy' AND seconds_from_launch <= 1800 THEN wallet END) AS holders_at_30m,

      -- OUTCOME LABELS
      MAX(CASE WHEN seconds_from_launch >= 1800  THEN 1 ELSE 0 END) AS survived_30m,
      MAX(CASE WHEN seconds_from_launch >= 3600  THEN 1 ELSE 0 END) AS survived_1h,
      MAX(CASE WHEN seconds_from_launch >= 86400 THEN 1 ELSE 0 END) AS survived_24h,

      COUNT(DISTINCT CASE WHEN action = 'buy' AND seconds_from_launch > 300 THEN wallet END) AS new_buyers_after_5m,
      SUM(CASE WHEN seconds_from_launch > 300 THEN amount_usd ELSE 0 END)                    AS volume_after_5m,
      COUNT(DISTINCT wallet)                                                                   AS total_unique_wallets,
      ROUND(
        SUM(CASE WHEN action = 'buy' THEN amount_usd ELSE 0 END)
        / NULLIF(COUNT(DISTINCT CASE WHEN action = 'buy' THEN wallet END), 0), 2
      ) AS volume_per_unique_buyer

    FROM trades_with_launch
    GROUP BY token_address
  )

SELECT
  tm.token_address,
  tm.launch_time,
  tm.volume_30s, tm.volume_1m, tm.volume_3m, tm.volume_5m, tm.volume_30m, tm.volume_after_5m,
  tm.buyers_30s, tm.sellers_30s, tm.buyers_1m, tm.sellers_1m,
  tm.buyers_3m, tm.sellers_3m, tm.buyers_5m, tm.sellers_5m,
  tm.buyers_30m, tm.sellers_30m,
  tm.unique_wallets_5m, tm.unique_wallets_30m, tm.total_unique_wallets,
  tm.buy_txns_30s, tm.sell_txns_30s, tm.buy_txns_1m, tm.sell_txns_1m,
  tm.buy_txns_5m, tm.sell_txns_5m,
  tm.buy_size_p25_1m, tm.buy_size_p50_1m, tm.buy_size_p75_1m, tm.buy_size_p95_1m,
  tm.buy_size_p25_5m, tm.buy_size_p50_5m, tm.buy_size_p75_5m, tm.buy_size_p95_5m,
  ROUND(tm.buyers_30s * 1.0 / NULLIF(tm.sellers_30s, 0), 2) AS buy_sell_ratio_30s,
  ROUND(tm.buyers_1m  * 1.0 / NULLIF(tm.sellers_1m,  0), 2) AS buy_sell_ratio_1m,
  ROUND(tm.buyers_5m  * 1.0 / NULLIF(tm.sellers_5m,  0), 2) AS buy_sell_ratio_5m,
  ROUND(tm.buyers_30m * 1.0 / NULLIF(tm.sellers_30m, 0), 2) AS buy_sell_ratio_30m,
  tm.volume_per_unique_buyer,
  tm.holders_at_1m, tm.holders_at_5m, tm.holders_at_30m,
  (tm.holders_at_5m  - tm.holders_at_1m)  AS holder_growth_1m_to_5m,
  (tm.holders_at_30m - tm.holders_at_5m)  AS holder_growth_5m_to_30m,
  c.top5_holder_pct, c.top10_holder_pct, c.top20_holder_pct,
  c.top10_volume_pct, c.net_flow_excl_top10,
  b.bundler_wallets_10s, b.bundler_wallets_30s, b.bundler_wallets_60s, b.bundler_wallets_5m,
  ROUND(b.bundler_wallets_60s * 100.0 / NULLIF(tm.buyers_1m, 0), 2) AS bundler_pct_of_buyers_1m,
  rw.early_buyers, rw.late_buyers, rw.late_to_early_ratio,
  ROUND(tm.new_buyers_after_5m * 100.0 / NULLIF(tm.total_unique_wallets, 0), 2) AS organic_buyer_pct,
  ROUND(tm.unique_wallets_30m  * 100.0 / NULLIF(tm.unique_wallets_5m,   0), 2)  AS wallet_retention_5m_to_30m,
  pm.price_at_launch, pm.peak_price_5m, pm.price_stddev_5m,
  pm.net_buy_pressure_5m, pm.upside_burst_5m,
  tm.survived_30m, tm.survived_1h, tm.survived_24h

FROM token_metrics tm
LEFT JOIN concentration           c  ON tm.token_address = c.token_address
LEFT JOIN bundler_metrics         b  ON tm.token_address = b.token_address
LEFT JOIN repeated_wallet_metrics rw ON tm.token_address = rw.token_address
LEFT JOIN price_metrics           pm ON tm.token_address = pm.token_address
ORDER BY tm.launch_time DESC;