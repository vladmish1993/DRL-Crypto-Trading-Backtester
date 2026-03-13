WITH
  -- INSTRUCTION CALLS: single scan covers both graduation + withdrawal
  instruction_events AS (
    SELECT
      bytearray_substring(data, 1, 8)  AS discriminator,
      account_arguments[3]              AS token_address_arg3,
      account_arguments[2]              AS token_address_arg2,
      block_time,
      inner_instructions,
      is_inner
    FROM solana.instruction_calls
    WHERE executing_account = '6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P'
      AND block_time >= NOW() - INTERVAL '7' DAY
      AND tx_success = TRUE
      AND bytearray_substring(data, 1, 8) IN (
        0x9beae792ec9ea21e,  -- graduation (migrate to Raydium)
        0xb712469c946da122   -- withdrawal (liquidity removed = rug)
      )
  ),

  graduations AS (
    SELECT token_address, MIN(graduated_at) AS graduated_at
    FROM (
      SELECT token_address_arg3 AS token_address, block_time AS graduated_at
      FROM instruction_events
      WHERE discriminator = 0x9beae792ec9ea21e
        AND (cardinality(inner_instructions) > 0 OR is_inner = true)
      UNION
      SELECT token_address_arg2 AS token_address, block_time AS graduated_at
      FROM instruction_events
      WHERE discriminator = 0x9beae792ec9ea21e
        AND (cardinality(inner_instructions) > 0 OR is_inner = true)
    )
    GROUP BY token_address
  ),

  withdrawals AS (
    SELECT token_address, MIN(withdrawn_at) AS withdrawn_at
    FROM (
      SELECT token_address_arg3 AS token_address, block_time AS withdrawn_at
      FROM instruction_events
      WHERE discriminator = 0xb712469c946da122
      UNION
      SELECT token_address_arg2 AS token_address, block_time AS withdrawn_at
      FROM instruction_events
      WHERE discriminator = 0xb712469c946da122
    )
    GROUP BY token_address
  ),

  -- TRANSFERS: single scan covers token_creates + deployer_transfers
  all_transfers AS (
    SELECT
      token_mint_address,
      tx_signer,
      from_owner,
      action,
      amount,
      block_time,
      outer_instruction_index,
      inner_instruction_index
    FROM tokens_solana.transfers
    WHERE block_time >= NOW() - INTERVAL '7' DAY
      AND outer_executing_account = '6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P'
      AND action IN ('mint', 'transfer')
      AND token_mint_address IS NOT NULL
  ),

  token_creates AS (
    SELECT
      token_mint_address AS token_address,
      tx_signer          AS dev_wallet,
      MIN(block_time)    AS created_at,
      MAX(amount) / 1e6  AS total_supply
    FROM all_transfers
    WHERE action = 'mint'
    GROUP BY token_mint_address, tx_signer
  ),

  deployer_transfers AS (
    SELECT
      t.token_mint_address AS token_address,
      COUNT(*)             AS deployer_transfer_count
    FROM all_transfers t
    JOIN token_creates tc ON t.token_mint_address = tc.token_address
                          AND t.from_owner = tc.dev_wallet
    WHERE t.action = 'transfer'
      AND t.outer_instruction_index = 2
      AND t.inner_instruction_index = 0
    GROUP BY t.token_mint_address
  ),

  -- PUMP CALL BUY: single scan covers snipers + self_buys + manipulators
  pump_buys AS (
    SELECT
      pcb.account_mint    AS token_address,
      pcb.account_user    AS wallet,
      pcb.call_block_time AS block_time,
      pcb.amount / 1e6    AS token_amount
    FROM pumpdotfun_solana.pump_call_buy pcb
    WHERE pcb.call_block_time >= NOW() - INTERVAL '7' DAY
      AND pcb.account_mint IN (SELECT token_address FROM token_creates)
  ),

  pump_buy_metrics AS (
    SELECT
      pb.token_address,
      COUNT(DISTINCT CASE
        WHEN pb.wallet != tc.dev_wallet
         AND pb.block_time BETWEEN tc.created_at + INTERVAL '1' SECOND
                               AND tc.created_at + INTERVAL '40' SECOND
        THEN pb.wallet END)                           AS sniper_count,
      COUNT(CASE WHEN pb.wallet = tc.dev_wallet
                 THEN 1 END)                          AS self_buy_count
    FROM pump_buys pb
    JOIN token_creates tc ON pb.token_address = tc.token_address
    GROUP BY pb.token_address
  ),

  manipulator_counts AS (
    SELECT
      token_address,
      COUNT(DISTINCT wallet) AS manipulator_count
    FROM (
      SELECT pb.token_address, pb.wallet
      FROM pump_buys pb
      JOIN token_creates tc ON pb.token_address = tc.token_address
      WHERE pb.wallet != tc.dev_wallet
      GROUP BY pb.token_address, pb.wallet
      HAVING COUNT(*) > 6
    )
    GROUP BY token_address
  ),

  -- PUMP.FUN TRADES: for dev behaviour and wallet freshness
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
    SELECT token_address, MIN(block_time) AS launch_time
    FROM raw_trades
    GROUP BY token_address
  ),

  trades_with_launch AS (
    SELECT
      rt.block_time, rt.wallet, rt.token_address, rt.action,
      rt.token_amount, rt.amount_usd, lt.launch_time,
      date_diff('second', lt.launch_time, rt.block_time) AS seconds_from_launch
    FROM raw_trades rt
    JOIN launch_times lt ON rt.token_address = lt.token_address
  ),

  dev_behaviour AS (
    SELECT
      tc.token_address,
      tc.dev_wallet,
      MAX(CASE WHEN twl.wallet = tc.dev_wallet AND twl.action = 'sell' AND twl.seconds_from_launch <= 300  THEN 1 ELSE 0 END) AS dev_sold_in_5m,
      SUM(CASE WHEN twl.wallet = tc.dev_wallet AND twl.action = 'sell' AND twl.seconds_from_launch <= 300  THEN twl.amount_usd ELSE 0 END) AS dev_sell_volume_5m,
      MAX(CASE WHEN twl.wallet = tc.dev_wallet AND twl.action = 'sell' AND twl.seconds_from_launch <= 1800 THEN 1 ELSE 0 END) AS dev_sold_in_30m,
      SUM(CASE WHEN twl.wallet = tc.dev_wallet AND twl.action = 'sell' THEN twl.amount_usd ELSE 0 END) AS dev_total_sell_volume,
      SUM(CASE WHEN twl.wallet = tc.dev_wallet AND twl.action = 'buy'  THEN twl.amount_usd ELSE 0 END) AS dev_total_buy_volume
    FROM token_creates tc
    LEFT JOIN trades_with_launch twl ON tc.token_address = twl.token_address
    GROUP BY tc.token_address, tc.dev_wallet
  ),

  -- Wallet freshness reuses raw_trades — no extra table scan
  wallet_first_seen AS (
    SELECT wallet, MIN(block_time) AS wallet_first_seen
    FROM raw_trades
    GROUP BY wallet
  ),

  fresh_wallet_metrics AS (
    SELECT
      twl.token_address,
      COUNT(DISTINCT twl.wallet)                              AS total_early_buyers,
      COUNT(DISTINCT CASE WHEN wfs.wallet_first_seen >= NOW() - INTERVAL '7'  DAY THEN twl.wallet END) AS fresh_wallet_count,
      COUNT(DISTINCT CASE WHEN wfs.wallet_first_seen <  NOW() - INTERVAL '30' DAY THEN twl.wallet END) AS established_wallet_count,
      ROUND(COUNT(DISTINCT CASE WHEN wfs.wallet_first_seen >= NOW() - INTERVAL '7'  DAY THEN twl.wallet END) * 100.0 / NULLIF(COUNT(DISTINCT twl.wallet), 0), 2) AS fresh_wallet_pct,
      ROUND(COUNT(DISTINCT CASE WHEN wfs.wallet_first_seen <  NOW() - INTERVAL '30' DAY THEN twl.wallet END) * 100.0 / NULLIF(COUNT(DISTINCT twl.wallet), 0), 2) AS established_wallet_pct
    FROM trades_with_launch twl
    LEFT JOIN wallet_first_seen wfs ON twl.wallet = wfs.wallet
    WHERE twl.action = 'buy'
      AND twl.seconds_from_launch <= 300
    GROUP BY twl.token_address
  ),

  migration_metrics AS (
    SELECT
      g.token_address, g.graduated_at,
      date_diff('second', lt.launch_time, g.graduated_at) AS seconds_to_graduation,
      date_diff('minute', lt.launch_time, g.graduated_at) AS minutes_to_graduation
    FROM graduations g
    JOIN launch_times lt ON g.token_address = lt.token_address
  ),

  withdrawal_metrics AS (
    SELECT
      w.token_address, w.withdrawn_at,
      date_diff('second', lt.launch_time, w.withdrawn_at) AS seconds_to_withdrawal,
      date_diff('minute', lt.launch_time, w.withdrawn_at) AS minutes_to_withdrawal
    FROM withdrawals w
    JOIN launch_times lt ON w.token_address = lt.token_address
  ),

  raydium_trades AS (
    SELECT
      COALESCE(token_bought_mint_address, token_sold_mint_address) AS token_address,
      COUNT(DISTINCT trader_id)                                     AS raydium_unique_traders,
      COUNT(DISTINCT CASE WHEN token_bought_mint_address IS NOT NULL THEN trader_id END) AS raydium_unique_buyers,
      SUM(amount_usd)                                               AS raydium_volume,
      COUNT(*)                                                      AS raydium_trade_count
    FROM dex_solana.trades
    WHERE project = 'raydium'
      AND block_time >= NOW() - INTERVAL '7' DAY
      AND amount_usd > 0
      AND COALESCE(token_bought_mint_address, token_sold_mint_address)
          IN (SELECT token_address FROM graduations)
    GROUP BY 1
  )

SELECT
  lt.token_address,
  lt.launch_time,
  tc.total_supply,
  db.dev_wallet,
  db.dev_sold_in_5m,
  db.dev_sell_volume_5m,
  db.dev_sold_in_30m,
  db.dev_total_sell_volume,
  db.dev_total_buy_volume,
  ROUND(db.dev_total_sell_volume * 100.0 / NULLIF(db.dev_total_sell_volume + db.dev_total_buy_volume, 0), 2) AS dev_sell_ratio_pct,
  COALESCE(pbm.self_buy_count, 0)          AS dev_self_buy_count,
  COALESCE(dt.deployer_transfer_count, 0)  AS deployer_transfer_count,
  COALESCE(pbm.sniper_count, 0)            AS sniper_count,
  COALESCE(mc.manipulator_count, 0)        AS manipulator_count,
  CASE WHEN mm.token_address IS NOT NULL THEN 1 ELSE 0 END AS reached_graduation,
  mm.minutes_to_graduation,
  mm.seconds_to_graduation,
  mm.graduated_at,
  CASE WHEN wm.token_address IS NOT NULL THEN 1 ELSE 0 END AS liquidity_withdrawn,
  wm.minutes_to_withdrawal,
  wm.seconds_to_withdrawal,
  wm.withdrawn_at,
  CASE WHEN mm.token_address IS NOT NULL AND wm.token_address IS NOT NULL THEN 1 ELSE 0 END AS graduated_then_rugged,
  rt.raydium_unique_traders,
  rt.raydium_unique_buyers,
  rt.raydium_volume,
  rt.raydium_trade_count,
  fw.total_early_buyers,
  fw.fresh_wallet_count,
  fw.established_wallet_count,
  fw.fresh_wallet_pct,
  fw.established_wallet_pct

FROM launch_times lt
LEFT JOIN token_creates        tc  ON lt.token_address = tc.token_address
LEFT JOIN dev_behaviour        db  ON lt.token_address = db.token_address
LEFT JOIN pump_buy_metrics     pbm ON lt.token_address = pbm.token_address
LEFT JOIN manipulator_counts   mc  ON lt.token_address = mc.token_address
LEFT JOIN deployer_transfers   dt  ON lt.token_address = dt.token_address
LEFT JOIN migration_metrics    mm  ON lt.token_address = mm.token_address
LEFT JOIN withdrawal_metrics   wm  ON lt.token_address = wm.token_address
LEFT JOIN raydium_trades        rt  ON lt.token_address = rt.token_address
LEFT JOIN fresh_wallet_metrics fw  ON lt.token_address = fw.token_address
ORDER BY lt.launch_time DESC;