-- ClickHouse Initialization Script for Indian AI Trader
-- Creates time-series tables for market data

-- ============================================
-- DATABASE
-- ============================================
CREATE DATABASE IF NOT EXISTS market_data;

-- ============================================
-- TICKS - Raw tick data
-- ============================================
CREATE TABLE IF NOT EXISTS market_data.ticks
(
    exchange_token String,
    exchange LowCardinality(String),
    symbol String,
    timestamp DateTime64(3, 'Asia/Kolkata'),
    ltp Decimal64(4),
    bid_price Nullable(Decimal64(4)),
    ask_price Nullable(Decimal64(4)),
    bid_qty Nullable(UInt32),
    ask_qty Nullable(UInt32),
    volume UInt64,
    open_interest Nullable(UInt64),
    received_at DateTime64(3) DEFAULT now64(3)
)
ENGINE = MergeTree()
PARTITION BY toYYYYMMDD(timestamp)
ORDER BY (exchange_token, timestamp)
TTL timestamp + INTERVAL 7 DAY;

-- ============================================
-- CANDLES - OHLCV aggregated data
-- ============================================
CREATE TABLE IF NOT EXISTS market_data.candles
(
    exchange_token String,
    exchange LowCardinality(String),
    symbol String,
    timeframe LowCardinality(String),  -- 1m, 5m, 15m, 1h, 1d
    timestamp DateTime64(3, 'Asia/Kolkata'),
    open Decimal64(4),
    high Decimal64(4),
    low Decimal64(4),
    close Decimal64(4),
    volume UInt64,
    open_interest Nullable(UInt64),
    trade_count Nullable(UInt32),
    vwap Nullable(Decimal64(4)),
    created_at DateTime64(3) DEFAULT now64(3)
)
ENGINE = MergeTree()
PARTITION BY (timeframe, toYYYYMM(timestamp))
ORDER BY (exchange_token, timeframe, timestamp);

-- ============================================
-- SIGNAL AUDIT - Decision audit log
-- ============================================
CREATE TABLE IF NOT EXISTS market_data.signal_audit
(
    signal_id UUID,
    instrument_id String,
    symbol String,
    timeframe LowCardinality(String),
    action LowCardinality(String),  -- LONG, SHORT, NO_TRADE
    confidence Decimal32(4),
    entry_price Nullable(Decimal64(4)),
    stop_loss Nullable(Decimal64(4)),
    target_1 Nullable(Decimal64(4)),
    target_2 Nullable(Decimal64(4)),
    reason_codes Array(String),
    feature_vector String,  -- JSON encoded
    model_version String,
    snapshot_id String,
    generated_at DateTime64(3, 'Asia/Kolkata'),
    
    -- Outcome tracking (updated later)
    outcome Nullable(String),
    actual_pnl_percent Nullable(Decimal32(4)),
    exit_price Nullable(Decimal64(4)),
    closed_at Nullable(DateTime64(3, 'Asia/Kolkata'))
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(generated_at)
ORDER BY (instrument_id, generated_at, signal_id);

-- ============================================
-- BACKTEST TRADES - Trade logs from backtests
-- ============================================
CREATE TABLE IF NOT EXISTS market_data.backtest_trades
(
    backtest_id UUID,
    trade_id UInt64,
    instrument_id String,
    symbol String,
    direction LowCardinality(String),  -- LONG, SHORT
    entry_time DateTime64(3, 'Asia/Kolkata'),
    exit_time DateTime64(3, 'Asia/Kolkata'),
    entry_price Decimal64(4),
    exit_price Decimal64(4),
    quantity UInt32,
    pnl Decimal64(2),
    pnl_percent Decimal32(4),
    exit_reason LowCardinality(String),  -- TARGET, STOP_LOSS, TIMEOUT
    signal_confidence Decimal32(4),
    created_at DateTime64(3) DEFAULT now64(3)
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(entry_time)
ORDER BY (backtest_id, entry_time, trade_id);

-- ============================================
-- PORTFOLIO EQUITY - Equity curve tracking
-- ============================================
CREATE TABLE IF NOT EXISTS market_data.portfolio_equity
(
    user_id UUID,
    mode LowCardinality(String),  -- PAPER, LIVE
    timestamp DateTime64(3, 'Asia/Kolkata'),
    equity Decimal64(2),
    cash Decimal64(2),
    invested Decimal64(2),
    realized_pnl Decimal64(2),
    unrealized_pnl Decimal64(2),
    positions_count UInt16
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (user_id, mode, timestamp);

-- ============================================
-- MATERIALIZED VIEWS for aggregations
-- ============================================

-- 5-minute candles from ticks
CREATE MATERIALIZED VIEW IF NOT EXISTS market_data.candles_5m_mv
TO market_data.candles
AS SELECT
    exchange_token,
    exchange,
    symbol,
    '5m' AS timeframe,
    toStartOfFiveMinutes(timestamp) AS timestamp,
    argMin(ltp, timestamp) AS open,
    max(ltp) AS high,
    min(ltp) AS low,
    argMax(ltp, timestamp) AS close,
    max(volume) - min(volume) AS volume,
    argMax(open_interest, timestamp) AS open_interest,
    count() AS trade_count,
    sum(ltp * volume) / sum(volume) AS vwap
FROM market_data.ticks
GROUP BY exchange_token, exchange, symbol, toStartOfFiveMinutes(timestamp);

-- Daily signal performance summary
CREATE MATERIALIZED VIEW IF NOT EXISTS market_data.signal_daily_stats_mv
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (date, action)
AS SELECT
    toDate(generated_at) AS date,
    action,
    count() AS total_signals,
    countIf(outcome = 'HIT_T1' OR outcome = 'HIT_T2') AS winning_signals,
    avg(confidence) AS avg_confidence,
    avg(actual_pnl_percent) AS avg_pnl
FROM market_data.signal_audit
WHERE outcome IS NOT NULL
GROUP BY toDate(generated_at), action;
