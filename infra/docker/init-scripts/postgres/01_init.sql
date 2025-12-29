-- PostgreSQL Initialization Script for Indian AI Trader
-- Creates all necessary tables for the platform

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- INSTRUMENTS
-- ============================================
CREATE TABLE IF NOT EXISTS instruments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    exchange VARCHAR(10) NOT NULL,
    segment VARCHAR(20) NOT NULL,
    trading_symbol VARCHAR(50) NOT NULL,
    name VARCHAR(200),
    exchange_token VARCHAR(20) NOT NULL,
    instrument_token VARCHAR(20),
    lot_size INTEGER DEFAULT 1,
    tick_size DECIMAL(10, 4) DEFAULT 0.05,
    instrument_type VARCHAR(20) DEFAULT 'EQ',
    expiry DATE,
    strike DECIMAL(12, 2),
    option_type VARCHAR(2),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(exchange, trading_symbol)
);

CREATE INDEX idx_instruments_symbol ON instruments(trading_symbol);
CREATE INDEX idx_instruments_token ON instruments(exchange_token);
CREATE INDEX idx_instruments_search ON instruments USING gin(to_tsvector('english', trading_symbol || ' ' || COALESCE(name, '')));

-- ============================================
-- USERS
-- ============================================
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(200),
    is_active BOOLEAN DEFAULT true,
    is_verified BOOLEAN DEFAULT false,
    groww_api_key VARCHAR(100),
    groww_access_token TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- WATCHLISTS
-- ============================================
CREATE TABLE IF NOT EXISTS watchlists (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    is_default BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS watchlist_items (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    watchlist_id UUID REFERENCES watchlists(id) ON DELETE CASCADE,
    instrument_id UUID REFERENCES instruments(id) ON DELETE CASCADE,
    sort_order INTEGER DEFAULT 0,
    added_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(watchlist_id, instrument_id)
);

-- ============================================
-- ORDERS
-- ============================================
CREATE TABLE IF NOT EXISTS orders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    instrument_id UUID REFERENCES instruments(id),
    order_type VARCHAR(20) NOT NULL,  -- MARKET, LIMIT, SL, SL-M
    transaction_type VARCHAR(10) NOT NULL,  -- BUY, SELL
    quantity INTEGER NOT NULL,
    price DECIMAL(12, 2),
    trigger_price DECIMAL(12, 2),
    status VARCHAR(20) DEFAULT 'PENDING',  -- PENDING, OPEN, COMPLETED, CANCELLED, REJECTED
    mode VARCHAR(10) DEFAULT 'PAPER',  -- PAPER, LIVE
    signal_id UUID,
    broker_order_id VARCHAR(50),
    filled_quantity INTEGER DEFAULT 0,
    average_price DECIMAL(12, 2),
    exchange_timestamp TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_orders_user ON orders(user_id);
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_orders_created ON orders(created_at);

-- ============================================
-- POSITIONS
-- ============================================
CREATE TABLE IF NOT EXISTS positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    instrument_id UUID REFERENCES instruments(id),
    quantity INTEGER NOT NULL,
    average_price DECIMAL(12, 2) NOT NULL,
    mode VARCHAR(10) DEFAULT 'PAPER',  -- PAPER, LIVE
    opened_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    closed_at TIMESTAMP WITH TIME ZONE,
    realized_pnl DECIMAL(14, 2) DEFAULT 0,
    
    UNIQUE(user_id, instrument_id, mode)
);

CREATE INDEX idx_positions_user ON positions(user_id);

-- ============================================
-- PORTFOLIO
-- ============================================
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    snapshot_date DATE NOT NULL,
    total_value DECIMAL(16, 2) NOT NULL,
    invested_value DECIMAL(16, 2) NOT NULL,
    cash_balance DECIMAL(16, 2) NOT NULL,
    realized_pnl DECIMAL(14, 2) DEFAULT 0,
    unrealized_pnl DECIMAL(14, 2) DEFAULT 0,
    positions_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(user_id, snapshot_date)
);

-- ============================================
-- BACKTESTS
-- ============================================
CREATE TABLE IF NOT EXISTS backtests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    strategy_config JSONB NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    initial_capital DECIMAL(16, 2) NOT NULL,
    status VARCHAR(20) DEFAULT 'PENDING',  -- PENDING, RUNNING, COMPLETED, FAILED
    progress_percent INTEGER DEFAULT 0,
    
    -- Results
    final_value DECIMAL(16, 2),
    total_return DECIMAL(10, 4),
    sharpe_ratio DECIMAL(8, 4),
    max_drawdown DECIMAL(8, 4),
    win_rate DECIMAL(8, 4),
    total_trades INTEGER,
    
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_backtests_user ON backtests(user_id);
CREATE INDEX idx_backtests_status ON backtests(status);

-- ============================================
-- SIGNALS (cached from signal service)
-- ============================================
CREATE TABLE IF NOT EXISTS signals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    instrument_id UUID REFERENCES instruments(id),
    timeframe VARCHAR(10) NOT NULL,
    action VARCHAR(10) NOT NULL,  -- LONG, SHORT, NO_TRADE
    confidence DECIMAL(5, 4) NOT NULL,
    entry_price DECIMAL(12, 2),
    stop_loss DECIMAL(12, 2),
    target_1 DECIMAL(12, 2),
    target_2 DECIMAL(12, 2),
    reason_codes TEXT[],
    feature_snapshot JSONB,
    model_version VARCHAR(20),
    generated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE,
    
    -- Outcome tracking
    outcome VARCHAR(20),  -- HIT_T1, HIT_T2, HIT_SL, EXPIRED
    actual_pnl_percent DECIMAL(8, 4),
    closed_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_signals_instrument ON signals(instrument_id);
CREATE INDEX idx_signals_generated ON signals(generated_at);
CREATE INDEX idx_signals_action ON signals(action);

-- ============================================
-- FUNCTIONS
-- ============================================

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply trigger to relevant tables
CREATE TRIGGER update_instruments_updated_at BEFORE UPDATE ON instruments
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_watchlists_updated_at BEFORE UPDATE ON watchlists
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_orders_updated_at BEFORE UPDATE ON orders
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================
-- SEED DATA - Default NIFTY 50 Watchlist
-- ============================================
INSERT INTO instruments (exchange, segment, trading_symbol, name, exchange_token, lot_size)
VALUES 
    ('NSE', 'CASH', 'RELIANCE', 'Reliance Industries Ltd', '2885', 1),
    ('NSE', 'CASH', 'TCS', 'Tata Consultancy Services Ltd', '11536', 1),
    ('NSE', 'CASH', 'HDFCBANK', 'HDFC Bank Ltd', '1333', 1),
    ('NSE', 'CASH', 'INFY', 'Infosys Ltd', '1594', 1),
    ('NSE', 'CASH', 'ICICIBANK', 'ICICI Bank Ltd', '4963', 1),
    ('NSE', 'CASH', 'HINDUNILVR', 'Hindustan Unilever Ltd', '1394', 1),
    ('NSE', 'CASH', 'SBIN', 'State Bank of India', '3045', 1),
    ('NSE', 'CASH', 'BHARTIARTL', 'Bharti Airtel Ltd', '10604', 1),
    ('NSE', 'CASH', 'ITC', 'ITC Ltd', '1660', 1),
    ('NSE', 'CASH', 'KOTAKBANK', 'Kotak Mahindra Bank Ltd', '1922', 1),
    ('NSE', 'CASH', 'LT', 'Larsen & Toubro Ltd', '11483', 1),
    ('NSE', 'CASH', 'AXISBANK', 'Axis Bank Ltd', '5900', 1),
    ('NSE', 'CASH', 'ASIANPAINT', 'Asian Paints Ltd', '236', 1),
    ('NSE', 'CASH', 'MARUTI', 'Maruti Suzuki India Ltd', '10999', 1),
    ('NSE', 'CASH', 'SUNPHARMA', 'Sun Pharmaceutical Industries Ltd', '3351', 1),
    ('NSE', 'CASH', 'TITAN', 'Titan Company Ltd', '3506', 1),
    ('NSE', 'CASH', 'BAJFINANCE', 'Bajaj Finance Ltd', '317', 1),
    ('NSE', 'CASH', 'WIPRO', 'Wipro Ltd', '3787', 1),
    ('NSE', 'CASH', 'HCLTECH', 'HCL Technologies Ltd', '1330', 1),
    ('NSE', 'CASH', 'ULTRACEMCO', 'UltraTech Cement Ltd', '11532', 1)
ON CONFLICT (exchange, trading_symbol) DO NOTHING;

COMMIT;
