-- Database creation
CREATE DATABASE stock_prediction_db;

-- Connect to the database
\c stock_prediction_db;

-- Table for raw stock data
CREATE TABLE IF NOT EXISTS raw_stock_data (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    open DECIMAL(15, 4),
    high DECIMAL(15, 4),
    low DECIMAL(15, 4),
    close DECIMAL(15, 4),
    adj_close DECIMAL(15, 4),
    volume BIGINT,
    data_type VARCHAR(20), -- 'stock', 'index', 'economic'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, date)
);

-- Table for company financial data
CREATE TABLE IF NOT EXISTS company_financials (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    revenue DECIMAL(20, 2),
    net_income DECIMAL(20, 2),
    earnings_per_share DECIMAL(10, 4),
    total_assets DECIMAL(20, 2),
    total_liabilities DECIMAL(20, 2),
    cash_flow DECIMAL(20, 2),
    market_cap DECIMAL(20, 2),
    pe_ratio DECIMAL(10, 4),
    pb_ratio DECIMAL(10, 4),
    beta DECIMAL(10, 4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, date)
);

-- Table for analyst recommendations
CREATE TABLE IF NOT EXISTS analyst_recommendations (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    strong_buy INT,
    buy INT,
    hold INT,
    sell INT,
    strong_sell INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, date)
);

-- Table for engineered features
CREATE TABLE IF NOT EXISTS feature_store (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    feature_name VARCHAR(100) NOT NULL,
    feature_value DECIMAL(15, 6),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, date, feature_name)
);

-- Table for model predictions
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    prediction_date DATE NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    prediction_type VARCHAR(20), -- 'buy', 'sell', 'hold'
    predicted_price DECIMAL(15, 4),
    confidence DECIMAL(5, 4),
    actual_price DECIMAL(15, 4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, prediction_date, model_name)
);

-- Table for trading results
CREATE TABLE IF NOT EXISTS trading_results (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    trade_date DATE NOT NULL,
    action VARCHAR(10), -- 'buy', 'sell', 'hold'
    quantity DECIMAL(15, 4),
    price DECIMAL(15, 4),
    transaction_fee DECIMAL(15, 4),
    portfolio_value DECIMAL(20, 2),
    cash_balance DECIMAL(20, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for performance metrics
CREATE TABLE IF NOT EXISTS performance_metrics (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    evaluation_date DATE NOT NULL,
    strategy_name VARCHAR(100),
    roi DECIMAL(10, 4),
    net_profit DECIMAL(20, 2),
    win_rate DECIMAL(5, 4),
    beta DECIMAL(10, 4),
    sharpe_ratio DECIMAL(10, 4),
    max_drawdown DECIMAL(10, 4),
    total_trades INT,
    profitable_trades INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better query performance
CREATE INDEX idx_raw_stock_ticker_date ON raw_stock_data(ticker, date);
CREATE INDEX idx_financials_ticker_date ON company_financials(ticker, date);
CREATE INDEX idx_predictions_ticker_date ON predictions(ticker, prediction_date);
CREATE INDEX idx_trading_ticker_date ON trading_results(ticker, trade_date);
CREATE INDEX idx_features_ticker_date ON feature_store(ticker, date);

-- Create views for easy data access
CREATE OR REPLACE VIEW latest_stock_prices AS
SELECT DISTINCT ON (ticker) 
    ticker, date, close, volume
FROM raw_stock_data
WHERE data_type = 'stock'
ORDER BY ticker, date DESC;

CREATE OR REPLACE VIEW portfolio_summary AS
SELECT 
    trade_date,
    SUM(portfolio_value) as total_portfolio_value,
    SUM(cash_balance) as total_cash
FROM trading_results
GROUP BY trade_date
ORDER BY trade_date DESC;
