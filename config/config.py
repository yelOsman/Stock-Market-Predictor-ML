import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Database Configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'database': os.getenv('DB_NAME', 'stock_prediction_db'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', '')
}

# Stock and Index Selection
STOCKS = ['AAPL', 'NVDA', 'MSFT', 'AVGO', 'META', 'AMZN', 'TSLA']  # Minimum 1 stock required
INDICES = ['^GSPC', 'NQ=F', 'RTY=F', '^DJI']  # Minimum 1 index required

# Additional Economic Indicators
ECONOMIC_INDICATORS = []
# GC=F: Gold, BZ=F: Oil, ^VIX: Volatility Index
# DX-Y.NYB: Dollar Index, ^TNX: 10Y Treasury, ^TYX: 30Y Treasury

# Time Periods
TRAIN_START = '2000-01-01'
TRAIN_END = '2020-01-01'
TEST_START = '2020-01-01'
TEST_END = '2023-01-01'
VALIDATION_START = '2023-01-01'
VALIDATION_END = '2024-01-01'
PREDICTION_START = '2024-01-01'
PREDICTION_END = '2025-12-31'

# Trading Configuration
INITIAL_CAPITAL = 100000  # $100,000
TRANSACTION_FEE = 0.001  # 0.1% fee per transaction

# Model Configuration
RANDOM_SEED = 42
TEST_SIZE = 0.2

# Feature Engineering Parameters
TECHNICAL_INDICATORS = {
    'SMA': [5, 10, 20, 50, 200],
    'EMA': [12, 26],
    'RSI_PERIOD': 14,
    'MACD': {'fast': 12, 'slow': 26, 'signal': 9},
    'BOLLINGER_BANDS': {'window': 20, 'std': 2}
}
