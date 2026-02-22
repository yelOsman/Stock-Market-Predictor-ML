"""
Data Loading Module
Loads data into PostgreSQL database
"""
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from typing import Dict, List
import sys
sys.path.append('../config')
from config import DB_CONFIG


class DatabaseLoader:
    """Load data into PostgreSQL database"""
    
    def __init__(self):
        self.conn = None
        self.cursor = None
        
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(
                host=DB_CONFIG['host'],
                port=DB_CONFIG['port'],
                database=DB_CONFIG['database'],
                user=DB_CONFIG['user'],
                password=DB_CONFIG['password']
            )
            self.cursor = self.conn.cursor()
            print("✓ Database connection established")
        except Exception as e:
            print(f"✗ Error connecting to database: {str(e)}")
            raise
    
    def disconnect(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        print("✓ Database connection closed")
    
    def load_historical_data(self, df: pd.DataFrame, data_type: str = 'stock'):
        """
        Load historical stock data into raw_stock_data table
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with columns: Date, Open, High, Low, Close, Volume, Ticker
        data_type : str
            Type of data: 'stock', 'index', or 'economic'
        """
        
        try:
            # Prepare data for insertion
            df = df.copy()
            df['data_type'] = data_type
            
            # Select and rename columns
            columns = ['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 
                      'Volume', 'data_type']
            
            # Handle 'Adj Close' if it exists
            if 'Adj Close' in df.columns:
                df['adj_close'] = df['Adj Close']
                columns.append('adj_close')
            
            df_to_insert = df[columns].copy()
            
            # Convert to list of tuples
            records = df_to_insert.values.tolist()
            
            # SQL insert query
            if 'adj_close' in columns:
                sql = """
                INSERT INTO raw_stock_data 
                (ticker, date, open, high, low, close, volume, data_type, adj_close)
                VALUES %s
                ON CONFLICT (ticker, date) 
                DO UPDATE SET 
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume,
                    adj_close = EXCLUDED.adj_close
                """
            else:
                sql = """
                INSERT INTO raw_stock_data 
                (ticker, date, open, high, low, close, volume, data_type)
                VALUES %s
                ON CONFLICT (ticker, date) 
                DO UPDATE SET 
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume
                """
            
            # Execute batch insert
            execute_values(self.cursor, sql, records)
            self.conn.commit()
            
            print(f"✓ Loaded {len(records)} records for {data_type}")
            
        except Exception as e:
            self.conn.rollback()
            print(f"✗ Error loading historical data: {str(e)}")
            raise
    
    def load_company_financials(self, company_info: Dict[str, Dict]):
        """Load company financial information"""
        
        try:
            records = []
            for ticker, info in company_info.items():
                record = (
                    ticker,
                    pd.Timestamp.now().date(),
                    info.get('market_cap'),
                    info.get('beta'),
                    info.get('pe_ratio'),
                    info.get('pb_ratio')
                )
                records.append(record)
            
            sql = """
            INSERT INTO company_financials 
            (ticker, date, market_cap, beta, pe_ratio, pb_ratio)
            VALUES %s
            ON CONFLICT (ticker, date) 
            DO UPDATE SET 
                market_cap = EXCLUDED.market_cap,
                beta = EXCLUDED.beta,
                pe_ratio = EXCLUDED.pe_ratio,
                pb_ratio = EXCLUDED.pb_ratio
            """
            
            execute_values(self.cursor, sql, records)
            self.conn.commit()
            
            print(f"✓ Loaded financial data for {len(records)} companies")
            
        except Exception as e:
            self.conn.rollback()
            print(f"✗ Error loading company financials: {str(e)}")
            raise
    
    def load_features(self, df: pd.DataFrame):
        """Load engineered features to feature_store table"""
        
        try:
            # Get feature columns (exclude base columns and target)
            exclude_cols = ['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 
                          'Volume', 'Target', 'Next_Open', 'Next_Close', 'Adj Close']
            
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            records = []
            for _, row in df.iterrows():
                ticker = row['Ticker']
                date = row['Date']
                
                for feature_name in feature_cols:
                    value = row[feature_name]
                    if pd.notna(value):  # Skip NaN values
                        records.append((ticker, date, feature_name, float(value)))
            
            sql = """
            INSERT INTO feature_store 
            (ticker, date, feature_name, feature_value)
            VALUES %s
            ON CONFLICT (ticker, date, feature_name) 
            DO UPDATE SET 
                feature_value = EXCLUDED.feature_value
            """
            
            # Batch insert in chunks for large datasets
            chunk_size = 10000
            for i in range(0, len(records), chunk_size):
                chunk = records[i:i+chunk_size]
                execute_values(self.cursor, sql, chunk)
                self.conn.commit()
            
            print(f"✓ Loaded {len(records)} feature records")
            
        except Exception as e:
            self.conn.rollback()
            print(f"✗ Error loading features: {str(e)}")
            raise
    
    def load_predictions(self, df: pd.DataFrame, model_name: str):
        """Load model predictions"""
        
        try:
            records = []
            for _, row in df.iterrows():
                record = (
                    row['Ticker'],
                    row['Date'],
                    model_name,
                    row.get('Prediction_Type'),
                    row.get('Predicted_Price'),
                    row.get('Confidence'),
                    row.get('Actual_Price')
                )
                records.append(record)
            
            sql = """
            INSERT INTO predictions 
            (ticker, prediction_date, model_name, prediction_type, 
             predicted_price, confidence, actual_price)
            VALUES %s
            ON CONFLICT (ticker, prediction_date, model_name) 
            DO UPDATE SET 
                prediction_type = EXCLUDED.prediction_type,
                predicted_price = EXCLUDED.predicted_price,
                confidence = EXCLUDED.confidence,
                actual_price = EXCLUDED.actual_price
            """
            
            execute_values(self.cursor, sql, records)
            self.conn.commit()
            
            print(f"✓ Loaded {len(records)} predictions")
            
        except Exception as e:
            self.conn.rollback()
            print(f"✗ Error loading predictions: {str(e)}")
            raise
    
    def load_trading_results(self, df: pd.DataFrame):
        """Load trading results"""
        
        try:
            records = []
            for _, row in df.iterrows():
                record = (
                    row['Ticker'],
                    row['Date'],
                    row['Action'],
                    row.get('Quantity', 0),
                    row.get('Price', 0),
                    row.get('Transaction_Fee', 0),
                    row.get('Portfolio_Value', 0),
                    row.get('Cash_Balance', 0)
                )
                records.append(record)
            
            sql = """
            INSERT INTO trading_results 
            (ticker, trade_date, action, quantity, price, 
             transaction_fee, portfolio_value, cash_balance)
            VALUES %s
            """
            
            execute_values(self.cursor, sql, records)
            self.conn.commit()
            
            print(f"✓ Loaded {len(records)} trading records")
            
        except Exception as e:
            self.conn.rollback()
            print(f"✗ Error loading trading results: {str(e)}")
            raise
    
    def load_performance_metrics(self, metrics: Dict, ticker: str, 
                                 evaluation_date: str, strategy_name: str):
        """Load performance metrics"""
        
        try:
            record = (
                ticker,
                evaluation_date,
                strategy_name,
                metrics.get('roi'),
                metrics.get('net_profit'),
                metrics.get('win_rate'),
                metrics.get('beta'),
                metrics.get('sharpe_ratio'),
                metrics.get('max_drawdown'),
                metrics.get('total_trades'),
                metrics.get('profitable_trades')
            )
            
            sql = """
            INSERT INTO performance_metrics 
            (ticker, evaluation_date, strategy_name, roi, net_profit, 
             win_rate, beta, sharpe_ratio, max_drawdown, 
             total_trades, profitable_trades)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            self.cursor.execute(sql, record)
            self.conn.commit()
            
            print(f"✓ Loaded performance metrics for {ticker}")
            
        except Exception as e:
            self.conn.rollback()
            print(f"✗ Error loading performance metrics: {str(e)}")
            raise


def load_data(data_dict: Dict, data_type: str = 'historical'):
    """
    Main function to load data into PostgreSQL
    
    Parameters:
    -----------
    data_dict : Dict
        Dictionary containing data to load
    data_type : str
        Type of data: 'historical', 'features', 'predictions', etc.
        
    Example:
    --------
    >>> from extract import extract_data
    >>> data = extract_data(['AAPL'], '2024-01-01', '2024-12-31')
    >>> load_data(data['historical'], 'historical')
    """
    
    loader = DatabaseLoader()
    
    try:
        loader.connect()
        
        if data_type == 'historical':
            for ticker, df in data_dict.items():
                # Determine data type based on ticker
                if ticker.startswith('^'):
                    dtype = 'index'
                elif ticker.endswith('=F'):  
                    dtype = 'index'
                elif ticker in ['GC=F', 'BZ=F', 'DX-Y.NYB']:
                    dtype = 'economic'
                else:
                    dtype = 'stock'
                
                loader.load_historical_data(df, dtype)
        
        elif data_type == 'company_info':
            loader.load_company_financials(data_dict)
        
        print("\n" + "="*80)
        print("Data loading completed successfully!")
        print("="*80)
        
    except Exception as e:
        print(f"\nError during data loading: {str(e)}")
        raise
    
    finally:
        loader.disconnect()


if __name__ == "__main__":
    # Test loading
    from extract import extract_data
    
    test_tickers = ['AAPL']
    data = extract_data(test_tickers, '2024-01-01', '2024-12-31')
    
    load_data(data['historical'], 'historical')
    load_data(data['company_info'], 'company_info')
