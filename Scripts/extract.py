"""
Data Extraction Module
Fetches stock data from Yahoo Finance
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict
import sys
sys.path.append('../config')
from config import *


class DataExtractor:
    """Extract stock data from Yahoo Finance"""
    
    def __init__(self, tickers: List[str], start_date: str, end_date: str):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        
    def extract_historical_data(self) -> Dict[str, pd.DataFrame]:
        """Extract historical OHLCV data"""
        data_dict = {}
        
        for ticker in self.tickers:
            try:
                print(f"Fetching historical data for {ticker}...")
                stock = yf.Ticker(ticker)
                hist_data = stock.history(start=self.start_date, end=self.end_date)
                
                if not hist_data.empty:
                    hist_data['Ticker'] = ticker
                    hist_data.reset_index(inplace=True)
                    data_dict[ticker] = hist_data
                    print(f"✓ Successfully fetched {len(hist_data)} records for {ticker}")
                else:
                    print(f"✗ No data found for {ticker}")
                    
            except Exception as e:
                print(f"✗ Error fetching data for {ticker}: {str(e)}")
                
        return data_dict
    
    def extract_company_info(self) -> Dict[str, Dict]:
        """Extract company information and financials"""
        info_dict = {}
        
        for ticker in self.tickers:
            try:
                print(f"Fetching company info for {ticker}...")
                stock = yf.Ticker(ticker)
                info = stock.info
                
                # Extract key financial metrics
                company_data = {
                    'ticker': ticker,
                    'market_cap': info.get('marketCap'),
                    'beta': info.get('beta'),
                    'pe_ratio': info.get('trailingPE'),
                    'pb_ratio': info.get('priceToBook'),
                    'dividend_yield': info.get('dividendYield'),
                    'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
                    'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
                    'avg_volume': info.get('averageVolume'),
                    'sector': info.get('sector'),
                    'industry': info.get('industry')
                }
                
                info_dict[ticker] = company_data
                print(f"✓ Successfully fetched info for {ticker}")
                
            except Exception as e:
                print(f"✗ Error fetching info for {ticker}: {str(e)}")
                
        return info_dict
    
    def extract_financial_statements(self) -> Dict[str, Dict]:
        """Extract financial statements (Income, Balance Sheet, Cash Flow)"""
        financial_dict = {}
        
        for ticker in self.tickers:
            try:
                print(f"Fetching financial statements for {ticker}...")
                stock = yf.Ticker(ticker)
                
                financial_data = {
                    'income_statement': stock.income_stmt,
                    'balance_sheet': stock.balance_sheet,
                    'cash_flow': stock.cashflow,
                    'quarterly_income': stock.quarterly_income_stmt,
                    'quarterly_balance_sheet': stock.quarterly_balance_sheet,
                    'quarterly_cashflow': stock.quarterly_cashflow
                }
                
                financial_dict[ticker] = financial_data
                print(f"✓ Successfully fetched financial statements for {ticker}")
                
            except Exception as e:
                print(f"✗ Error fetching financial statements for {ticker}: {str(e)}")
                
        return financial_dict
    
    def extract_analyst_recommendations(self) -> Dict[str, pd.DataFrame]:
        """Extract analyst recommendations"""
        recommendations_dict = {}
        
        for ticker in self.tickers:
            try:
                print(f"Fetching analyst recommendations for {ticker}...")
                stock = yf.Ticker(ticker)
                recommendations = stock.recommendations
                
                if recommendations is not None and not recommendations.empty:
                    recommendations['Ticker'] = ticker
                    recommendations_dict[ticker] = recommendations
                    print(f"✓ Successfully fetched recommendations for {ticker}")
                else:
                    print(f"✗ No recommendations found for {ticker}")
                    
            except Exception as e:
                print(f"✗ Error fetching recommendations for {ticker}: {str(e)}")
                
        return recommendations_dict
    
    def extract_all_data(self) -> Dict:
        """Extract all available data"""
        print("="*80)
        print("Starting data extraction process...")
        print("="*80)
        
        all_data = {
            'historical': self.extract_historical_data(),
            'company_info': self.extract_company_info(),
            'financials': self.extract_financial_statements(),
            'recommendations': self.extract_analyst_recommendations()
        }
        
        print("="*80)
        print("Data extraction completed!")
        print("="*80)
        
        return all_data


def extract_data(ticker_list: List[str], start_date: str = None, 
                end_date: str = None) -> Dict:
    """
    Main function to extract data from Yahoo Finance
    
    Parameters:
    -----------
    ticker_list : List[str]
        List of ticker symbols
    start_date : str
        Start date in format 'YYYY-MM-DD'
    end_date : str
        End date in format 'YYYY-MM-DD'
        
    Returns:
    --------
    Dict : Dictionary containing all extracted data
    
    Example:
    --------
    >>> tickers = ['AAPL', 'NVDA', '^GSPC']
    >>> data = extract_data(tickers, '2020-01-01', '2025-12-31')
    """
    
    if start_date is None:
        start_date = TRAIN_START
    if end_date is None:
        end_date = PREDICTION_END
        
    extractor = DataExtractor(ticker_list, start_date, end_date)
    return extractor.extract_all_data()


if __name__ == "__main__":
    # Test the extraction
    test_tickers = ['AAPL', 'NVDA', '^GSPC']
    data = extract_data(test_tickers, '2024-01-01', '2024-12-31')
    
    print("\n" + "="*80)
    print("Sample of extracted data:")
    print("="*80)
    
    for ticker, df in data['historical'].items():
        print(f"\n{ticker}:")
        print(df.head())
