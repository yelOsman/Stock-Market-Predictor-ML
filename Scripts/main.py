"""
Main ETL Pipeline
Orchestrates the complete ETL process
"""
import sys
sys.path.append('../config')
from config import *
from extract import extract_data
from transformation import transform_data
from load import load_data
import pandas as pd
from datetime import datetime


def run_etl_pipeline(tickers: list, start_date: str, end_date: str, 
                     strategy: str = 'buy_sell'):
    """
    Run complete ETL pipeline
    
    Parameters:
    -----------
    tickers : list
        List of ticker symbols
    start_date : str
        Start date for data extraction
    end_date : str
        End date for data extraction
    strategy : str
        Trading strategy ('buy_sell', 'buy_hold_sell', 'next_day_price')
    """
    
    print("="*80)
    print("STOCK PREDICTION ETL PIPELINE")
    print("="*80)
    print(f"Tickers: {tickers}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Strategy: {strategy}")
    print("="*80)
    
    # Step 1: Extract Data
    print("\n[STEP 1/4] EXTRACTING DATA FROM YAHOO FINANCE")
    print("-"*80)
    all_data = extract_data(tickers, start_date, end_date)
    
    # Step 2: Transform Data
    print("\n[STEP 2/4] TRANSFORMING DATA & ENGINEERING FEATURES")
    print("-"*80)
    transformed_data = transform_data(all_data['historical'], strategy)
    
    # Step 3: Load Historical Data
    print("\n[STEP 3/4] LOADING DATA TO POSTGRESQL")
    print("-"*80)
    load_data(all_data['historical'], 'historical')
    load_data(all_data['company_info'], 'company_info')
    
    # Step 4: Save Transformed Data
    print("\n[STEP 4/4] SAVING TRANSFORMED DATA")
    print("-"*80)
    output_path = f'../data/transformed_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    transformed_data.to_csv(output_path, index=False)
    print(f"✓ Transformed data saved to: {output_path}")
    
    print("\n" + "="*80)
    print("ETL PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    return transformed_data


def run_full_pipeline():
    """
    Run complete pipeline with all stocks and indices
    """
    
    # Combine all tickers
    all_tickers = STOCKS + INDICES + ECONOMIC_INDICATORS
    
    print("\n>>> RUNNING FULL ETL PIPELINE FOR ALL PERIODS <<<\n")
    
    # Run for entire historical period
    full_data = run_etl_pipeline(
        tickers=all_tickers,
        start_date=TRAIN_START,
        end_date=PREDICTION_END,
        strategy='buy_sell'
    )
    
    print(f"\nTotal records processed: {len(full_data)}")
    print(f"Features created: {len(full_data.columns)}")
    print(f"Date range: {full_data['Date'].min()} to {full_data['Date'].max()}")
    
    return full_data


def run_incremental_update():
    """
    Run incremental update for latest data (useful for daily updates)
    """
    
    from datetime import datetime, timedelta
    
    # Get last 30 days of data
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    all_tickers = STOCKS + INDICES + ECONOMIC_INDICATORS
    
    print("\n>>> RUNNING INCREMENTAL ETL UPDATE <<<\n")
    
    incremental_data = run_etl_pipeline(
        tickers=all_tickers,
        start_date=start_date,
        end_date=end_date,
        strategy='buy_sell'
    )
    
    return incremental_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ETL Pipeline')
    parser.add_argument('--mode', type=str, default='full', 
                       choices=['full', 'incremental', 'test'],
                       help='Pipeline mode: full, incremental, or test')
    parser.add_argument('--tickers', type=str, nargs='+', 
                       help='List of tickers (optional, overrides config)')
    parser.add_argument('--start', type=str, help='Start date YYYY-MM-DD')
    parser.add_argument('--end', type=str, help='End date YYYY-MM-DD')
    parser.add_argument('--strategy', type=str, default='buy_sell',
                       choices=['buy_sell', 'buy_hold_sell', 'next_day_price'],
                       help='Trading strategy')
    
    args = parser.parse_args()
    
    if args.mode == 'full':
        run_full_pipeline()
        
    elif args.mode == 'incremental':
        run_incremental_update()
        
    elif args.mode == 'test':
        # Test with single stock
        test_tickers = args.tickers or ['AAPL']
        test_start = args.start or '2024-01-01'
        test_end = args.end or '2024-12-31'
        
        run_etl_pipeline(test_tickers, test_start, test_end, args.strategy)
    
    print("\n✓ Pipeline execution completed!")
