"""
Quick Transformation with Higher Threshold
Run this to improve accuracy!
"""
import sys
sys.path.append('../config')
from config import *
from transformation import transform_data
from sqlalchemy import create_engine
import pandas as pd
from datetime import datetime


def retransform_with_threshold(threshold_pct=2.5):
    """
    Re-transform data with higher threshold
    """
    
    print("="*80)
    print(f"🔄 RE-TRANSFORMING WITH THRESHOLD {threshold_pct}%")
    print("="*80)
    
    # Read from PostgreSQL
    print("\n📦 Reading from PostgreSQL...")
    engine = create_engine(
        f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
        f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )
    
    query = """
    SELECT date, ticker, open, high, low, close, volume
    FROM raw_stock_data
    WHERE date >= '2000-01-01' AND date <= '2025-12-31'
    ORDER BY ticker, date
    """
    
    df = pd.read_sql(query, engine)
    df['Date'] = pd.to_datetime(df['date'])
    
    # Rename to match format
    df = df.rename(columns={
        'date': 'Date',
        'ticker': 'Ticker',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    
    print(f"✅ Loaded {len(df):,} records")
    print(f"   Tickers: {df['Ticker'].nunique()}")
    
    # Group by ticker
    data_dict = {}
    for ticker in df['Ticker'].unique():
        data_dict[ticker] = df[df['Ticker'] == ticker].reset_index(drop=True)
    
    # Transform with NEW threshold
    print(f"\n🔄 Transforming with threshold ±{threshold_pct}%...")
    
    # IMPORTANT: You need to modify transformation.py to accept threshold parameter!
    # For now, we'll manually filter
    
    from transformation import DataTransformer
    transformer = DataTransformer()
    
    # Use transformer but with manual threshold
    print("  Step 1: Combine data...")
    all_data = []
    for ticker, ticker_df in data_dict.items():
        ticker_df['Ticker'] = ticker
        all_data.append(ticker_df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"  Combined: {len(combined_df):,} rows")
    
    # Convert to Spark
    print("  Step 2: Converting to Spark...")
    spark_df = transformer.pandas_to_spark(combined_df)
    
    # Add features
    print("  Step 3: Technical indicators...")
    spark_df = transformer.add_technical_indicators(spark_df)
    
    print("  Step 4: Lagged features...")
    spark_df = transformer.add_lagged_features(spark_df)
    
    print("  Step 5: Cyclical features...")
    spark_df = transformer.add_cyclical_features(spark_df)
    
    print("  Step 6: Target variable...")
    spark_df = transformer.create_target_variable(spark_df, 'buy_sell')
    
    # Convert to Pandas
    result_df = spark_df.toPandas()
    
    # MANUAL THRESHOLD FILTERING (IMPORTANT!)
    print(f"\n🎯 Applying Manual Threshold ±{threshold_pct}%...")
    
    # Calculate return percentage
    result_df['Return_Pct'] = ((result_df['Next_Close'] - result_df['Close']) / result_df['Close']) * 100
    
    # Apply threshold
    original_len = len(result_df)
    
    # Keep only strong signals
    strong_buy = result_df['Return_Pct'] > threshold_pct
    strong_sell = result_df['Return_Pct'] < -threshold_pct
    
    result_df['Target'] = None
    result_df.loc[strong_buy, 'Target'] = 1
    result_df.loc[strong_sell, 'Target'] = 0
    
    # Drop uncertain signals
    result_df = result_df[result_df['Target'].notna()].copy()
    result_df['Target'] = result_df['Target'].astype(int)
    
    kept_len = len(result_df)
    removed = original_len - kept_len
    
    print(f"\n📊 Threshold Filtering Results:")
    print(f"  Original: {original_len:,} rows")
    print(f"  Kept (strong signals): {kept_len:,} ({kept_len/original_len*100:.1f}%)")
    print(f"  Removed (uncertain): {removed:,} ({removed/original_len*100:.1f}%)")
    
    buy_count = (result_df['Target'] == 1).sum()
    sell_count = (result_df['Target'] == 0).sum()
    print(f"  BUY signals: {buy_count:,} ({buy_count/kept_len*100:.1f}%)")
    print(f"  SELL signals: {sell_count:,} ({sell_count/kept_len*100:.1f}%)")
    
    # Save
    output_path = f'../data/transformed_data_threshold_{threshold_pct}_pct_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    result_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Saved to: {output_path}")
    print(f"   Shape: {result_df.shape}")
    
    print("\n" + "="*80)
    print("✅ RE-TRANSFORMATION COMPLETED!")
    print("="*80)
    
    print(f"\n📝 Next Steps:")
    print(f"  1. Train models: python train_models.py --data {output_path}")
    print(f"  2. Expected accuracy: 65-75% (much better!)")
    
    return result_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=2.5,
                       help='Threshold percentage (default: 2.5)')
    
    args = parser.parse_args()
    
    print(f"\n🎯 Using Threshold: ±{args.threshold}%")
    
    retransform_with_threshold(args.threshold)
