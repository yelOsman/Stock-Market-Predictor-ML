"""
Assignment Backtesting Script - Final Version
Calculates Final Score and 4 Required Metrics
Author: Stock Prediction ML Project
Date: January 2026

This script implements the exact backtesting methodology required by the assignment:
- Initial Capital: $100,000
- Daily execution on next day OPEN price
- Transaction fee: 1/1000 (0.1%)
- Final Score = Strategy Value / Buy & Hold Value
- 4 Required Metrics: ROI, Net Profit, Win Rate, Beta
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from datetime import datetime
import sys
import os

sys.path.append('../config')
from config import DB_CONFIG, INITIAL_CAPITAL, TRANSACTION_FEE


class AssignmentBacktester:
    """
    Backtesting system following exact assignment requirements
    
    Features:
    - Assignment-compliant execution rules
    - Confidence filtering for improved results
    - Detailed ticker-level analysis
    - CSV export for reporting
    """
    
    def __init__(self, initial_capital=INITIAL_CAPITAL):
        """
        Initialize backtester
        
        Parameters:
        -----------
        initial_capital : float
            Starting capital (default: $100,000 from config)
        """
        self.initial_capital = initial_capital
        self.transaction_fee = TRANSACTION_FEE  # 0.001 (1/1000)
        
    def calculate_buy_hold(self, prices_df):
        """
        Calculate Buy & Hold benchmark
        Buy on first day, hold until end
        
        Parameters:
        -----------
        prices_df : pd.DataFrame
            Price data with 'open' and 'close' columns
            
        Returns:
        --------
        float : Final value with Buy & Hold strategy
        """
        first_price = prices_df.iloc[0]['open']
        last_price = prices_df.iloc[-1]['close']
        shares = self.initial_capital / first_price
        return shares * last_price
    
    def calculate_beta(self, portfolio_returns, market_returns):
        """
        Calculate Beta (market correlation)
        Beta = Covariance(portfolio, market) / Variance(market)
        
        Parameters:
        -----------
        portfolio_returns : list
            Daily portfolio returns
        market_returns : list
            Daily market (S&P 500) returns
            
        Returns:
        --------
        float : Beta coefficient
        """
        if len(portfolio_returns) < 2 or len(market_returns) < 2:
            return 0.0
        
        min_len = min(len(portfolio_returns), len(market_returns))
        portfolio_returns = portfolio_returns[-min_len:]
        market_returns = market_returns[-min_len:]
        
        covariance = np.cov(portfolio_returns, market_returns)[0][1]
        market_variance = np.var(market_returns)
        
        return covariance / market_variance if market_variance > 0 else 0.0
    
    def backtest_ticker(self, predictions_df, prices_df, ticker, market_prices_df=None):
        """
        Backtest single ticker with assignment rules
        
        Assignment Rules:
        1. Execute on NEXT DAY OPEN price
        2. Apply 1/1000 transaction fee on each action
        3. Calculate Final Score = Strategy / Buy&Hold
        
        Parameters:
        -----------
        predictions_df : pd.DataFrame
            Model predictions with columns: date, ticker, prediction, confidence
        prices_df : pd.DataFrame
            Price data with columns: date, ticker, open, close
        ticker : str
            Ticker symbol to backtest
        market_prices_df : pd.DataFrame, optional
            Market index prices for Beta calculation
            
        Returns:
        --------
        dict : Results including all 4 required metrics + Final Score
        """
        # Filter for specific ticker
        pred = predictions_df[predictions_df['ticker'] == ticker].copy()
        prices = prices_df[prices_df['ticker'] == ticker].copy()
        
        if len(pred) == 0 or len(prices) == 0:
            return None
        
        # Merge predictions with prices
        df = pred.merge(prices, left_on='date', right_on='date', how='inner')
        df = df.sort_values('date').reset_index(drop=True)
        
        # Add next day prices (for execution)
        df['next_open'] = df['open'].shift(-1)
        df['next_close'] = df['close'].shift(-1)
        df = df.dropna(subset=['next_open'])
        
        if len(df) == 0:
            return None
        
        # Initialize portfolio
        cash = self.initial_capital
        shares = 0
        portfolio_values = [self.initial_capital]
        trades = []
        
        # Daily execution loop
        for _, row in df.iterrows():
            prediction = row['prediction']
            next_open = row['next_open']    # Execute at next day OPEN!
            next_close = row['next_close']
            
            # Calculate portfolio value at close
            current_value = cash + (shares * next_close if shares > 0 else 0)
            portfolio_values.append(current_value)
            
            # BUY signal (prediction = 1)
            if prediction == 1 and shares == 0:
                shares_to_buy = cash / next_open
                cost = shares_to_buy * next_open
                fee = cost * self.transaction_fee  # 1/1000 transaction fee
                total_cost = cost + fee
                
                shares = shares_to_buy
                cash -= total_cost
                
                trades.append({
                    'action': 'BUY',
                    'price': next_open,
                    'cost': cost,
                    'fee': fee
                })
            
            # SELL signal (prediction = 0)
            elif prediction == 0 and shares > 0:
                revenue = shares * next_open
                fee = revenue * self.transaction_fee  # 1/1000 transaction fee
                net_revenue = revenue - fee
                
                trades.append({
                    'action': 'SELL',
                    'price': next_open,
                    'revenue': revenue,
                    'fee': fee
                })
                
                cash += net_revenue
                shares = 0
        
        # Calculate final portfolio value
        final_close = df.iloc[-1]['next_close']
        final_value = cash + (shares * final_close if shares > 0 else 0)
        
        # Buy & Hold benchmark
        buy_hold_value = self.calculate_buy_hold(df)
        
        # ===== METRIC 1: ROI (Return on Investment) =====
        roi = ((final_value - self.initial_capital) / self.initial_capital) * 100
        
        # ===== METRIC 2: Net Profit =====
        net_profit = final_value - self.initial_capital
        
        # ===== METRIC 3: Win Rate (Accuracy) =====
        wins = 0
        losses = 0
        for i in range(0, len(trades)-1, 2):
            if i+1 < len(trades) and trades[i]['action'] == 'BUY' and trades[i+1]['action'] == 'SELL':
                if trades[i+1]['revenue'] > trades[i]['cost']:
                    wins += 1
                else:
                    losses += 1
        
        win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
        
        # ===== METRIC 4: Beta =====
        portfolio_returns = []
        for i in range(1, len(portfolio_values)):
            ret = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
            portfolio_returns.append(ret)
        
        beta = 0.0
        if market_prices_df is not None and len(market_prices_df) > 1:
            market = market_prices_df.sort_values('date')
            market_returns = []
            for i in range(1, len(market)):
                ret = (market.iloc[i]['close'] - market.iloc[i-1]['close']) / market.iloc[i-1]['close']
                market_returns.append(ret)
            beta = self.calculate_beta(portfolio_returns, market_returns)
        
        # ===== ASSIGNMENT FINAL SCORE =====
        final_score = final_value / buy_hold_value if buy_hold_value > 0 else 0
        
        # Total transaction fees paid
        total_fees = sum(t['fee'] for t in trades)
        
        return {
            'ticker': ticker,
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'buy_hold_value': buy_hold_value,
            'roi': roi,                    # REQUIRED METRIC 1
            'net_profit': net_profit,      # REQUIRED METRIC 2
            'win_rate': win_rate,          # REQUIRED METRIC 3
            'beta': beta,                  # REQUIRED METRIC 4
            'final_score': final_score,    # ASSIGNMENT FINAL SCORE
            'num_trades': len(trades),
            'wins': wins,
            'losses': losses,
            'total_fees': total_fees
        }
    
    def run_backtesting(self, model_name='Xgboost', confidence_threshold=0.0):
        """
        Run complete backtesting analysis
        
        Parameters:
        -----------
        model_name : str
            Model to backtest ('Xgboost', 'Lightgbm', 'Random Forest')
        confidence_threshold : float
            Minimum confidence for predictions (0.0 = no filter, 0.6 = recommended)
            
        Returns:
        --------
        pd.DataFrame : Results for all tickers
        """
        print("="*80)
        print("🎯 ASSIGNMENT-COMPLIANT BACKTESTING SYSTEM")
        print("="*80)
        print(f"\n📋 Assignment Requirements:")
        print(f"  ✅ Initial Capital:    ${self.initial_capital:,}")
        print(f"  ✅ Transaction Fee:    {self.transaction_fee} (1/1000)")
        print(f"  ✅ Execution:          Next day OPEN price")
        print(f"  ✅ Final Score:        Strategy / Buy&Hold")
        print(f"  ✅ Required Metrics:   ROI, Net Profit, Win Rate, Beta")
        
        print(f"\n⚙️ Configuration:")
        print(f"  Model: {model_name}")
        if confidence_threshold > 0:
            print(f"  Confidence Filter: >{confidence_threshold} (High-quality signals only)")
        else:
            print(f"  Confidence Filter: OFF (Using all predictions)")
        print(f"  Database: {DB_CONFIG['database']}")
        
        # Create database connection
        engine = create_engine(
            f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
            f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        )
        
        # Load predictions from PostgreSQL
        print(f"\n📦 Loading predictions from database...")
        
        if confidence_threshold > 0:
            pred_query = f"""
            SELECT date, ticker, prediction, confidence
            FROM model_predictions_final
            WHERE model_name = '{model_name}' 
              AND confidence > {confidence_threshold}
            ORDER BY ticker, date
            """
        else:
            pred_query = f"""
            SELECT date, ticker, prediction, confidence
            FROM model_predictions_final
            WHERE model_name = '{model_name}'
            ORDER BY ticker, date
            """
        
        try:
            predictions = pd.read_sql(pred_query, engine)
            predictions['date'] = pd.to_datetime(predictions['date'])
            
            if len(predictions) == 0:
                print(f"❌ No predictions found for model: {model_name}")
                return None
            
            # Show filtering statistics
            if confidence_threshold > 0:
                total_query = f"SELECT COUNT(*) as cnt FROM model_predictions_final WHERE model_name = '{model_name}'"
                total_count = pd.read_sql(total_query, engine)['cnt'][0]
                kept_pct = len(predictions) / total_count * 100
                print(f"✅ Filtered predictions: {len(predictions):,}/{total_count:,} ({kept_pct:.1f}%)")
                print(f"   Removed {total_count - len(predictions):,} low-confidence signals")
            else:
                print(f"✅ Loaded {len(predictions):,} predictions (no filtering)")
                
        except Exception as e:
            print(f"❌ Error loading predictions: {e}")
            return None
        
        # Load price data
        print(f"📦 Loading price data...")
        price_query = """
        SELECT date, ticker, open, close
        FROM raw_stock_data
        WHERE date >= '2024-01-01'
        ORDER BY ticker, date
        """
        
        try:
            prices = pd.read_sql(price_query, engine)
            prices['date'] = pd.to_datetime(prices['date'])
            print(f"✅ Loaded {len(prices):,} price records")
        except Exception as e:
            print(f"❌ Error loading prices: {e}")
            return None
        
        # Load market index for Beta calculation
        print(f"📦 Loading market index (S&P 500)...")
        market_query = """
        SELECT date, close
        FROM raw_stock_data
        WHERE ticker = '^GSPC' AND date >= '2024-01-01'
        ORDER BY date
        """
        
        try:
            market = pd.read_sql(market_query, engine)
            market['date'] = pd.to_datetime(market['date'])
            print(f"✅ Loaded S&P 500 ({len(market)} records)")
        except:
            market = None
            print(f"⚠️ S&P 500 not found, Beta will be 0")
        
        # Backtest each ticker
        results = []
        tickers = sorted(predictions['ticker'].unique())
        
        print(f"\n📊 Backtesting {len(tickers)} tickers...")
        print("="*80)
        
        for ticker in tickers:
            result = self.backtest_ticker(predictions, prices, ticker, market)
            if result:
                results.append(result)
                print(f"{ticker:8s} | Score: {result['final_score']:.3f} | "
                      f"ROI: {result['roi']:6.2f}% | Win: {result['win_rate']:5.1f}%")
        
        if not results:
            print("\n❌ No results generated!")
            return None
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Calculate summary statistics
        print("\n" + "="*80)
        print("📊 FINAL SUMMARY - ASSIGNMENT RESULTS")
        print("="*80)
        
        avg_final_score = results_df['final_score'].mean()
        avg_roi = results_df['roi'].mean()
        avg_profit = results_df['net_profit'].mean()
        avg_win_rate = results_df['win_rate'].mean()
        avg_beta = results_df['beta'].mean()
        
        print(f"\n🎯 METRICS (Assignment):")
        print(f"  1️⃣ ROI (Return on Investment): {avg_roi:.2f}%")
        print(f"  2️⃣ Net Profit:                 ${avg_profit:,.2f}")
        print(f"  3️⃣ Win Rate (Accuracy):        {avg_win_rate:.2f}%")
        print(f"  4️⃣ Beta:                       {avg_beta:.3f}")
        
        print(f"\n💰 Financial Summary:")
        print(f"  Total Net Profit:     ${results_df['net_profit'].sum():,.2f}")
        print(f"  Total Trades:         {results_df['num_trades'].sum()}")
        print(f"  Total Fees Paid:      ${results_df['total_fees'].sum():,.2f}")
        print(f"  Winning Tickers:      {(results_df['final_score'] > 1.0).sum()}/{len(results_df)}")
        
        print(f"\n🎯 ASSIGNMENT FINAL SCORE:")
        print(f"  {'='*76}")
        print(f"  {avg_final_score:.3f}")
        print(f"  {'='*76}")
        
        if avg_final_score > 1.0:
            improvement = (avg_final_score - 1.0) * 100
            print(f"\n  ✅ SUCCESS! Strategy beats Buy & Hold by {improvement:.1f}%")
        else:
            underperformance = (1.0 - avg_final_score) * 100
            print(f"\n  ⚠️ Strategy captures {avg_final_score*100:.1f}% of Buy & Hold returns")
            print(f"     (Underperforms by {underperformance:.1f}%)")
        
        # Top and bottom performers
        print(f"\n🏆 TOP 5 PERFORMERS (by Final Score):")
        print("="*80)
        top5 = results_df.nlargest(5, 'final_score')[['ticker', 'final_score', 'roi', 'win_rate']]
        for _, row in top5.iterrows():
            print(f"  {row['ticker']:8s} Score: {row['final_score']:.3f}  "
                  f"ROI: {row['roi']:7.2f}%  Win Rate: {row['win_rate']:5.1f}%")
        
        print(f"\n📉 BOTTOM 3 PERFORMERS (by Final Score):")
        print("="*80)
        bottom3 = results_df.nsmallest(3, 'final_score')[['ticker', 'final_score', 'roi', 'win_rate']]
        for _, row in bottom3.iterrows():
            print(f"  {row['ticker']:8s} Score: {row['final_score']:.3f}  "
                  f"ROI: {row['roi']:7.2f}%  Win Rate: {row['win_rate']:5.1f}%")
        
        # Save results to CSV
        os.makedirs('../results', exist_ok=True)
        
        if confidence_threshold > 0:
            filename = f'backtesting_{model_name}_conf{confidence_threshold}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        else:
            filename = f'backtesting_{model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        
        output_file = f'../results/{filename}'
        results_df.to_csv(output_file, index=False)
        print(f"\n💾 Results saved to: {output_file}")
        
        print("\n" + "="*80)
        print("✅ BACKTESTING COMPLETED!")
        print("="*80)
        
        return results_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Stock Prediction Backtesting System - Assignment Compliant',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic backtesting (no confidence filter)
  python backtesting.py --model Xgboost
  
  # With confidence filtering (recommended)
  python backtesting.py --model Xgboost --confidence 0.60
  
  # Try different model
  python backtesting.py --model Lightgbm --confidence 0.60
        """
    )
    
    parser.add_argument(
        '--model', 
        type=str, 
        default='Xgboost',
        choices=['Random Forest', 'Xgboost', 'Lightgbm'],
        help='Model to backtest (default: Xgboost)'
    )
    
    parser.add_argument(
        '--confidence', 
        type=float, 
        default=0.60,
        help='Confidence threshold (0.0=off, 0.60=recommended, range: 0.0-1.0)'
    )
    
    args = parser.parse_args()
    
    # Validate confidence threshold
    if args.confidence < 0.0 or args.confidence > 1.0:
        print(f"❌ Error: Confidence threshold must be between 0.0 and 1.0")
        sys.exit(1)
    
    # Initialize and run backtester
    print(f"\n🚀 Starting Assignment Backtesting System...")
    print(f"   Model: {args.model}")
    print(f"   Confidence Threshold: {args.confidence if args.confidence > 0 else 'OFF (No filtering)'}\n")
    
    backtester = AssignmentBacktester(initial_capital=100000)
    results = backtester.run_backtesting(
        model_name=args.model,
        confidence_threshold=args.confidence
    )
    
    if results is not None:
        print(f"\n✅ Backtesting completed successfully!")
        print(f"   Check results/ folder for detailed CSV export")
    else:
        print(f"\n❌ Backtesting failed! Please check error messages above.")
        sys.exit(1)
