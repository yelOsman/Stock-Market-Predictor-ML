"""
Visualization Module
Generates performance charts from backtesting results
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from datetime import datetime
import sys

# Add config to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
try:
    from config import DB_CONFIG
except ImportError:
    # Fallback for direct execution
    DB_CONFIG = {}

def get_latest_results_file():
    """Find the most recent backtesting results CSV"""
    list_of_files = glob.glob('../results/backtesting_*.csv')
    if not list_of_files:
        return None
    return max(list_of_files, key=os.path.getctime)

def plot_roi_comparison(df):
    """Plot ROI for each ticker"""
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    # Sort by ROI
    df_sorted = df.sort_values('roi', ascending=False)
    
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in df_sorted['roi']]
    
    ax = sns.barplot(x='ticker', y='roi', data=df_sorted, palette=colors)
    plt.title('Return on Investment (ROI) by Ticker', fontsize=15, fontweight='bold')
    plt.ylabel('ROI (%)', fontsize=12)
    plt.xlabel('Ticker', fontsize=12)
    plt.xticks(rotation=45)
    
    # Add values on top of bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig('../plots/roi_comparison.png', dpi=300)
    print("✓ ROI Comparison chart saved to plots/roi_comparison.png")

def plot_win_rates(df):
    """Plot Win Rate for each ticker"""
    plt.figure(figsize=(12, 6))
    
    df_sorted = df.sort_values('win_rate', ascending=False)
    
    ax = sns.barplot(x='ticker', y='win_rate', data=df_sorted, color='#3498db')
    plt.title('Trading Win Rate by Ticker', fontsize=15, fontweight='bold')
    plt.ylabel('Win Rate (%)', fontsize=12)
    plt.xlabel('Ticker', fontsize=12)
    plt.axhline(50, color='red', linestyle='--', alpha=0.5, label='50% Benchmark')
    plt.legend()
    
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig('../plots/win_rates.png', dpi=300)
    print("✓ Win Rates chart saved to plots/win_rates.png")

def plot_final_scores(df):
    """Plot Strategy vs Buy & Hold (Final Score)"""
    plt.figure(figsize=(12, 6))
    
    df_sorted = df.sort_values('final_score', ascending=False)
    
    # 1.0 means equal to Buy & Hold
    colors = ['#f1c40f' if x > 1.0 else '#95a5a6' for x in df_sorted['final_score']]
    
    ax = sns.barplot(x='ticker', y='final_score', data=df_sorted, palette=colors)
    plt.axhline(1.0, color='black', linestyle='-', linewidth=2, label='Buy & Hold Benchmark')
    
    plt.title('Strategy Performance Score (Relative to Buy & Hold)', fontsize=15, fontweight='bold')
    plt.ylabel('Score (Ratio)', fontsize=12)
    plt.xlabel('Ticker', fontsize=12)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('../plots/performance_scores.png', dpi=300)
    print("✓ Performance Scores chart saved to plots/performance_scores.png")

def generate_summary_report(df):
    """Generate a clean text summary of visualizations"""
    print("\n" + "="*50)
    print("📊 VISUALIZATION SUMMARY")
    print("="*50)
    print(f"Total Tickers Analyzed: {len(df)}")
    print(f"Average ROI:           {df['roi'].mean():.2f}%")
    print(f"Average Win Rate:      {df['win_rate'].mean():.2f}%")
    print(f"Best Performer:        {df.loc[df['roi'].idxmax(), 'ticker']} ({df['roi'].max():.2f}%)")
    print(f"Strategy Beat B&H in:  {(df['final_score'] > 1.0).sum()} tickers")
    print("="*50)

def main():
    """Main execution function"""
    # Ensure plots directory exists
    os.makedirs('../plots', exist_ok=True)
    
    results_file = get_latest_results_file()
    
    if not results_file:
        print("❌ No backtesting results found in '../results/'.")
        print("   Please run 'python backtesting.py' first.")
        return
    
    print(f"📂 Found latest results: {results_file}")
    df = pd.read_csv(results_file)
    
    # Set global style
    sns.set_context("talk")
    
    # Generate charts
    plot_roi_comparison(df)
    plot_win_rates(df)
    plot_final_scores(df)
    
    # Summary
    generate_summary_report(df)
    
    print("\n✨ All visualizations updated! Check the 'plots/' folder.")

if __name__ == "__main__":
    main()
