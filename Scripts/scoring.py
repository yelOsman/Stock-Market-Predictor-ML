"""
Scoring Script - Complete Version
Generates predictions and writes to PostgreSQL
Compatible with your existing config, extract, and load files
"""
import pandas as pd
import numpy as np
import joblib
import psycopg2
from psycopg2.extras import execute_values
from sqlalchemy import create_engine
from datetime import datetime
import sys
import os

sys.path.append('../config')
from config import DB_CONFIG, PREDICTION_START, PREDICTION_END


class ModelScorer:
    """
    Load models and generate predictions
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.selectors = {}
        self.feature_names = {}
        
    def load_models(self):
        """
        Load all saved models, scalers, and selectors
        """
        print("="*80)
        print("📦 LOADING MODELS")
        print("="*80)
        
        model_names = ['random_forest', 'xgboost', 'lightgbm']
        
        for model_name in model_names:
            try:
                # Model
                model_path = f'../models/{model_name}.pkl'
                if not os.path.exists(model_path):
                    print(f"⚠️ {model_name.upper()}: Model file not found")
                    continue
                    
                self.models[model_name] = joblib.load(model_path)
                
                # Scaler
                scaler_path = f'../models/scaler_{model_name}.pkl'
                self.scalers[model_name] = joblib.load(scaler_path)
                
                # Selector
                selector_path = f'../models/selector_{model_name}.pkl'
                self.selectors[model_name] = joblib.load(selector_path)
                
                # Features
                features_path = f'../models/features_{model_name}.txt'
                with open(features_path, 'r') as f:
                    self.feature_names[model_name] = [line.strip() for line in f if line.strip()]
                
                print(f"✅ {model_name.upper()}: Loaded successfully ({len(self.feature_names[model_name])} features)")
                
            except Exception as e:
                print(f"❌ {model_name.upper()}: Error loading - {e}")
        
        print(f"\n✅ Loaded {len(self.models)}/{len(model_names)} models")
        
        if len(self.models) == 0:
            raise Exception("❌ No models loaded! Please train models first using train_models.py")
    
    def load_transformed_data(self, csv_path):
        """
        Load pre-transformed data from CSV
        """
        print(f"\n📂 Loading transformed data from CSV...")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"❌ File not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Filter for prediction period (2024-2025)
        df_prediction = df[df['Date'] >= PREDICTION_START].copy()
        
        if len(df_prediction) == 0:
            print(f"⚠️ Warning: No data found for prediction period (>= {PREDICTION_START})")
            print(f"   Using all available data from {df['Date'].min()}")
            df_prediction = df.copy()
        
        print(f"✅ Loaded {len(df_prediction):,} records for prediction")
        print(f"   Date range: {df_prediction['Date'].min()} to {df_prediction['Date'].max()}")
        print(f"   Tickers: {df_prediction['Ticker'].nunique()}")
        
        return df_prediction
    
    def generate_predictions(self, df):
        """
        Generate predictions for all models
        """
        print("\n" + "="*80)
        print("🔮 GENERATING PREDICTIONS")
        print("="*80)
        
        # Get available features from data
        exclude_cols = [
            'Date', 'Ticker', 'Target', 'Next_Open', 'Next_Close',
            'Dividends', 'Stock Splits', 'Return_Pct', 'Adj Close',
            'Price_Change', 'Gain', 'Loss', 'Avg_Gain', 'Avg_Loss', 'RS', 'BB_Std'
        ]
        
        available_features = [
            col for col in df.columns
            if col not in exclude_cols and df[col].dtype in ['float64', 'int64', 'float32', 'int32']
        ]
        
        print(f"\n📊 Available features in data: {len(available_features)}")
        
        all_predictions = []
        
        for model_name, model in self.models.items():
            print(f"\n{'─'*80}")
            print(f"⏳ {model_name.upper()}")
            
            try:
                # Prepare features
                X = df[available_features].copy()
                
                # Fill NaN values
                X = X.fillna(X.mean())
                
                # Feature selection
                selector = self.selectors[model_name]
                X_selected = selector.transform(X)
                
                # Scaling
                scaler = self.scalers[model_name]
                X_scaled = scaler.transform(X_selected)
                
                print(f"  🎯 Features: {len(available_features)} → {X_scaled.shape[1]}")
                
                # Predict
                predictions = model.predict(X_scaled)
                
                # Get confidence scores
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_scaled)
                    confidence = np.max(proba, axis=1)
                else:
                    # For models without predict_proba
                    confidence = np.ones(len(predictions))
                
                # Create predictions DataFrame
                pred_df = pd.DataFrame({
                    'date': df['Date'],
                    'ticker': df['Ticker'],
                    'close_price': df['Close'],
                    'prediction': predictions,
                    'confidence': confidence,
                    'model_name': model_name.replace('_', ' ').title()
                })
                
                all_predictions.append(pred_df)
                
                # Statistics
                buy_count = (predictions == 1).sum()
                sell_count = (predictions == 0).sum()
                
                print(f"  ✅ {len(predictions):,} predictions generated")
                print(f"  📊 BUY:  {buy_count:,} ({buy_count/len(predictions)*100:.1f}%)")
                print(f"  📊 SELL: {sell_count:,} ({sell_count/len(predictions)*100:.1f}%)")
                print(f"  🎯 Avg Confidence: {confidence.mean():.3f}")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                import traceback
                traceback.print_exc()
        
        if all_predictions:
            predictions_df = pd.concat(all_predictions, ignore_index=True)
            print(f"\n✅ Total predictions: {len(predictions_df):,}")
            return predictions_df
        else:
            print(f"\n❌ No predictions generated!")
            return None
    
    def write_to_postgresql(self, predictions_df):
        """
        Write predictions to PostgreSQL
        """
        print("\n" + "="*80)
        print("💾 WRITING PREDICTIONS TO POSTGRESQL")
        print("="*80)
        
        try:
            # Connect to database using config
            conn = psycopg2.connect(
                host=DB_CONFIG['host'],
                port=DB_CONFIG['port'],
                database=DB_CONFIG['database'],
                user=DB_CONFIG['user'],
                password=DB_CONFIG['password']
            )
            cursor = conn.cursor()
            
            # Create table if not exists
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS model_predictions_final (
                id SERIAL PRIMARY KEY,
                date DATE NOT NULL,
                ticker VARCHAR(10) NOT NULL,
                close_price NUMERIC(15,4),
                prediction INTEGER,
                confidence NUMERIC(10,6),
                model_name VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date, ticker, model_name)
            );
            
            CREATE INDEX IF NOT EXISTS idx_predictions_date_ticker 
            ON model_predictions_final(date, ticker);
            
            CREATE INDEX IF NOT EXISTS idx_predictions_model 
            ON model_predictions_final(model_name);
            """
            
            cursor.execute(create_table_sql)
            conn.commit()
            
            print(f"✅ Table 'model_predictions_final' ready")
            
            # Prepare data for insertion
            predictions_df['created_at'] = datetime.now()
            
            records = predictions_df[
                ['date', 'ticker', 'close_price', 'prediction', 'confidence', 'model_name', 'created_at']
            ].values.tolist()
            
            # Insert with ON CONFLICT UPDATE
            insert_sql = """
            INSERT INTO model_predictions_final 
            (date, ticker, close_price, prediction, confidence, model_name, created_at)
            VALUES %s
            ON CONFLICT (date, ticker, model_name) 
            DO UPDATE SET 
                close_price = EXCLUDED.close_price,
                prediction = EXCLUDED.prediction,
                confidence = EXCLUDED.confidence,
                created_at = EXCLUDED.created_at
            """
            
            execute_values(cursor, insert_sql, records)
            conn.commit()
            
            print(f"✅ {len(records):,} predictions written to database")
            
            # Verify
            cursor.execute("""
                SELECT model_name, COUNT(*) as count 
                FROM model_predictions_final 
                GROUP BY model_name
                ORDER BY model_name
            """)
            
            results = cursor.fetchall()
            print(f"\n📊 Database verification:")
            for model_name, count in results:
                print(f"  {model_name}: {count:,} predictions")
            
            # Close connection
            cursor.close()
            conn.close()
            
            print(f"\n✅ Predictions successfully saved to PostgreSQL!")
            
        except Exception as e:
            print(f"❌ Error writing to database: {e}")
            import traceback
            traceback.print_exc()
            raise


def run_scoring(transformed_data_path):
    """
    Main scoring function
    
    Parameters:
    -----------
    transformed_data_path : str
        Path to transformed CSV file (output from main.py or transformation.py)
    """
    
    print("="*80)
    print("🎯 STOCK PREDICTION SCORING PIPELINE")
    print("="*80)
    print(f"\n📋 Configuration:")
    print(f"  Data file: {transformed_data_path}")
    print(f"  Database: {DB_CONFIG['database']}")
    print(f"  Host: {DB_CONFIG['host']}:{DB_CONFIG['port']}")
    
    # Initialize scorer
    scorer = ModelScorer()
    
    # Load models
    scorer.load_models()
    
    # Load data
    df = scorer.load_transformed_data(transformed_data_path)
    
    # Generate predictions
    predictions_df = scorer.generate_predictions(df)
    
    if predictions_df is not None:
        # Write to PostgreSQL
        scorer.write_to_postgresql(predictions_df)
        
        print("\n" + "="*80)
        print("✅ SCORING COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        # Save to CSV as backup
        output_csv = f'../data/predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        predictions_df.to_csv(output_csv, index=False)
        print(f"\n💾 Backup saved to: {output_csv}")
        
        return predictions_df
    else:
        print("\n❌ Scoring failed!")
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Predictions (Scoring)')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to transformed CSV file')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.data):
        print(f"❌ Error: File not found: {args.data}")
        print(f"\nAvailable files in ../data/:")
        if os.path.exists('../data'):
            for f in os.listdir('../data'):
                if f.endswith('.csv'):
                    print(f"  - {f}")
        sys.exit(1)
    
    predictions = run_scoring(args.data)
    
    if predictions is not None:
        print(f"\n💾 Predictions generated successfully!")
        print(f"\nSample predictions:")
        print(predictions.head(10).to_string(index=False))
