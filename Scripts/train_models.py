"""
Model Training Script
Trains Random Forest, XGBoost, and LightGBM models
Saves models and features for scoring
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
import xgboost as xgb
from lightgbm import LGBMClassifier
import joblib
from datetime import datetime
import sys
sys.path.append('../config')
from config import *


class StockPredictor:
    """
    Complete stock prediction model trainer
    """
    
    def __init__(self, n_features=35):
        self.n_features = n_features
        self.models = {}
        self.scalers = {}
        self.selectors = {}
        self.feature_names = {}
        
    def prepare_data(self, df):
        """
        Prepare data with proper time-based splits
        """
        print("\n" + "="*80)
        print("📊 DATA PREPARATION")
        print("="*80)
        
        # Ensure Date is datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Drop rows with NaN target (filtered by threshold in transformation)
        original_len = len(df)
        df = df.dropna(subset=['Target']).copy()
        if len(df) < original_len:
            print(f"  Filtered out {original_len - len(df):,} uncertain signals (Target=NaN)")
        
        # Feature columns
        exclude_cols = [
            'Date', 'Ticker', 'Target', 'Next_Open', 'Next_Close',
            'Dividends', 'Stock Splits', 'Return_Pct', 'Adj Close',
            'Price_Change', 'Gain', 'Loss', 'Avg_Gain', 'Avg_Loss', 'RS',
            'BB_Std'  # Intermediate columns from transformation
        ]
        
        self.feature_columns = [
            col for col in df.columns
            if col not in exclude_cols and df[col].dtype in ['float64', 'int64']
        ]
        
        print(f"\n📊 Original features: {len(self.feature_columns)}")
        
        # Time-based splits (as per assignment)
        print(f"\n📅 Time-based splits:")
        
        df_train = df[df['Date'] < TRAIN_END].copy()
        print(f"  Train:  {df_train['Date'].min()} to {df_train['Date'].max()} ({len(df_train):,} samples)")
        
        df_test = df[(df['Date'] >= TEST_START) & (df['Date'] < TEST_END)].copy()
        print(f"  Test:   {df_test['Date'].min()} to {df_test['Date'].max()} ({len(df_test):,} samples)")
        
        df_val = df[(df['Date'] >= VALIDATION_START) & (df['Date'] < VALIDATION_END)].copy()
        print(f"  Val:    {df_val['Date'].min()} to {df_val['Date'].max()} ({len(df_val):,} samples)")
        
        # Extract features and target
        X_train = df_train[self.feature_columns].fillna(df_train[self.feature_columns].mean())
        y_train = df_train['Target']
        
        X_test = df_test[self.feature_columns].fillna(X_train.mean())
        y_test = df_test['Target']
        
        X_val = df_val[self.feature_columns].fillna(X_train.mean())
        y_val = df_val['Target']
        
        # Class distribution
        print(f"\n📊 Class Distribution:")
        print(f"  Train: {dict(zip(*np.unique(y_train, return_counts=True)))}")
        print(f"  Test:  {dict(zip(*np.unique(y_test, return_counts=True)))}")
        print(f"  Val:   {dict(zip(*np.unique(y_val, return_counts=True)))}")
        
        return X_train, X_test, X_val, y_train, y_test, y_val
    
    def train_model(self, model_name, X_train, y_train, X_val, y_val):
        """
        Train a single model with hyperparameter tuning
        """
        print(f"\n{'='*80}")
        print(f"🚀 TRAINING {model_name.upper()}")
        print(f"{'='*80}")
        
        # Feature selection
        print(f"\n🎯 Feature Selection: {len(self.feature_columns)} → {self.n_features}")
        selector = SelectKBest(mutual_info_classif, k=self.n_features)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_val_selected = selector.transform(X_val)
        
        selected_indices = selector.get_support(indices=True)
        selected_features = [self.feature_columns[i] for i in selected_indices]
        
        # Scaling
        print(f"⚖️ Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_val_scaled = scaler.transform(X_val_selected)
        
        # Model configuration
        if model_name == 'random_forest':
            base_model = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1)
            param_dist = {
                'n_estimators': [100, 150, 200],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [5, 10],
                'min_samples_leaf': [2, 4],
                'class_weight': ['balanced']
            }
        
        elif model_name == 'xgboost':
            base_model = xgb.XGBClassifier(
                random_state=RANDOM_SEED,
                n_jobs=-1,
                eval_metric='logloss'
            )
            param_dist = {
                'n_estimators': [100, 150, 200],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.7, 0.8],
                'colsample_bytree': [0.7, 0.8]
            }
        
        elif model_name == 'lightgbm':
            base_model = LGBMClassifier(
                random_state=RANDOM_SEED,
                n_jobs=-1,
                verbose=-1
            )
            param_dist = {
                'n_estimators': [100, 150, 200],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.15],
                'num_leaves': [31, 50]
            }
        
        # Hyperparameter tuning
        print(f"🔍 RandomizedSearchCV (12 iterations)...")
        search = RandomizedSearchCV(
            base_model,
            param_dist,
            n_iter=12,
            cv=3,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1,
            random_state=RANDOM_SEED
        )
        
        search.fit(X_train_scaled, y_train)
        model = search.best_estimator_
        
        print(f"\n✅ Best params: {search.best_params_}")
        print(f"✅ Best CV F1: {search.best_score_:.4f}")
        
        # Validation metrics
        val_pred = model.predict(X_val_scaled)
        val_acc = accuracy_score(y_val, val_pred)
        val_f1 = f1_score(y_val, val_pred, average='weighted')
        
        print(f"\n📊 Validation Metrics:")
        print(f"  Accuracy: {val_acc:.4f}")
        print(f"  F1-Score: {val_f1:.4f}")
        
        # Save
        self.models[model_name] = model
        self.scalers[model_name] = scaler
        self.selectors[model_name] = selector
        self.feature_names[model_name] = selected_features
        
        return model, val_f1
    
    def evaluate_all(self, X_test, y_test):
        """
        Evaluate all trained models on test set
        """
        print(f"\n{'='*80}")
        print(f"📊 TEST SET EVALUATION")
        print(f"{'='*80}")
        
        results = {}
        
        for model_name, model in self.models.items():
            print(f"\n{model_name.upper()}:")
            
            # Transform test data
            X_test_selected = self.selectors[model_name].transform(X_test)
            X_test_scaled = self.scalers[model_name].transform(X_test_selected)
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            
            # Metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            print(f"  Accuracy:  {acc:.4f}")
            print(f"  Precision: {prec:.4f}")
            print(f"  Recall:    {rec:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            
            results[model_name] = {
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1_score': f1
            }
        
        return results
    
    def save_models(self):
        """
        Save all models, scalers, selectors, and feature names
        """
        print(f"\n💾 Saving models...")
        
        for model_name in self.models.keys():
            # Model
            model_path = f'../models/{model_name}.pkl'
            joblib.dump(self.models[model_name], model_path)
            
            # Scaler
            scaler_path = f'../models/scaler_{model_name}.pkl'
            joblib.dump(self.scalers[model_name], scaler_path)
            
            # Selector
            selector_path = f'../models/selector_{model_name}.pkl'
            joblib.dump(self.selectors[model_name], selector_path)
            
            # Features
            features_path = f'../models/features_{model_name}.txt'
            with open(features_path, 'w') as f:
                f.write('\n'.join(self.feature_names[model_name]))
            
            print(f"  ✅ {model_name}: model, scaler, selector, features saved")
        
        print("\n✅ All models saved successfully!")


def train_all_models(data_path):
    """
    Main training function
    """
    print("="*80)
    print("🚀 STOCK PREDICTION MODEL TRAINING")
    print("="*80)
    print(f"\n⚙️ Configuration:")
    print(f"  Data: {data_path}")
    print(f"  Random Seed: {RANDOM_SEED}")
    print(f"  Train Period: {TRAIN_START} to {TRAIN_END}")
    print(f"  Test Period: {TEST_START} to {TEST_END}")
    print(f"  Validation Period: {VALIDATION_START} to {VALIDATION_END}")
    
    # Load data
    print(f"\n📂 Loading data...")
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df):,} records")
    
    # Initialize trainer
    trainer = StockPredictor(n_features=35)
    
    # Prepare data
    X_train, X_test, X_val, y_train, y_test, y_val = trainer.prepare_data(df)
    
    # Train all models
    for model_name in ['random_forest', 'xgboost', 'lightgbm']:
        trainer.train_model(model_name, X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    results = trainer.evaluate_all(X_test, y_test)
    
    # Save models
    trainer.save_models()
    
    # Summary
    print(f"\n{'='*80}")
    print(f" TRAINING SUMMARY")
    print(f"{'='*80}")
    
    results_df = pd.DataFrame(results).T
    print(results_df.to_string())
    
    best_model = results_df['f1_score'].idxmax()
    print(f"\n🎯 Best Model: {best_model.upper()}")
    print(f"   F1-Score: {results_df.loc[best_model, 'f1_score']:.4f}")
    
    print(f"\n✅ Training complete! Models saved in ../models/")
    
    return trainer, results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Stock Prediction Models')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to transformed CSV file')
    
    args = parser.parse_args()
    
    trainer, results = train_all_models(args.data)
    
    
