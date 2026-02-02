import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')

def load_data():
    curr_path = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(curr_path, 'data')
    x_train_path = os.path.join(data_dir, 'x_train.csv')
    y_train_path = os.path.join(data_dir, 'y_train.csv')
    x_test_path = os.path.join(data_dir, 'x_test.csv')
    
    print("Loading data...")
    X_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path)
    X_test = pd.read_csv(x_test_path)
    
    if 'ID' in X_train.columns: X_train = X_train.set_index('ID')
    if 'ID' in y_train.columns: y_train = y_train.set_index('ID')
    if 'ID' in X_test.columns: X_test = X_test.set_index('ID')
        
    df_train = X_train.join(y_train, rsuffix='_target')
    X_test['RET'] = np.nan # Dummy target
    
    # Mark train/test for splitting later
    df_train['is_train'] = True
    X_test['is_train'] = False
    
    df_all = pd.concat([df_train, X_test])
    return df_all

def reconstruct_returns_matrix(df):
    """
    Uses RET_1 to RET_20 to build a denoised, gap-filled matrix of returns.
    """
    print("Reconstructing Returns Matrix (Denoising)...")
    
    stock_col = None
    for col in ['STOCK', 'STOCK_ID', 'ASSET_ID']:
        if col in df.columns:
            stock_col = col
            break
            
    if stock_col is None: return None, None
    
    all_viewpoints = []
    # Collect all available lag perspectives
    for k in range(1, 21):
        col = f'RET_{k}'
        if col in df.columns:
            view = df.pivot(index='DATE', columns=stock_col, values=col)
            # Shift index so that data aligns with the actual event date
            view.index = view.index - (k - 1)
            all_viewpoints.append(view)
            
    if not all_viewpoints: return None, None
    
    # Average them to get the "consensus" return for each date
    full_returns = pd.concat(all_viewpoints).groupby(level=0).mean()
    full_returns = full_returns.sort_index().fillna(0)
    
    print(f"Matrix Shape: {full_returns.shape}")
    return full_returns, stock_col

def compute_matrix_features(returns_matrix, window_short=5, window_long=20):
    print("Computing Technical Indicators on Matrix...")
    
    features = {}
    
    # 1. Momentum / Reversion
    # Simple Moving Averages
    sma_short = returns_matrix.rolling(window=window_short).mean()
    sma_long = returns_matrix.rolling(window=window_long).mean()
    
    features['SMA_Short'] = sma_short
    features['SMA_Long'] = sma_long
    features['Mom_Signal'] = sma_short - sma_long
    
    # 2. Volatility
    vol_short = returns_matrix.rolling(window=window_short).std()
    vol_long = returns_matrix.rolling(window=window_long).std()
    features['Vol_Short'] = vol_short
    features['Vol_Long'] = vol_long
    features['Vol_Ratio'] = vol_short / (vol_long + 1e-8)
    
    # 3. RSI (Relative Strength Index) on Returns
    # Returns are essentially price changes.
    gains = returns_matrix.where(returns_matrix > 0, 0)
    losses = -returns_matrix.where(returns_matrix < 0, 0)
    
    avg_gain = gains.rolling(window=14).mean()
    avg_loss = losses.rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    features['RSI'] = 100 - (100 / (1 + rs))
    
    # 4. Bollinger Bands (on Cumulative Returns / Price Proxy)
    # Constructing a pseudo-price path
    cum_ret = (1 + returns_matrix).cumprod()
    bb_mid = cum_ret.rolling(window=20).mean()
    bb_std = cum_ret.rolling(window=20).std()
    # Distance from lower band (Mean Reversion signal)
    features['BB_Position'] = (cum_ret - bb_mid) / (2 * bb_std + 1e-8)
    
    # 5. MACD-like
    ewm12 = cum_ret.ewm(span=12).mean()
    ewm26 = cum_ret.ewm(span=26).mean()
    features['MACD'] = ewm12 - ewm26
    
    return features

def integrate_matrix_features(df, features_dict, stock_col):
    print("Merging Matrix Features...")
    
    df = df.reset_index() if df.index.name == 'ID' or 'ID' in df.index.names else df
    
    for name, matrix in features_dict.items():
        # Flatten
        flat = matrix.stack().reset_index()
        flat.columns = ['DATE', stock_col, name]
        
        # Merge
        df = df.merge(flat, on=['DATE', stock_col], how='left')
        
    return df

def feature_engineering(df):
    print("Generating Base & Rank Features...")
    
    # Standard lags and volume features if they exist
    # (Assuming basic columns are there)
    
    # Rank Features (Cross-Sectional)
    # Normalize returns across the market for that day
    if 'RET_1' in df.columns:
        df['R_RET_1'] = df.groupby('DATE')['RET_1'].transform(lambda x: x.rank(pct=True))
        
    if 'VOLUME_1' in df.columns:
        df['R_VOL_1'] = df.groupby('DATE')['VOLUME_1'].transform(lambda x: x.rank(pct=True))
    
    # Sector Neutralization
    if 'SECTOR' in df.columns and 'RET_1' in df.columns:
        df['Sector_Mean_Ret'] = df.groupby(['DATE', 'SECTOR'])['RET_1'].transform('mean')
        df['Rel_Sector_Ret'] = df['RET_1'] - df['Sector_Mean_Ret']
        
    return df

def train_ensemble(df_train, df_test):
    print("\nPreparing for Training...")
    
    df_train['target'] = df_train['RET'].astype(int)
    
    # Define features
    exclude = ['ID', 'target', 'is_train', 'RET', 'STOCK', 'DATE', 'INDUSTRY', 'SUB_INDUSTRY']
    features = [c for c in df_train.columns if c not in exclude]
    
    # Handle Categories
    cat_feats = []
    if 'SECTOR' in features:
        df_train['SECTOR'] = df_train['SECTOR'].astype('category')
        df_test['SECTOR'] = df_test['SECTOR'].astype('category')
        cat_feats.append('SECTOR')
        
    if 'INDUSTRY_GROUP' in features:
        df_train['INDUSTRY_GROUP'] = df_train['INDUSTRY_GROUP'].astype('category')
        df_test['INDUSTRY_GROUP'] = df_test['INDUSTRY_GROUP'].astype('category')
        cat_feats.append('INDUSTRY_GROUP')
        
    print(f"Training with {len(features)} features: {features}")
    
    # --- LightGBM ---
    lgb_params = {
        'n_estimators': 1000,
        'learning_rate': 0.015,
        'max_depth': 7,
        'num_leaves': 63,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    # --- Random Forest ---
    rf_params = {
        'n_estimators': 300,
        'max_depth': 12,
        'min_samples_split': 10,
        'n_jobs': -1,
        'random_state': 42
    }
    
    # --- Extra Trees ---
    et_params = {
        'n_estimators': 300,
        'max_depth': 12,
        'min_samples_split': 10,
        'n_jobs': -1,
        'random_state': 42
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    dates = df_train['DATE'].unique()
    
    oof_preds = np.zeros(len(df_train))
    test_preds_accum = np.zeros(len(df_test))
    
    scores = []
    
    # We will split by DATE to prevent leakage if using lagged features inappropriately,
    # but since we are reconstructing the matrix carefully, simple KFold on dates is good.
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(dates)):
        # Add better progress tracking to show train_idx percentage
        print(f"Training fold {fold+1} / 5...")
        train_dates = dates[train_idx]
        val_dates = dates[val_idx]
        
        train_mask = df_train['DATE'].isin(train_dates)
        val_mask = df_train['DATE'].isin(val_dates)
        
        X_t, y_t = df_train.loc[train_mask, features], df_train.loc[train_mask, 'target']
        X_v, y_v = df_train.loc[val_mask, features], df_train.loc[val_mask, 'target']
        
        # Train LightGBM
        clf_lgb = lgb.LGBMClassifier(**lgb_params)
        clf_lgb.fit(X_t, y_t, eval_set=[(X_v, y_v)], categorical_feature=cat_feats)
        
        # Train RF (Impute NaNs for sklearn models)
        X_t_fill = X_t.fillna(0)
        X_v_fill = X_v.fillna(0)
        
        # Drop categorical for RF/ET if they can't handle it natively (Sklearn needs OneHot, but RF can handle numeric codes)
        # We'll just drop category cols for RF to be safe/fast or use codes.
        X_t_rf = X_t_fill.select_dtypes(exclude=['category', 'object'])
        X_v_rf = X_v_fill.select_dtypes(exclude=['category', 'object'])
        
        clf_rf = RandomForestClassifier(**rf_params)
        clf_rf.fit(X_t_rf, y_t)
        
        # clf_et = ExtraTreesClassifier(**et_params)
        # clf_et.fit(X_t_rf, y_t)
        
        # Ensemble Predictions (Weighted Average)
        p_lgb = clf_lgb.predict_proba(X_v)[:, 1]
        p_rf = clf_rf.predict_proba(X_v_rf)[:, 1]
        # p_et = clf_et.predict_proba(X_v_rf)[:, 1]
        
        # 0.7 LGB, 0.3 RF
        fold_preds = 0.7 * p_lgb + 0.3 * p_rf 
        
        # Threshold optimization? Default 0.5
        acc = accuracy_score(y_v, (fold_preds > 0.5).astype(int))
        scores.append(acc)
        
        print(f"Fold {fold+1} Accuracy: {acc:.4f} (LGB: {accuracy_score(y_v, clf_lgb.predict(X_v)):.4f})")
        
        # Predict on Test
        # Retrain on full fold? No, simpler to average predictions
        test_preds_accum += (0.7 * clf_lgb.predict_proba(df_test[features])[:, 1] + 
                             0.3 * clf_rf.predict_proba(df_test[features].select_dtypes(exclude=['category', 'object']).fillna(0))[:, 1])
        
    print(f"\nOverall CV Accuracy: {np.mean(scores):.4f} (+- {np.std(scores):.4f})")
    
    # Finalize Test Preds
    test_preds_avg = test_preds_accum / 5
    submission = pd.DataFrame({
        'ID': df_test.index if 'ID' not in df_test.columns else df_test['ID'],
        'RET': (test_preds_avg > 0.5).astype(bool)
    })
    
    submission.to_csv('submission_powerful.csv', index=False)
    print("Saved submission_powerful.csv")

def main():
    df_all = load_data()
    
    # 1. Reconstruct Matrix
    full_returns, stock_col = reconstruct_returns_matrix(df_all)
    
    if full_returns is not None:
        # 2. Compute Technicals
        matrix_feats = compute_matrix_features(full_returns)
        
        # 3. Merge Back
        df_all = integrate_matrix_features(df_all, matrix_feats, stock_col)
    
    # 4. Standard Feature Engineering
    df_all = feature_engineering(df_all)
    
    # 5. Split and Train
    df_train = df_all[df_all['is_train'] == True].copy()
    df_test = df_all[df_all['is_train'] == False].copy()
    
    train_ensemble(df_train, df_test)

if __name__ == "__main__":
    main()
