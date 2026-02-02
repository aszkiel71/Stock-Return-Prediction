import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import lightgbm as lgb
import warnings
import results_logger

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
    
    # Add dummy target to test
    X_test['RET'] = np.nan
    
    return df_train, X_test

def feature_engineering(df):
    print("Generating Features: Rank Transformation & Technicals...")
    
    rank_cols = [c for c in df.columns if 'RET_' in c or 'VOLUME_' in c]
    for col in rank_cols:
        df[f'R_{col}'] = df.groupby('DATE')[col].transform(lambda x: x.rank(pct=True))
        
    if 'R_RET_1' in df.columns and 'SECTOR' in df.columns:
        df['R_Sector_Mean'] = df.groupby(['DATE', 'SECTOR'])['R_RET_1'].transform('mean')
        df['R_Rel_Sector'] = df['R_RET_1'] - df['R_Sector_Mean']

    r_ret_cols = [f'R_RET_{i}' for i in range(1, 6)]
    if all(c in df.columns for c in r_ret_cols):
        df['R_Vol_5d'] = df[r_ret_cols].std(axis=1)
        df['R_Mom_5d'] = df[r_ret_cols].mean(axis=1)
        
    return df

def get_pca_ou_features(df, n_components=20):
    print("Calculating PCA & OU Features...")
    stock_col = None
    for col in ['STOCK', 'STOCK_ID', 'ASSET_ID']:
        if col in df.columns:
            stock_col = col
            break
    if stock_col is None: return df

    returns = df.pivot(index='DATE', columns=stock_col, values='RET_1').fillna(0)
    pca = PCA(n_components=n_components)
    pca.fit(returns)
    factors = pca.transform(returns)
    common = pca.inverse_transform(factors)
    residuals = pd.DataFrame(returns - common, index=returns.index, columns=returns.columns)
    cum_residuals = residuals.cumsum()
    
    window = 60
    theta_df = pd.DataFrame(index=cum_residuals.index, columns=cum_residuals.columns)
    mu_df = pd.DataFrame(index=cum_residuals.index, columns=cum_residuals.columns)
    
    for stock in cum_residuals.columns:
        series = cum_residuals[stock]
        series_lag = series.shift(1)
        b = series.rolling(window).cov(series_lag) / (series_lag.rolling(window).var() + 1e-8)
        a = series.rolling(window).mean() - b * series_lag.rolling(window).mean()
        
        theta_df[stock] = 1 - b
        mu_df[stock] = a / (1 - b + 1e-8)
        
    roll_std = cum_residuals.rolling(window=window).std()
    ou_signal = (cum_residuals - mu_df) / (roll_std + 1e-8)
    
    def flatten(wide, name):
        flat = wide.stack().reset_index()
        flat.columns = ['DATE', stock_col, name]
        return flat

    f_sig = flatten(ou_signal, 'OU_Signal')
    f_theta = flatten(theta_df, 'OU_Theta')
    
    df = df.reset_index()
    df = df.merge(f_sig, on=['DATE', stock_col], how='left')
    df = df.merge(f_theta, on=['DATE', stock_col], how='left')
    
    return df

def train_and_evaluate(df_train, df_test):
    print("Starting Ensemble Training (LightGBM + Random Forest)...")
    df_train['target'] = df_train['RET'].astype(int)
    
    features = [c for c in df_train.columns if 'R_RET_' in c]
    features += [c for c in df_train.columns if 'R_VOLUME_' in c]
    features += ['R_Rel_Sector', 'R_Vol_5d', 'R_Mom_5d', 'OU_Signal', 'OU_Theta']
    if 'SECTOR' in df_train.columns:
        df_train['SECTOR_CAT'] = df_train['SECTOR'].astype('category').cat.codes
        df_test['SECTOR_CAT'] = df_test['SECTOR'].astype('category').cat.codes
        features.append('SECTOR_CAT')

    features = [f for f in features if f in df_train.columns]
    
    dates = df_train['DATE'].unique()
    kf = KFold(n_splits=5, shuffle=True)
    
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.03, num_leaves=31, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, n_jobs=-1, verbose=-1
    )
    
    rf_clf = RandomForestClassifier(
        n_estimators=100, max_depth=8, n_jobs=-1
    )
    
    ensemble = VotingClassifier(
        estimators=[('lgb', lgb_clf), ('rf', rf_clf)],
        voting='soft'
    )
    
    scores = []
    for i, (train_idx, val_idx) in enumerate(kf.split(dates)):
        t_dates, v_dates = dates[train_idx], dates[val_idx]
        train_mask, val_mask = df_train['DATE'].isin(t_dates), df_train['DATE'].isin(v_dates)
        
        X_train, y_train = df_train.loc[train_mask, features].fillna(0), df_train.loc[train_mask, 'target']
        X_val, y_val = df_train.loc[val_mask, features].fillna(0), df_train.loc[val_mask, 'target']
        
        ensemble.fit(X_train, y_train)
        acc = accuracy_score(y_val, ensemble.predict(X_val))
        scores.append(acc)
        print(f"Fold {i+1}: Accuracy = {acc:.4f}")
        
    print(f"\nOverall Accuracy: {np.mean(scores):.4f} (+- {np.std(scores):.4f})")
    
    results_logger.log_results(
        "Rank_Ensemble", 
        scores, 
        np.mean(scores), 
        np.std(scores), 
        description="LGBM + Random Forest with Rank Features, OU, Sector Neutrality"
    )

    # Full Train and Predict
    print("Retraining on full dataset and generating submission...")
    # Re-instantiate needed to reset
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.03, num_leaves=31, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, n_jobs=-1, verbose=-1
    )
    rf_clf = RandomForestClassifier(
        n_estimators=100, max_depth=8, n_jobs=-1
    )
    ensemble = VotingClassifier(
        estimators=[('lgb', lgb_clf), ('rf', rf_clf)],
        voting='soft'
    )
    
    X_full = df_train[features].fillna(0)
    y_full = df_train['target']
    
    ensemble.fit(X_full, y_full)
    
    X_test_final = df_test[features].fillna(0)
    test_preds = ensemble.predict(X_test_final)
    
    submission = pd.DataFrame({
        'ID': df_test.index if 'ID' not in df_test.columns else df_test['ID'],
        'RET': test_preds.astype(bool)
    })
    
    submission.to_csv('submission_rank_ensemble_model.csv', index=False)
    print("Saved submission_rank_ensemble_model.csv")

def main():
    try:
        df_train, df_test = load_data()
        
        df_train['is_train'] = True
        df_test['is_train'] = False
        df_all = pd.concat([df_train, df_test])
        
        df_all = feature_engineering(df_all)
        df_all = get_pca_ou_features(df_all)
        
        # Split back
        df_train = df_all[df_all['is_train'] == True].copy()
        df_test = df_all[df_all['is_train'] == False].copy()
        
        # Restore index logic
        # get_pca_ou_features uses merge which might drop index
        # We need to ensure ID is preserved.
        # In this script, get_pca_ou_features does reset_index at end.
        # So ID is likely in 'ID' column (if it was named ID) or 'index' column.
        
        if 'ID' in df_train.columns: df_train = df_train.set_index('ID')
        if 'ID' in df_test.columns: df_test = df_test.set_index('ID')
            
        train_and_evaluate(df_train, df_test)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
