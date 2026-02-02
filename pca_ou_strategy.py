import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
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
    
    X_test['RET'] = np.nan
    
    return df_train, X_test

def feature_engineering(df):
    print("Generating Base Features...")
    
    if 'RET_1' in df.columns:
        df['Market_Ret_1'] = df.groupby('DATE')['RET_1'].transform('mean')
        df['Rel_Market_Ret_1'] = df['RET_1'] - df['Market_Ret_1']
        
        if 'SECTOR' in df.columns:
            df['Sector_Ret_1'] = df.groupby(['DATE', 'SECTOR'])['RET_1'].transform('mean')
            df['Rel_Sector_Ret_1'] = df['RET_1'] - df['Sector_Ret_1']
    
    ret_cols = [f'RET_{i}' for i in range(1, 6)]
    if all(c in df.columns for c in ret_cols):
        df['Vol_5d'] = df[ret_cols].std(axis=1)
        
    return df

def vectorized_ou_params(series, window=60):
    series_lag = series.shift(1)
    
    rolling_cov = series.rolling(window=window).cov(series_lag)
    rolling_var = series_lag.rolling(window=window).var()
    rolling_mean_t = series.rolling(window=window).mean()
    rolling_mean_lag = series_lag.rolling(window=window).mean()
    
    b = rolling_cov / (rolling_var + 1e-8)
    
    a = rolling_mean_t - b * rolling_mean_lag
    
    theta = 1 - b
    
    mu = a / (theta + 1e-8)
    
    return theta, mu

def get_pca_ou_features(df, n_components=20):
    print(f"Calculating PCA & OU Features with GAP FILLING (n_components={n_components})...")
    
    stock_col = None
    for col in ['STOCK', 'STOCK_ID', 'ASSET_ID']:
        if col in df.columns:
            stock_col = col
            break
    
    if stock_col is None:
        return df


    # --- GAP FILLING / TIMELINE RECONSTRUCTION (Denoised) ---
    all_viewpoints = []
    for k in range(1, 21):
        col = f'RET_{k}'
        if col in df.columns:
            # Pivot the lag k
            view = df.pivot(index='DATE', columns=stock_col, values=col)
            # Shift index: Date T, Lag k represents return at Date T-(k-1)
            view.index = view.index - (k - 1)
            all_viewpoints.append(view)
    
    # Average all viewpoints for each date to denoise and fill gaps
    full_returns = pd.concat(all_viewpoints).groupby(level=0).mean()
    full_returns = full_returns.sort_index().fillna(0)
    print(f"Reconstructed Matrix Shape: {full_returns.shape} (Denoised via {len(all_viewpoints)} lags)")

    pca = PCA(n_components=n_components)
    pca.fit(full_returns)
    factors = pca.transform(full_returns)
    common = pca.inverse_transform(factors)

    residuals = pd.DataFrame(full_returns - common, index=full_returns.index, columns=full_returns.columns)
    # print(residuals.shape)
    cum_residuals = residuals.cumsum()
    
    print("Computing Vectorized OU Parameters (this is fast)...")
    
    ou_features = []
    
    window = 60
    
    theta_df = pd.DataFrame(index=cum_residuals.index, columns=cum_residuals.columns)
    mu_df = pd.DataFrame(index=cum_residuals.index, columns=cum_residuals.columns)
    
    cum_lag = cum_residuals.shift(1)
    
    for stock in cum_residuals.columns:
        series = cum_residuals[stock]
        t, m = vectorized_ou_params(series, window=window)
        theta_df[stock] = t
        mu_df[stock] = m

    ou_deviation = cum_residuals - mu_df
    
    roll_std = cum_residuals.rolling(window=window).std()
    ou_signal = ou_deviation / (roll_std + 1e-8)
    
    print("Merging features...")
    
    def flatten(wide_df, name):
        flat = wide_df.stack().reset_index()
        flat.columns = ['DATE', stock_col, name]
        return flat

    f_theta = flatten(theta_df, 'OU_Theta')
    f_mu = flatten(mu_df, 'OU_Mu')
    f_sig = flatten(ou_signal, 'OU_Signal')
    f_res = flatten(residuals, 'PCA_Resid')
    
    df = df.reset_index()
    df = df.merge(f_theta, on=['DATE', stock_col], how='left')
    df = df.merge(f_sig, on=['DATE', stock_col], how='left')
    df = df.merge(f_res, on=['DATE', stock_col], how='left')
    
    return df

def train_and_evaluate(df_train, df_test):
    print("Starting LightGBM Training...")
    
    df_train['target'] = df_train['RET'].astype(int)
    
    features = [
        'RET_1', 'RET_2', 'RET_3', 'RET_4', 'RET_5',
        'VOLUME_1', 'VOLUME_2', 'VOLUME_3',
        'Rel_Market_Ret_1', 'Rel_Sector_Ret_1', 'Vol_5d',
        'PCA_Resid', 'OU_Theta', 'OU_Signal',
        'SECTOR'
    ]
    
    features = [f for f in features if f in df_train.columns]
    
    cat_feats = []
    if 'SECTOR' in df_train.columns:
        df_train['SECTOR'] = df_train['SECTOR'].astype('category')
        df_test['SECTOR'] = df_test['SECTOR'].astype('category')
        cat_feats.append('SECTOR')
    
    print(f"Features ({len(features)}): {features}")
    

    dates = df_train['DATE'].unique()
    kf = KFold(n_splits=5, shuffle=True)
    
    scores = []
    
    params = {
        'n_estimators': 600,
        'learning_rate': 0.02,
        'max_depth': 6,
        'num_leaves': 31,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_jobs': -1,
        'verbose': -1
    }
    
    for i, (train_date_idx, val_date_idx) in enumerate(kf.split(dates)):
        t_dates = dates[train_date_idx]
        v_dates = dates[val_date_idx]
        
        train_mask = df_train['DATE'].isin(t_dates)
        val_mask = df_train['DATE'].isin(v_dates)
        
        X_t = df_train.loc[train_mask, features]
        y_t = df_train.loc[train_mask, 'target']
        
        X_v = df_train.loc[val_mask, features]
        y_v = df_train.loc[val_mask, 'target']
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_t, y_t, categorical_feature=cat_feats)
        
        preds = model.predict(X_v)
        acc = accuracy_score(y_v, preds)
        
        scores.append(acc)
        print(f"Fold {i+1}: Accuracy = {acc:.4f}")
        
    print(f"\nOverall Accuracy: {np.mean(scores):.4f} (+- {np.std(scores):.4f})")
    
    results_logger.log_results(
        "PCA_OU_Strategy", 
        scores, 
        np.mean(scores), 
        np.std(scores), 
        description="LightGBM with PCA Residuals and OU Process Features"
    )

    # Full Train and Predict
    print("Retraining on full dataset and generating submission...")
    model_full = lgb.LGBMClassifier(**params)
    model_full.fit(df_train[features], df_train['target'], categorical_feature=cat_feats)
    
    test_preds = model_full.predict(df_test[features])
    
    submission = pd.DataFrame({
        'ID': df_test.index if 'ID' not in df_test.columns else df_test['ID'],
        'RET': test_preds.astype(bool)
    })
    

    
    submission.to_csv('submission_pca_ou_strategy.csv', index=False)
    print("Saved submission_pca_ou_strategy.csv")

def main():
    try:
        df_train, df_test = load_data()
        
        # Concat for features
        df_train['is_train'] = True
        df_test['is_train'] = False
        df_all = pd.concat([df_train, df_test])
        
        df_all = feature_engineering(df_all)
        df_all = get_pca_ou_features(df_all, n_components=33)
        
        df_train = df_all[df_all['is_train'] == True].copy()
        df_test = df_all[df_all['is_train'] == False].copy()
        
        
        if 'ID' in df_train.columns:
            df_train = df_train.set_index('ID')
        if 'ID' in df_test.columns:
            df_test = df_test.set_index('ID')
            
        train_and_evaluate(df_train, df_test)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
