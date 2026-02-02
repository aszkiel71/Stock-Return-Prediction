import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
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

def feature_engineering_advanced(df):
    print("Generating Advanced Features (Ranks + Clusters)...")
    
    # Ensure RET_1 exists
    if 'RET_1' not in df.columns:
        print("WARNING: RET_1 not found in columns:", df.columns.tolist()[:10])
    
    feat_cols = [c for c in df.columns if 'RET_' in c or 'VOLUME_' in c] # More specific to avoid capturing 'RET' target or other things
    print(f"Found {len(feat_cols)} feature columns to rank.")
    
    for col in feat_cols:
        df[f'R_{col}'] = df.groupby('DATE')[col].transform(lambda x: x.rank(pct=True))

    # Verify R_RET_1 creation
    if 'R_RET_1' not in df.columns:
         print("WARNING: R_RET_1 was not created! Checking RET_1 presence...")
         if 'RET_1' in df.columns:
             print("Creating R_RET_1 explicitly.")
             df['R_RET_1'] = df.groupby('DATE')['RET_1'].transform(lambda x: x.rank(pct=True))
         else:
             print("ERROR: Cannot create R_RET_1 because RET_1 is missing.")

    cluster_feats = [f'R_RET_{i}' for i in range(1, 6) if f'R_RET_{i}' in df.columns]
    
    if cluster_feats:
        X_cluster = df[cluster_feats].fillna(0.5) 
        
        kmeans = MiniBatchKMeans(n_clusters=50, batch_size=4096)
        # Using fit_predict on all data (Transductive)
        df['CLUSTER'] = kmeans.fit_predict(X_cluster)
        
        df['R_Cluster_Mean'] = df.groupby(['DATE', 'CLUSTER'])['R_RET_1'].transform('mean')
        df['R_Rel_Cluster'] = df['R_RET_1'] - df['R_Cluster_Mean']
        
    if 'SECTOR' in df.columns and 'R_RET_1' in df.columns:
        df['R_Sector_Mean'] = df.groupby(['DATE', 'SECTOR'])['R_RET_1'].transform('mean')
        df['R_Rel_Sector'] = df['R_RET_1'] - df['R_Sector_Mean']
        
    return df

def get_pca_ou_features(df, n_components=30):
    print("Calculating PCA & OU Features...")
    stock_col = None
    for col in ['STOCK', 'STOCK_ID', 'ASSET_ID']:
        if col in df.columns:
            stock_col = col
            break
    if stock_col is None: return df

    returns = df.pivot(index='DATE', columns=stock_col, values='R_RET_1').fillna(0.5)
    
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
        lag = series.shift(1)
        
        cov = series.rolling(window).cov(lag)
        var = lag.rolling(window).var()
        b = cov / (var + 1e-8)
        
        mean_t = series.rolling(window).mean()
        mean_lag = lag.rolling(window).mean()
        a = mean_t - b * mean_lag
        
        theta_df[stock] = 1 - b
        mu_df[stock] = a / (1 - b + 1e-8)
        
    roll_std = cum_residuals.rolling(window).std()
    ou_signal = (cum_residuals - mu_df) / (roll_std + 1e-8)
    
    def flatten(wide, name):
        flat = wide.stack().reset_index()
        flat.columns = ['DATE', stock_col, name]
        return flat

    df = df.reset_index()
    df = df.merge(flatten(ou_signal, 'OU_Signal'), on=['DATE', stock_col], how='left')
    df = df.merge(flatten(theta_df, 'OU_Theta'), on=['DATE', stock_col], how='left')
    
    return df

def train_and_evaluate(df_train, df_test):
    print("Starting Advanced Ensemble Training...")
    df_train['target'] = df_train['RET'].astype(int)
    
    features = [c for c in df_train.columns if 'R_RET_' in c or 'R_VOLUME_' in c]
    features += ['R_Rel_Cluster', 'R_Rel_Sector']
    features += ['OU_Signal', 'OU_Theta']
    if 'SECTOR' in df_train.columns:
        df_train['SECTOR_CAT'] = df_train['SECTOR'].astype('category').cat.codes
        df_test['SECTOR_CAT'] = df_test['SECTOR'].astype('category').cat.codes
        features.append('SECTOR_CAT')
        
    features = list(set([f for f in features if f in df_train.columns]))
    print(f"Training with {len(features)} Features.")
    
    dates = df_train['DATE'].unique()
    kf = KFold(n_splits=5, shuffle=True)
    
    lgb_dart = lgb.LGBMClassifier(
        boosting_type='dart',
        n_estimators=700,
        learning_rate=0.02,
        num_leaves=63,
        max_depth=8,
        subsample=0.7,
        colsample_bytree=0.7,
        drop_rate=0.1,
        n_jobs=-1,
        verbose=-1
    )
    
    et_clf = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=10,
        n_jobs=-1
    )
    
    ensemble = VotingClassifier(
        estimators=[('lgb', lgb_dart), ('et', et_clf)],
        voting='soft',
        weights=[2, 1]
    )
    
    scores = []
    
    for i, (train_idx, val_idx) in enumerate(kf.split(dates)):
        t_dates, v_dates = dates[train_idx], dates[val_idx]
        train_mask, val_mask = df_train['DATE'].isin(t_dates), df_train['DATE'].isin(v_dates)
        
        X_train = df_train.loc[train_mask, features].fillna(0.5)
        y_train = df_train.loc[train_mask, 'target']
        X_val = df_train.loc[val_mask, features].fillna(0.5)
        y_val = df_train.loc[val_mask, 'target']
        
        ensemble.fit(X_train, y_train)
        
        preds = ensemble.predict(X_val)
        acc = accuracy_score(y_val, preds)
        scores.append(acc)
        print(f"Fold {i+1}: Accuracy = {acc:.4f}")
        
    print(f"\nOverall Accuracy: {np.mean(scores):.4f} (+- {np.std(scores):.4f})")
    
    results_logger.log_results(
        "Advanced_Clustering_Ensemble", 
        scores, 
        np.mean(scores), 
        np.std(scores), 
        description="LGBM DART + ExtraTrees with Ranks, Clusters, PCA, OU Features"
    )

    # Full Train and Predict
    print("Retraining on full dataset and generating submission...")
    # Re-instantiate needed to reset
    lgb_dart = lgb.LGBMClassifier(
        boosting_type='dart',
        n_estimators=700,
        learning_rate=0.02,
        num_leaves=63,
        max_depth=8,
        subsample=0.7,
        colsample_bytree=0.7,
        drop_rate=0.1,
        n_jobs=-1,
        verbose=-1
    )
    et_clf = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=10,
        n_jobs=-1
    )
    ensemble = VotingClassifier(
        estimators=[('lgb', lgb_dart), ('et', et_clf)],
        voting='soft',
        weights=[2, 1]
    )

    X_full = df_train[features].fillna(0.5)
    y_full = df_train['target']
    
    ensemble.fit(X_full, y_full)
    
    X_test_final = df_test[features].fillna(0.5)
    test_preds = ensemble.predict(X_test_final)
    
    submission = pd.DataFrame({
        'ID': df_test.index if 'ID' not in df_test.columns else df_test['ID'],
        'RET': test_preds.astype(bool)
    })
    
    submission.to_csv('submission_advanced_clustering_ensemble.csv', index=False)
    print("Saved submission_advanced_clustering_ensemble.csv")

if __name__ == "__main__":
    try:
        df_train, df_test = load_data()
        
        df_train['is_train'] = True
        df_test['is_train'] = False
        df_all = pd.concat([df_train, df_test])
        
        df_all = feature_engineering_advanced(df_all)
        df_all = get_pca_ou_features(df_all, n_components=30)
        
        # Split back
        df_train = df_all[df_all['is_train'] == True].copy()
        df_test = df_all[df_all['is_train'] == False].copy()
        
        # Restore index
        if 'ID' in df_train.columns: df_train = df_train.set_index('ID')
        if 'ID' in df_test.columns: df_test = df_test.set_index('ID')
        
        train_and_evaluate(df_train, df_test)
    except Exception as e:
        import traceback
        traceback.print_exc()
