import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
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

def generate_shapelet_features(df):
    print("Generating Shapelet Features...")
    
    ret_cols = [f'RET_{i}' for i in range(20, 0, -1)]
    ret_cols = [c for c in ret_cols if c in df.columns]
    
    if not ret_cols:
        return df
    
    X_seq = df[ret_cols].fillna(0).values
    
    X_mean = X_seq.mean(axis=1, keepdims=True)
    X_std = X_seq.std(axis=1, keepdims=True) + 1e-8
    X_norm = (X_seq - X_mean) / X_std
    
    shapes_5 = {
        'UpTrend': np.linspace(-1, 1, 5),
        'DownTrend': np.linspace(1, -1, 5),
        'V_Shape': np.array([1, 0, -1, 0, 1]), 
        'A_Shape': np.array([-1, 0, 1, 0, -1]), 
        'Reversal_Up': np.array([-1, -1, -1, 0, 1]), 
        'Reversal_Down': np.array([1, 1, 1, 0, -1])  
    }
    
    X_last_5 = X_norm[:, -5:]
    
    for name, shape in shapes_5.items():
        s_norm = (shape - shape.mean()) / (shape.std() + 1e-8)
        
        dot_prod = np.dot(X_last_5, s_norm)
        corr = dot_prod / 5.0
        
        df[f'Shape_{name}_Corr'] = corr
        
        dist = np.linalg.norm(X_last_5 - s_norm, axis=1)
        df[f'Shape_{name}_Dist'] = dist

    if X_norm.shape[1] >= 20:
        shapes_20 = {
            'Long_Up': np.linspace(-1, 1, 20),
            'Long_Down': np.linspace(1, -1, 20),
            'U_Turn': np.concatenate([np.linspace(1, -1, 10), np.linspace(-1, 1, 10)])
        }
        
        X_last_20 = X_norm[:, -20:]
        
        for name, shape in shapes_20.items():
            s_norm = (shape - shape.mean()) / (shape.std() + 1e-8)
            dot_prod = np.dot(X_last_20, s_norm)
            df[f'Shape_{name}_Corr'] = dot_prod / 20.0
            
    return df

def feature_engineering_basic(df):
    print("Generating Base Ranks & Features...")
    feat_cols = [c for c in df.columns if 'RET' in c or 'VOLUME' in c]
    for col in feat_cols:
        df[f'R_{col}'] = df.groupby('DATE')[col].transform(lambda x: x.rank(pct=True))
        
    if 'SECTOR' in df.columns and 'R_RET_1' in df.columns:
        df['R_Sector_Mean'] = df.groupby(['DATE', 'SECTOR'])['R_RET_1'].transform('mean')
        df['R_Rel_Sector'] = df['R_RET_1'] - df['R_Sector_Mean']
        
    return df

def get_pca_ou_features(df, n_components=20):
    stock_col = None
    for col in ['STOCK', 'STOCK_ID']:
        if col in df.columns: stock_col = col; break
    if not stock_col: return df
    
    returns = df.pivot(index='DATE', columns=stock_col, values='R_RET_1').fillna(0.5)
    pca = PCA(n_components=n_components)
    pca.fit(returns)
    factors = pca.transform(returns)
    common = pca.inverse_transform(factors)
    residuals = pd.DataFrame(returns - common, index=returns.index, columns=returns.columns)
    cum_res = residuals.cumsum()
    
    window = 60
    mu_df = cum_res.rolling(window).mean() 
    roll_std = cum_res.rolling(window).std()
    ou_signal = (cum_res - mu_df) / (roll_std + 1e-8)
    
    def flatten(wide, name):
        flat = wide.stack().reset_index()
        flat.columns = ['DATE', stock_col, name]
        return flat
        
    df = df.reset_index()
    df = df.merge(flatten(ou_signal, 'OU_Signal'), on=['DATE', stock_col], how='left')
    return df

def train_and_evaluate(df_train, df_test):
    print("Starting Training with Shapelets...")
    df_train['target'] = df_train['RET'].astype(int)
    
    features = [c for c in df_train.columns if 'R_RET_' in c or 'R_VOLUME_' in c]
    features += [c for c in df_train.columns if 'Shape_' in c]
    features += ['R_Rel_Sector', 'OU_Signal']
    if 'SECTOR' in df_train.columns:
        df_train['SECTOR_CAT'] = df_train['SECTOR'].astype('category').cat.codes
        df_test['SECTOR_CAT'] = df_test['SECTOR'].astype('category').cat.codes
        features.append('SECTOR_CAT')
        
    features = list(set([f for f in features if f in df_train.columns]))
    print(f"Using {len(features)} features.")
    
    dates = df_train['DATE'].unique()
    kf = KFold(n_splits=5, shuffle=True)
    
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.02, num_leaves=40, max_depth=7,
        subsample=0.8, colsample_bytree=0.7, n_jobs=-1, verbose=-1
    )
    
    et_clf = ExtraTreesClassifier(n_estimators=200, max_depth=10, n_jobs=-1)
    
    ensemble = VotingClassifier(
        estimators=[('lgb', lgb_clf), ('et', et_clf)],
        voting='soft', weights=[2, 1]
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
        acc = accuracy_score(y_val, ensemble.predict(X_val))
        scores.append(acc)
        print(f"Fold {i+1}: Accuracy = {acc:.4f}")
        
    print(f"\nOverall Accuracy: {np.mean(scores):.4f} (+- {np.std(scores):.4f})")
    
    results_logger.log_results(
        "Shapelet_Pattern_Strategy", 
        scores, 
        np.mean(scores), 
        np.std(scores), 
        description="Ensemble (LGBM+ET) with Shapelet Pattern Matching Features"
    )

    # Full Train & Predict
    print("Retraining on full dataset and generating submission...")
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.02, num_leaves=40, max_depth=7,
        subsample=0.8, colsample_bytree=0.7, n_jobs=-1, verbose=-1
    )
    et_clf = ExtraTreesClassifier(n_estimators=200, max_depth=10, n_jobs=-1)
    ensemble = VotingClassifier(
        estimators=[('lgb', lgb_clf), ('et', et_clf)],
        voting='soft', weights=[2, 1]
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
    
    submission.to_csv('submission_shapelet_pattern_strategy.csv', index=False)
    print("Saved submission_shapelet_pattern_strategy.csv")

if __name__ == "__main__":
    try:
        df_train, df_test = load_data()
        
        df_train['is_train'] = True
        df_test['is_train'] = False
        df_all = pd.concat([df_train, df_test])
        
        df_all = feature_engineering_basic(df_all)
        df_all = generate_shapelet_features(df_all)
        df_all = get_pca_ou_features(df_all)
        
        # Split back
        df_train = df_all[df_all['is_train'] == True].copy()
        df_test = df_all[df_all['is_train'] == False].copy()
        
        if 'ID' in df_train.columns: df_train = df_train.set_index('ID')
        if 'ID' in df_test.columns: df_test = df_test.set_index('ID')
        
        train_and_evaluate(df_train, df_test)
    except Exception as e:
        import traceback
        traceback.print_exc()
