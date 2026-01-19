import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import os


curr_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(curr_path, '.', 'data')


x_train = pd.read_csv(os.path.join(data_path, 'x_train.csv'), index_col="ID")
y_train = pd.read_csv(os.path.join(data_path, 'y_train.csv'), index_col="ID")
x_test = pd.read_csv(os.path.join(data_path, 'x_test.csv'), index_col="ID")


def feature_engineering(df, shift = 1):
    col_name = f'RET_{shift}'
    if col_name in df.columns:
        df[f'Mean_Sector_Ret_{shift}'] = df.groupby(['SECTOR', 'DATE'])[col_name].transform('mean')


def add_pair_features(df, shift=1):
    """
    Generates 'Pair Trading' style features by calculating the spread
    between a stock's return and its group's (Sector, Market) return.
    Hypothesis: Stocks deviating from their group will revert (Mean Reversion).
    """
    col_name = f'RET_{shift}'
    if col_name in df.columns:
        # 1. Market Spread (Stock vs All Stocks on that Date)
        market_mean = df.groupby('DATE')[col_name].transform('mean')
        df[f'Spread_Market_{shift}'] = df[col_name] - market_mean

        # # 2. Sector Spread (Stock vs Sector Peers) - The classic "Pair" proxy
        # sector_mean = df.groupby(['SECTOR', 'DATE'])[col_name].transform('mean')
        # df[f'Spread_Sector_{shift}'] = df[col_name] - sector_mean
        
        # sector_std = df.groupby(['SECTOR', 'DATE'])[col_name].transform('std')
        # df[f'Z_Score_Sector_{shift}'] = (df[f'Spread_Sector_{shift}'] / (sector_std + 1e-8))


def random_forest_classifier(x_train, y_train, x_test, n_splits=5):
    print("Training Random Forest Classifier")
    
    train_dates = x_train['DATE'].unique()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    scores = []
    test_probs_sum = np.zeros(len(x_test))
    
    x_test_filled = x_test.fillna(0)

    for i, (train_date_idx, val_date_idx) in enumerate(kf.split(train_dates)):
        t_dates = train_dates[train_date_idx]
        v_dates = train_dates[val_date_idx]
        
        t_mask = x_train['DATE'].isin(t_dates)
        v_mask = x_train['DATE'].isin(v_dates)
        
        X_t = x_train.loc[t_mask].fillna(0)
        y_t = y_train.loc[t_mask].values.ravel()
        X_v = x_train.loc[v_mask].fillna(0)
        y_v = y_train.loc[v_mask].values.ravel()
        
        model = RandomForestClassifier(n_estimators = 100, max_depth=6, n_jobs=-1)
        model.fit(X_t, y_t)
        
        val_probs = model.predict_proba(X_v)[:, 1]
        
        val_dates = x_train.loc[v_mask, 'DATE']
        temp_df = pd.DataFrame({'DATE': val_dates, 'pred': val_probs})
        val_preds = temp_df.groupby('DATE')['pred'].transform(lambda x: x > x.median()).astype(int).values
        
        score = accuracy_score(y_v, val_preds)
        scores.append(score)
        print(f"Fold {i+1} - Accuracy: {score:.4f}")
        
        test_probs_sum += model.predict_proba(x_test_filled)[:, 1]

    avg_test_probs = test_probs_sum / n_splits
    test_res = pd.DataFrame({'DATE': x_test['DATE'], 'pred': avg_test_probs})
    final_test_preds = test_res.groupby('DATE')['pred'].transform(lambda x: x > x.median()).astype(int).values
    
    mean_acc = np.mean(scores)
    std_acc = np.std(scores)
    print(f'Overall Accuracy: {mean_acc:.4f} (+- {std_acc:.4f})')
    
    return final_test_preds


selected_features = ["DATE", "SECTOR", "RET_1", "VOLUME_1", "RET_2", "VOLUME_2", "RET_3", "VOLUME_3", "RET_4", "VOLUME_4",
                     "RET_5", "VOLUME_5"]
x_train = x_train[selected_features]
x_test = x_test[selected_features]


test_predictions = random_forest_classifier(x_train, y_train, x_test)
# Overall Accuracy: 0.5133 (+- 0.0020)
# Overall Accuracy: 0.5128 (+- 0.0027)
# Overall Accuracy: 0.5132 (+- 0.0021)
# Overall Accuracy: 0.5123 (+- 0.0033)

# Overall Accuracy: 0.5128 (+- 0.0016)

feature_engineering(x_train)
feature_engineering(x_test)
feature_engineering(x_train, shift=2)
feature_engineering(x_test, shift=2)
feature_engineering(x_train, shift=3)
feature_engineering(x_test, shift=3)

test_predictions = random_forest_classifier(x_train, y_train, x_test)
# Overall Accuracy: 0.5147 (+- 0.0037)
# Overall Accuracy: 0.5129 (+- 0.0031)
# Overall Accuracy: 0.5147 (+- 0.0038)
# Overall Accuracy: 0.5141 (+- 0.0031)

# Overall Accuracy: 0.5130 (+- 0.0023)

# Przetestowano kilka losowych seedow. Wnioski: Feature engineering nieznacznie ale poprawia wyniki.


# Add Pair Trading Features
add_pair_features(x_train, shift=1)
add_pair_features(x_test, shift=1)
add_pair_features(x_train, shift=2)
add_pair_features(x_test, shift=2)
add_pair_features(x_train, shift=3)
add_pair_features(x_test, shift=3)

test_predictions = random_forest_classifier(x_train, y_train, x_test)
# Z reguly psuje wynik :(


# To do:
# - Hyperparameter tuning
# - LightGBM, XGBOost
# - Pca
# - wiecej feature engineering :
#   Convert returns to Ranks (0 to 1) per day.
#    * Instead of asking "Did stock go up?", ask "Did stock perform better than 80% of other stocks today?".
#    * This makes your data Stationary (always 0 to 1) and removes "Market Beta" (general market crashes won't confuse the model).

    