import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import os
import results_logger

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
    col_name = f'RET_{shift}'
    if col_name in df.columns:
        market_mean = df.groupby('DATE')[col_name].transform('mean')
        df[f'Spread_Market_{shift}'] = df[col_name] - market_mean

def random_forest_classifier(x_train, y_train, x_test, n_splits=5, description="Initial Run"):
    print(f"Training Random Forest Classifier ({description})")
    
    train_dates = x_train['DATE'].unique()
    kf = KFold(n_splits=n_splits, shuffle=True)
    
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
    
    results_logger.log_results("RandomForest_Baseline", scores, mean_acc, std_acc, description)
    
    return final_test_preds

selected_features = ["DATE", "SECTOR", "RET_1", "VOLUME_1", "RET_2", "VOLUME_2", "RET_3", "VOLUME_3", "RET_4", "VOLUME_4",
                     "RET_5", "VOLUME_5"]
x_train = x_train[selected_features]
x_test = x_test[selected_features]

test_predictions = random_forest_classifier(x_train, y_train, x_test, description="Initial Run")

feature_engineering(x_train)
feature_engineering(x_test)
feature_engineering(x_train, shift=2)
feature_engineering(x_test, shift=2)
feature_engineering(x_train, shift=3)
feature_engineering(x_test, shift=3)

test_predictions = random_forest_classifier(x_train, y_train, x_test, description="With Feature Engineering")

add_pair_features(x_train, shift=1)
add_pair_features(x_test, shift=1)
add_pair_features(x_train, shift=2)
add_pair_features(x_test, shift=2)
add_pair_features(x_train, shift=3)
add_pair_features(x_test, shift=3)

test_predictions = random_forest_classifier(x_train, y_train, x_test, description="With Pair Trading Features")

# Save submission
submission = pd.DataFrame({'ID': x_test.index, 'RET': test_predictions.astype(bool)})
submission.to_csv('submission_random_forest.csv', index=False)
print("Saved submission_random_forest.csv")