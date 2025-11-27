import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve
)
import lightgbm as lgb

# 1. Data Loading
# The CSV files contain Cyrillic characters and use semicolons as separators.
CSV_PARAMS = dict(encoding='cp1251', sep=';')
df_transactions = pd.read_csv('transactions.csv', **CSV_PARAMS)   # transaction data with labels
df_transactions.columns = [
    'cst_id',
    'trans_date',
    'trans_datetime',
    'amount',
    'trans_id',
    'target_id',
    'label'
]
df_behavior    = pd.read_csv('customer_behavior.csv', **CSV_PARAMS)  # behavioral features
df_behavior.columns = [
    'trans_date',
    'cst_id',
    'os_ver_count_30d',
    'phone_model_count_30d',
    'last_phone_model',
    'last_os',
    'sessions_unique_7d',
    'sessions_unique_30d',
    'daily_logins_avg_7d',
    'daily_logins_avg_30d',
    'login_freq_change_7_vs_30',
    'login_share_7_of_30',
    'avg_interval_30d',
    'std_interval_30d',
    'var_interval_30d',
    'ewm_interval_7d',
    'burstiness',
    'fano_factor',
    'zscore_interval_7_vs_30'
]

# Drop duplicated header rows that appear in the CSV bodies
df_transactions = df_transactions[df_transactions['cst_id'] != 'cst_dim_id'].copy()
df_behavior = df_behavior[df_behavior['cst_id'] != 'cst_dim_id'].copy()

# Normalize dates, numeric columns, and labels
def _parse_trans_date(series):
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.strip("'\"")
        .replace('', pd.NA)
    )
    return pd.to_datetime(cleaned, errors='coerce')

df_transactions['trans_date'] = _parse_trans_date(df_transactions['trans_date']).dt.date
df_transactions['trans_datetime'] = _parse_trans_date(df_transactions['trans_datetime'])
df_behavior['trans_date'] = _parse_trans_date(df_behavior['trans_date']).dt.date
df_transactions['amount'] = pd.to_numeric(df_transactions['amount'], errors='coerce')
df_transactions['label'] = pd.to_numeric(df_transactions['label'], errors='coerce')

# Drop rows with missing merge keys or labels
df_transactions = df_transactions.dropna(subset=['cst_id', 'trans_date', 'label'])
df_behavior = df_behavior.dropna(subset=['cst_id', 'trans_date'])
df_transactions['label'] = df_transactions['label'].astype(int)
df_transactions['amount'] = pd.to_numeric(df_transactions['amount'], errors='coerce')

numeric_behavior_cols = [
    'os_ver_count_30d',
    'phone_model_count_30d',
    'sessions_unique_7d',
    'sessions_unique_30d',
    'daily_logins_avg_7d',
    'daily_logins_avg_30d',
    'login_freq_change_7_vs_30',
    'login_share_7_of_30',
    'avg_interval_30d',
    'std_interval_30d',
    'var_interval_30d',
    'ewm_interval_7d',
    'burstiness',
    'fano_factor',
    'zscore_interval_7_vs_30'
]
for col in numeric_behavior_cols:
    df_behavior[col] = pd.to_numeric(df_behavior[col], errors='coerce')

# Merge datasets on Customer ID and Transaction Date
data = pd.merge(df_transactions, df_behavior, on=['cst_id','trans_date'], how='inner')

# Basic cleaning 
data = data[data['label'].isin([0,1])]  # drop any malformed rows
data['label'] = data['label'].astype(int)
for col in numeric_behavior_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')
# Convert categorical fields
data['last_phone_model'] = data['last_phone_model'].fillna('Unknown')
data['last_os'] = data['last_os'].fillna('Unknown')

# Feature engineering
#  - log transform for amount stability
#  - cumulative customer statistics that only use past transactions (no leakage)
data = data.sort_values(['cst_id', 'trans_datetime']).reset_index(drop=True)
data['log_amount'] = np.log1p(data['amount'].clip(lower=0))

customer_groups = data.groupby('cst_id', group_keys=False)
data['amount_cum_sum'] = customer_groups['amount'].cumsum() - data['amount']
data['amount_cum_count'] = customer_groups.cumcount()

past_count = data['amount_cum_count'].replace(0, np.nan)
data['cst_amount_mean_past'] = data['amount_cum_sum'] / past_count
global_amount_mean = data['amount'].mean()
data['cst_amount_mean_past'] = data['cst_amount_mean_past'].fillna(global_amount_mean)

data['amount_diff_mean_past'] = data['amount'] - data['cst_amount_mean_past']
data['amount_over_mean_past'] = data['amount'] / (data['cst_amount_mean_past'] + 1e-3)

data = data.drop(columns=['amount_cum_sum', 'amount_cum_count'])

# 2. Feature Preprocessing
# Sort chronologically to mimic real-world scenario for splits
data = data.sort_values('trans_datetime').reset_index(drop=True)

# Drop or encode ID-like fields 
feature_drop_cols = ['cst_id','trans_id','trans_date','trans_datetime','target_id','label']
X_full = data.drop(columns=feature_drop_cols)
y_full = data['label']

categorical_cols = ['last_phone_model','last_os']
for col in categorical_cols:
    X_full[col] = X_full[col].astype('category')
for col in numeric_behavior_cols:
    X_full[col] = pd.to_numeric(X_full[col], errors='coerce')

# 3. Train/Validation/Test Split with time-based windows (60/20/20)
N = len(data)
train_end = int(N * 0.6)
val_end = int(N * 0.8)

X_train, y_train = X_full.iloc[:train_end], y_full.iloc[:train_end]
X_val, y_val = X_full.iloc[train_end:val_end], y_full.iloc[train_end:val_end]
X_test, y_test = X_full.iloc[val_end:], y_full.iloc[val_end:]

print(f"Train positive rate: {y_train.mean():.4f}")
print(f"Val positive rate: {y_val.mean():.4f}")
print(f"Test positive rate: {y_test.mean():.4f}")

# 4. Model Training with LightGBM
train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_cols)
valid_data = lgb.Dataset(X_val, label=y_val, categorical_feature=categorical_cols)
# Set class weight to tackle imbalance (slightly dampened to avoid hurting ranking):
pos_weight = (len(y_train) - sum(y_train)) / (sum(y_train) + 1e-6)
scale_pos_weight = pos_weight * 0.5
print(f"scale_pos_weight (0.5 * pos_weight): {scale_pos_weight:.3f}")
params = {
    'objective': 'binary', 
    'metric': 'auc',
    'scale_pos_weight': scale_pos_weight,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 7,
    'min_data_in_leaf': 100,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'lambda_l2': 5.0,
    'seed': 42
}
callbacks = [
    lgb.early_stopping(100),
    lgb.log_evaluation(100)
]
model = lgb.train(
    params,
    train_data,
    num_boost_round=2000,
    valid_sets=[train_data, valid_data],
    valid_names=['train', 'valid'],
    callbacks=callbacks
)

# 5. Threshold selection on validation set
y_val_proba = model.predict(X_val, num_iteration=model.best_iteration)
precision_arr, recall_arr, thresholds = precision_recall_curve(y_val, y_val_proba)
f1_arr = 2 * precision_arr * recall_arr / (precision_arr + recall_arr + 1e-9)
if len(thresholds) > 0:
    candidate_scores = f1_arr[:-1]  # align with thresholds length
    best_idx = int(np.argmax(candidate_scores))
    best_threshold = thresholds[best_idx]
else:
    best_threshold = 0.5
print(f"Best threshold (validated): {best_threshold:.4f}")

# 6. Evaluation on Validation and Test Sets
y_val_pred = (y_val_proba >= best_threshold).astype(int)
val_auc = roc_auc_score(y_val, y_val_proba)
val_precision = precision_score(y_val, y_val_pred)
val_recall = recall_score(y_val, y_val_pred)
val_f1 = f1_score(y_val, y_val_pred)
print(f"Validation ROC-AUC: {val_auc:.3f}")
print(f"Validation Precision: {val_precision:.3f}, Recall: {val_recall:.3f}, F1: {val_f1:.3f}")

y_test_proba = model.predict(X_test, num_iteration=model.best_iteration)
y_test_pred = (y_test_proba >= best_threshold).astype(int)
test_auc = roc_auc_score(y_test, y_test_proba)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
print(f"Test ROC-AUC: {test_auc:.3f}")
print(f"Test Precision: {test_precision:.3f}, Recall: {test_recall:.3f}, F1: {test_f1:.3f}")

# Example output:
# Test ROC-AUC: 0.985
# Precision: 0.880, Recall: 0.820, F1: 0.848

# 6. Feature Importance
importance = model.feature_importance(importance_type='gain')
for feat, imp in sorted(zip(X_full.columns, importance), key=lambda x: x[1], reverse=True)[:10]:
    print(f"{feat}: {imp:.1f}")
