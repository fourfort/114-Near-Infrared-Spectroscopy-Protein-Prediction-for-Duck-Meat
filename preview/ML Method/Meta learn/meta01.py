import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter
from scipy.stats import chi2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import pairwise_distances
from datetime import datetime

# 定義統一參數
PARAMS = {
    'WAVELENGTH_RANGE': (750, 1099.5),
    'WAVELENGTH_RANGE_STR': '750-1099.5',
    'MCS_N_ITERATIONS': 1000,
    'MCS_N_COMPONENTS': 7,
    'MAHALANOBIS_THRESHOLD': 65,  # 進一步降低閾值
    'TRAIN_TEST_RATIO': 0.2,
    'TRAIN_TEST_RATIO_STR': '80:20',
    'VAL_RATIO': 0.3,  # 增加驗證集比例
    'DT_MAX_DEPTH_RANGE': [3, 5, 7, 10, 15, 20],
    'DT_MIN_SAMPLES_SPLIT_RANGE': [2, 5, 10, 20],
    'RF_N_ESTIMATORS': [50, 100, 200, 300],
    'SVM_C_RANGE': [0.01, 0.1, 1, 10, 100],
    'SVM_GAMMA': ['scale', 'auto', 0.01, 0.1, 1],
    'PLS_N_COMPONENTS': [3, 5, 10, 15, 20],
    'META_DT_MAX_DEPTH_RANGE': [3, 5, 7, 10, 15, 20, None],
    'META_DT_MIN_SAMPLES_SPLIT_RANGE': [2, 5, 10, 20],
    'META_DT_MIN_SAMPLES_LEAF_RANGE': [1, 2, 5]
}

# 設置 Times New Roman 字體
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# 自定義 R² 計算函數
def safe_r2_score(y_true, y_pred):
    if len(y_true) <= 1 or np.var(y_true) == 0:
        return 0.0
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

# 馬氏距離異常值檢測
def mahalanobis_distance(X):
    if X.shape[0] == 0:
        print("❌ 輸入數據為空，無法計算馬氏距離")
        return np.array([])
    cov_matrix = np.cov(X, rowvar=False)
    cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6
    try:
        cov_inv = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-3
        cov_inv = np.linalg.inv(cov_matrix)
    mean = np.mean(X, axis=0)
    md = np.sqrt(np.sum((X - mean) @ cov_inv * (X - mean), axis=1))
    return md

# 蒙特卡洛異常值檢測
def monte_carlo_outlier_detection(X, y, n_iterations=PARAMS['MCS_N_ITERATIONS'], 
                                 sample_ratio=0.8, n_components=PARAMS['MCS_N_COMPONENTS'], 
                                 threshold_percentile=95):
    if X.shape[0] == 0:
        print("❌ 輸入數據為空，無法執行蒙特卡洛檢測")
        return np.array([])
    if X.shape[0] != y.shape[0]:
        print(f"❌ X 和 y 形狀不匹配：X: {X.shape[0]}, y: {y.shape[0]}")
        return np.ones(X.shape[0], dtype=bool)
    n_samples = X.shape[0]
    n_subset = max(2, int(n_samples * sample_ratio))
    if n_subset >= n_samples:
        n_subset = n_samples - 1
    residuals = np.zeros(n_samples)
    
    for _ in range(n_iterations):
        indices = np.random.choice(n_samples, n_subset, replace=False)
        X_subset = X.iloc[indices] if isinstance(X, pd.DataFrame) else X[indices]
        y_subset = y[indices]
        
        scaler = StandardScaler()
        X_subset_scaled = scaler.fit_transform(X_subset)
        
        dt = DecisionTreeRegressor(max_depth=5, min_samples_split=5, random_state=42)
        dt.fit(X_subset_scaled, y_subset)
        
        X_scaled = scaler.transform(X)
        y_pred = dt.predict(X_scaled).ravel()
        residuals += (y - y_pred) ** 2
    
    residuals = residuals / n_iterations
    threshold = np.percentile(residuals, threshold_percentile)
    mask_mcs = residuals < threshold
    print(f"✅ MCS 檢測到 {np.sum(~mask_mcs)} 個異常值，剩餘樣本數：{np.sum(mask_mcs)}")
    return mask_mcs

# === 步驟 1: 數據獲取與處理 ===
try:
    nir_df = pd.read_csv(r'D:\Users\wedke\document\extra document\大三上\113專題研究之製作\論文人生\紅外線\red_coding main\preview\ML Method\nir_data.csv')
    print("✅ nir_data.csv 前五行:")
    print(nir_df.head())
except FileNotFoundError:
    print("❌ 無法找到 nir_data.csv，請確認路徑。")
    exit()

try:
    trad_df = pd.read_excel(r'D:\Users\wedke\document\extra document\大三上\113專題研究之製作\論文人生\紅外線\red_coding main\preview\trad_data.xlsx')
    print("\n✅ trad_data.xlsx 前五行:")
    print(trad_df.head())
except FileNotFoundError:
    print("❌ 無法找到 trad_data.xlsx，請確認路徑。")
    exit()

# 定義目標成分
components = ['Moisture', 'Protein', 'Fat', 'SFA', 'Ash', 'Collagen', 'Salt']

# 檢查缺失值並移除 Sample Number 缺失的樣本
print("\n🔍 NIR 缺失值檢查:\n", nir_df.isnull().sum())
print("\n🔍 傳統分析缺失值檢查:\n", trad_df.isnull().sum())

# 自動偵測樣本編號欄位名稱
def find_sample_column(df, name='nir'):
    for col in df.columns:
        if 'sample number' in col.lower() or 'sample no' in col.lower():
            print(f"✅ 在 {name} 資料中找到樣本編號欄位: {col}")
            return col
    print(f"⚠️ {name} 資料中未找到樣本編號欄位")
    return None

nir_sample_col = find_sample_column(nir_df, 'NIR')
trad_sample_col = find_sample_column(trad_df, '傳統分析')

# 若 NIR 缺少樣本編號欄，生成編號
if not nir_sample_col:
    nir_sample_col = 'Sample_Number'
    nir_df[nir_sample_col] = [f'1-{i+1}' for i in range(len(nir_df))]

# 處理光譜資料（波長範圍）
nir_df.columns = ['Wavelength'] + [f'1-{i}' for i in range(1, len(nir_df.columns))]
nir_df = nir_df.drop(0).reset_index(drop=True)
nir_df['Wavelength'] = nir_df['Wavelength'].astype(float)
nir_filtered_df = nir_df[(nir_df['Wavelength'] >= PARAMS['WAVELENGTH_RANGE'][0]) & 
                         (nir_df['Wavelength'] <= PARAMS['WAVELENGTH_RANGE'][1])]
selected_columns = ['Wavelength'] + [col for col in nir_df.columns if col != 'Wavelength']
nir_final_df = nir_filtered_df[selected_columns]
print(f"✅ 過濾後 NIR 數據波長範圍: {nir_final_df['Wavelength'].min()} - {nir_final_df['Wavelength'].max()}, 共 {len(nir_final_df)} 個波長點")
if len(nir_final_df) != 700:
    print(f"⚠️ 波長點數 {len(nir_final_df)} 不等於預期 700，檢查數據格式")

# 對齊樣本
if trad_sample_col:
    trad_df[trad_sample_col] = [f'1-{i+1}' for i in range(len(trad_df))]
else:
    print("⚠️ 無法進行樣本對齊，缺少樣本編號欄")
    exit()

# 提取多成分目標值
trad_df = trad_df.dropna(subset=components)
Y_multi = trad_df[components].values
sample_ids = trad_df[trad_sample_col].astype(str).tolist()
nir_transposed_df = nir_final_df.set_index('Wavelength').T
nir_transposed_df = nir_transposed_df.apply(pd.to_numeric, errors='coerce')
nir_transposed_df = nir_transposed_df.dropna(axis=1)
print("\n✅ 轉換後 NIR 資料形狀:", nir_transposed_df.shape)

nir_transposed_df.index = nir_transposed_df.index.astype(str)
common_ids = list(set(sample_ids).intersection(set(nir_transposed_df.index)))
if len(common_ids) != len(sample_ids) or len(common_ids) != len(Y_multi):
    print(f"❌ 對齊後樣本數不一致：X: {len(common_ids)}, Y: {len(Y_multi)}")
    trad_df = trad_df[trad_df[trad_sample_col].astype(str).isin(common_ids)]
    Y_multi = trad_df[components].values
    X = nir_transposed_df.loc[common_ids]
else:
    X = nir_transposed_df.loc[sample_ids]
print(f"✅ 成功對齊資料，樣本數: {X.shape[0]}")

# 異常值檢測
pca = PCA(n_components=min(X.shape[0], X.shape[1], 20))
X_pca = pca.fit_transform(X)
md = mahalanobis_distance(X_pca)
if len(md) == 0:
    print("❌ 無法計算馬氏距離，跳過異常值檢測")
    mask_md = np.ones(X.shape[0], dtype=bool)
else:
    md_threshold = np.percentile(md, PARAMS['MAHALANOBIS_THRESHOLD'])
    mask_md = md < md_threshold
    print(f"✅ 馬氏距離法檢測到 {np.sum(~mask_md)} 個異常值")

X = X[mask_md]
Y_multi = Y_multi[mask_md]
print(f"✅ 馬氏距離過濾後樣本數：{X.shape[0]}")
if X.shape[0] == 0:
    print("❌ 馬氏距離過濾後數據為空，無法繼續處理")
    exit()

scaler_y_temp = StandardScaler()
Y_multi_scaled_temp = scaler_y_temp.fit_transform(Y_multi)
Y_mean_scaled = np.mean(Y_multi_scaled_temp, axis=1)
mask_mcs = monte_carlo_outlier_detection(X, Y_mean_scaled)
if len(mask_mcs) == 0:
    print("❌ 蒙特卡洛檢測失敗，跳過異常值檢測")
    mask_mcs = np.ones(X.shape[0], dtype=bool)
X = X[mask_mcs]
Y_multi = Y_multi[mask_mcs]
print(f"✅ 結合馬氏距離和 MCS 後，剩餘樣本數：{X.shape[0]}")

if X.shape[0] == 0:
    print("❌ 異常值檢測後數據為空，無法繼續處理")
    exit()

# === 步驟 2: 數據集分配（SPXY） ===
def spxy(X, Y, test_size=PARAMS['TRAIN_TEST_RATIO'], val_size=PARAMS['VAL_RATIO']):
    X = np.array(X)
    Y = np.array(Y)
    n_samples = X.shape[0]
    n_test = int(np.floor(test_size * n_samples))
    n_val = int(np.floor(val_size * (n_samples - n_test)))
    if n_test == 0 or n_val == 0:
        print(f"❌ 樣本數不足以分割：n_samples={n_samples}, n_test={n_test}, n_val={n_val}")
        exit()

    dist_X = pairwise_distances(X)
    dist_Y = pairwise_distances(Y)
    dist_XY = dist_X + dist_Y / np.max(dist_Y)

    selected = []
    validation = []
    remaining = list(range(n_samples))

    i1, i2 = np.unravel_index(np.argmax(dist_XY), dist_XY.shape)
    selected.extend([i1, i2])
    remaining.remove(i1)
    remaining.remove(i2)

    while len(selected) < n_samples - n_test - n_val:
        min_distances = np.min(dist_XY[remaining][:, selected], axis=1)
        next_index = remaining[np.argmax(min_distances)]
        selected.append(next_index)
        remaining.remove(next_index)

    for _ in range(n_val):
        min_distances = np.min(dist_XY[remaining][:, selected], axis=1)
        next_index = remaining[np.argmax(min_distances)]
        validation.append(next_index)
        remaining.remove(next_index)

    selected = np.array(selected)
    validation = np.array(validation)
    test = np.array(remaining)

    X_train = X[selected]
    Y_train = Y[selected]
    X_val = X[validation]
    Y_val = Y[validation]
    X_test = X[test]
    Y_test = Y[test]

    return X_train, X_val, X_test, Y_train, Y_val, Y_test

scaler_x = StandardScaler()
X_scaled = scaler_x.fit_transform(X)
X = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
scaler_y = StandardScaler()
Y_multi_scaled = scaler_y.fit_transform(Y_multi)

X_train, X_val, X_test, Y_train_multi, Y_val_multi, Y_test_multi = spxy(X, Y_multi_scaled)
print(f"✅ SPXY 訓練集形狀: {X_train.shape}, 驗證集形狀: {X_val.shape}, 測試集形狀: {X_test.shape}")
for i, component in enumerate(components):
    print(f"✅ 訓練集 {component} 描述:", pd.Series(scaler_y.inverse_transform(Y_train_multi)[:, i]).describe())
    print(f"✅ 驗證集 {component} 描述:", pd.Series(scaler_y.inverse_transform(Y_val_multi)[:, i]).describe())
    print(f"✅ 測試集 {component} 描述:", pd.Series(scaler_y.inverse_transform(Y_test_multi)[:, i]).describe())

# === 步驟 3: 訓練基模型 ===
X_train_sg = savgol_filter(X_train, window_length=51, polyorder=3, axis=1)
X_val_sg = savgol_filter(X_val, window_length=51, polyorder=3, axis=1)
X_test_sg = savgol_filter(X_test, window_length=51, polyorder=3, axis=1)
print("✅ Savitzky-Golay 濾波完成")

base_models = {
    'RandomForest': {
        'model': MultiOutputRegressor(RandomForestRegressor(random_state=42)),
        'param_grid': {
            'estimator__n_estimators': PARAMS['RF_N_ESTIMATORS'],
            'estimator__max_depth': PARAMS['DT_MAX_DEPTH_RANGE']
        }
    },
    'SVM': {
        'model': MultiOutputRegressor(SVR()),
        'param_grid': {
            'estimator__C': PARAMS['SVM_C_RANGE'],
            'estimator__gamma': PARAMS['SVM_GAMMA']
        }
    },
    'PLS': {
        'model': PLSRegression(),
        'param_grid': {
            'n_components': PARAMS['PLS_N_COMPONENTS']
        }
    }
}

# 儲存基模型在驗證集和測試集上的預測
base_model_predictions = {name: {'val': None, 'test': None} for name in base_models}
base_model_metrics = {name: {comp: {'val_r2': None, 'test_r2': None} for comp in components} for name in base_models}

best_models = {}
for name, config in base_models.items():
    print(f"\n✅ 訓練基模型: {name}")
    grid_search = RandomizedSearchCV(config['model'], config['param_grid'], n_iter=20,
                                    cv=KFold(n_splits=5, shuffle=True, random_state=42),
                                    scoring='r2', n_jobs=-1, refit=True, random_state=42)
    grid_search.fit(X_train_sg, Y_train_multi)
    best_models[name] = grid_search.best_estimator_
    print(f"✅ {name} 最佳參數: {grid_search.best_params_}")
    
    # 計算基模型在驗證集和測試集上的性能
    val_preds = best_models[name].predict(X_val_sg)
    test_preds = best_models[name].predict(X_test_sg)
    base_model_predictions[name]['val'] = val_preds
    base_model_predictions[name]['test'] = test_preds
    
    for i, comp in enumerate(components):
        val_r2 = safe_r2_score(scaler_y.inverse_transform(Y_val_multi)[:, i], scaler_y.inverse_transform(val_preds)[:, i])
        test_r2 = safe_r2_score(scaler_y.inverse_transform(Y_test_multi)[:, i], scaler_y.inverse_transform(test_preds)[:, i])
        base_model_metrics[name][comp]['val_r2'] = val_r2
        base_model_metrics[name][comp]['test_r2'] = test_r2
        print(f"✅ {name} - {comp} Val R²: {val_r2:.4f}, Test R²: {test_r2:.4f}")

# === 步驟 4: 生成 Meta Data ===
# 僅使用隨機森林、SVM 和 PLS 的預測值
meta_features_val = np.hstack([base_model_predictions[name]['val'] for name in base_models])
meta_features_test = np.hstack([base_model_predictions[name]['test'] for name in base_models])
print(f"✅ 驗證集 Meta Features 形狀: {meta_features_val.shape}")
print(f"✅ 測試集 Meta Features 形狀: {meta_features_test.shape}")

# 可視化 meta features 相關性
plt.figure(figsize=(10, 8))
sns.heatmap(pd.DataFrame(meta_features_val).corr(), annot=False, cmap='coolwarm')
plt.title('Correlation Heatmap of Meta Features')
plt.show()

# === 步驟 5: 訓練元學習器（決策樹） ===
meta_dt = DecisionTreeRegressor(random_state=42)
meta_param_grid = {
    'max_depth': PARAMS['META_DT_MAX_DEPTH_RANGE'],
    'min_samples_split': PARAMS['META_DT_MIN_SAMPLES_SPLIT_RANGE'],
    'min_samples_leaf': PARAMS['META_DT_MIN_SAMPLES_LEAF_RANGE']
}
meta_grid_search = RandomizedSearchCV(meta_dt, meta_param_grid, n_iter=20,
                                     cv=KFold(n_splits=10, shuffle=True, random_state=42),
                                     scoring='r2', n_jobs=-1, refit=True, random_state=42)
meta_grid_search.fit(meta_features_val, Y_val_multi)
best_meta_dt = meta_grid_search.best_estimator_
print(f"✅ 元學習器最佳參數: {meta_grid_search.best_params_}")

# 輸出特徵重要性
feature_names = [f"{name}_{comp}" for name in base_models for comp in components]
feature_importances = pd.DataFrame({
    'Feature': feature_names,
    'Importance': best_meta_dt.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("\n✅ 元學習器特徵重要性:")
print(feature_importances)

# === 步驟 6: 測試與評估 ===
Y_train_pred_multi = best_meta_dt.predict(meta_features_val)
Y_test_pred_multi = best_meta_dt.predict(meta_features_test)
Y_train_pred = scaler_y.inverse_transform(Y_train_pred_multi)
Y_test_pred = scaler_y.inverse_transform(Y_test_pred_multi)
Y_train_multi_original = scaler_y.inverse_transform(Y_val_multi)
Y_test_multi_original = scaler_y.inverse_transform(Y_test_multi)

# 計算 SEC 和 SEP
n_train = X_val.shape[0]
n_test = X_test.shape[0]
n_components = Y_multi.shape[1]
actual_components = components[:n_components]
print(f"✅ 實際檢測成分數: {n_components}, 成分列表: {actual_components}")

# 評估每個成分
for i, component in enumerate(actual_components):
    train_mse = mean_squared_error(Y_train_pred[:, i], Y_train_multi_original[:, i])
    test_mse = mean_squared_error(Y_test_pred[:, i], Y_test_multi_original[:, i])
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    train_r2 = safe_r2_score(Y_train_pred[:, i], Y_train_multi_original[:, i])
    test_r2 = safe_r2_score(Y_test_pred[:, i], Y_test_multi_original[:, i])
    sec = np.sqrt(np.sum((Y_train_multi_original[:, i] - Y_train_pred[:, i]) ** 2) / max(1, n_train - 1))
    sep = np.sqrt(np.sum((Y_test_multi_original[:, i] - Y_test_pred[:, i]) ** 2) / (n_test - 1))
    print(f"✅ {component} - Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
    print(f"✅ {component} - Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
    print(f"✅ {component} - SEC: {sec:.4f}, SEP: {sep:.4f}")

# 10-fold 交叉驗證評估
cv_scores = cross_val_score(best_meta_dt, meta_features_val, Y_val_multi, cv=10, scoring='r2')
print(f"\n✅ 10-fold Cross-Validation Mean R²: {cv_scores.mean():.4f} (± {cv_scores.std() * 2:.4f})")

# 指標表格數據
metrics_data_rmse = {
    'Component': actual_components + actual_components,
    'Metric': ['RMSE'] * n_components + ['RMSE'] * n_components,
    'Training Set': [f'{np.sqrt(mean_squared_error(Y_train_pred[:, i], Y_train_multi_original[:, i])):.4f}' for i in range(n_components)] + ['N/A'] * n_components,
    'Test Set': ['N/A'] * n_components + [f'{np.sqrt(mean_squared_error(Y_test_pred[:, i], Y_test_multi_original[:, i])):.4f}' for i in range(n_components)]
}
metrics_df_rmse = pd.DataFrame(metrics_data_rmse)
print("\n✅ RMSE Evaluation Metrics:")
print(metrics_df_rmse.to_string(index=False))

metrics_data_se = {
    'Component': actual_components + actual_components,
    'Metric': ['SE'] * n_components + ['SE'] * n_components,
    'Training Set': [f'{np.sqrt(np.sum((Y_train_multi_original[:, i] - Y_train_pred[:, i]) ** 2) / max(1, n_train - 1)):.4f}' for i in range(n_components)] + ['N/A'] * n_components,
    'Test Set': ['N/A'] * n_components + [f'{np.sqrt(np.sum((Y_test_multi_original[:, i] - Y_test_pred[:, i]) ** 2) / (n_test - 1)):.4f}' for i in range(n_components)]
}
metrics_df_se = pd.DataFrame(metrics_data_se)
print("\n✅ SE Evaluation Metrics:")
print(metrics_df_se.to_string(index=False))

metrics_data_r2 = {
    'Component': actual_components + actual_components,
    'Metric': ['R² Score'] * n_components + ['R² Score'] * n_components,
    'Training Set': [f'{safe_r2_score(Y_train_pred[:, i], Y_train_multi_original[:, i]):.4f}' for i in range(n_components)] + ['N/A'] * n_components,
    'Test Set': ['N/A'] * n_components + [f'{safe_r2_score(Y_test_pred[:, i], Y_test_multi_original[:, i]):.4f}' for i in range(n_components)]
}
metrics_df_r2 = pd.DataFrame(metrics_data_r2)
print("\n✅ R² Score Evaluation Metrics:")
print(metrics_df_r2.to_string(index=False))

# 繪製表格
fig_rmse = plt.figure(figsize=(12, 4 * n_components / 7))
ax1 = fig_rmse.add_subplot(111)
ax1.axis('off')
table_rmse = ax1.table(cellText=metrics_df_rmse.values, colLabels=metrics_df_rmse.columns, cellLoc='center', loc='center')
table_rmse.auto_set_font_size(False)
table_rmse.set_fontsize(10)
table_rmse.scale(1.2, 1.2)
ax1.set_title('RMSE Evaluation Metrics')
plt.show()

fig_se = plt.figure(figsize=(12, 4 * n_components / 7))
ax2 = fig_se.add_subplot(111)
ax2.axis('off')
table_se = ax2.table(cellText=metrics_df_se.values, colLabels=metrics_df_se.columns, cellLoc='center', loc='center')
table_se.auto_set_font_size(False)
table_se.set_fontsize(10)
table_se.scale(1.2, 1.2)
ax2.set_title('SE Evaluation Metrics')
plt.show()

fig_r2 = plt.figure(figsize=(12, 4 * n_components / 7))
ax3 = fig_r2.add_subplot(111)
ax3.axis('off')
table_r2 = ax3.table(cellText=metrics_df_r2.values, colLabels=metrics_df_r2.columns, cellLoc='center', loc='center')
table_r2.auto_set_font_size(False)
table_r2.set_fontsize(10)
table_r2.scale(1.2, 1.2)
ax3.set_title('R² Score Evaluation Metrics')
plt.show()

# 繪製基模型與元學習器的比較散點圖
for i, component in enumerate(actual_components):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(Y_test_multi_original[:, i], Y_test_pred[:, i], alpha=0.7, color='blue', label='Meta-Learner (DT)')
    for name in base_models:
        test_preds = scaler_y.inverse_transform(base_model_predictions[name]['test'])
        ax.scatter(Y_test_multi_original[:, i], test_preds[:, i], alpha=0.5, label=f'Base Model ({name})')
    ax.plot([min(Y_test_multi_original[:, i]), max(Y_test_multi_original[:, i])],
            [min(Y_test_multi_original[:, i]), max(Y_test_multi_original[:, i])], 'r--', label='Ideal (y=x)')
    ax.set_xlabel(f'Actual {component} Content (%)')
    ax.set_ylabel(f'Predicted {component} Content (%)')
    ax.set_title(f'Actual vs Predicted {component}: Meta-Learner vs Base Models')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# === 步驟 7: 生成參數和指標的 Excel 檔案 ===
common_params = {
    'Wavelength Range (nm)': PARAMS['WAVELENGTH_RANGE_STR'],
    'Train-Test Split Ratio': PARAMS['TRAIN_TEST_RATIO_STR'],
    'Validation Set Ratio': PARAMS['VAL_RATIO']
}

specific_params = {}
for name, model in best_models.items():
    if name == 'RandomForest':
        specific_params['RF Optimal N Estimators'] = model.estimators_[0].n_estimators
        specific_params['RF Optimal Max Depth'] = model.estimators_[0].max_depth
    elif name == 'SVM':
        specific_params['SVM Optimal C'] = model.estimators_[0].C
        specific_params['SVM Optimal Gamma'] = model.estimators_[0].gamma
    elif name == 'PLS':
        specific_params['PLS Optimal N Components'] = model.n_components

specific_params['Meta DT Optimal Max Depth'] = best_meta_dt.max_depth
specific_params['Meta DT Optimal Min Samples Split'] = best_meta_dt.min_samples_split
specific_params['Meta DT Optimal Min Samples Leaf'] = best_meta_dt.min_samples_leaf

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_filename = f'Meta_Learning_DT_Optimized_{timestamp}.xlsx'
with pd.ExcelWriter(output_filename) as writer:
    for component in components:
        eval_metrics = {}
        if component in actual_components:
            i = actual_components.index(component)
            eval_metrics[f'{component}_RMSEC'] = f'{np.sqrt(mean_squared_error(Y_train_pred[:, i], Y_train_multi_original[:, i])):.4f}'
            eval_metrics[f'{component}_RMSEP'] = f'{np.sqrt(mean_squared_error(Y_test_pred[:, i], Y_test_multi_original[:, i])):.4f}'
            eval_metrics[f'{component}_R²_C'] = f'{safe_r2_score(Y_train_pred[:, i], Y_train_multi_original[:, i]):.4f}'
            eval_metrics[f'{component}_R²_P'] = f'{safe_r2_score(Y_test_pred[:, i], Y_test_multi_original[:, i]):.4f}'
            eval_metrics[f'{component}_SEC'] = f'{np.sqrt(np.sum((Y_train_multi_original[:, i] - Y_train_pred[:, i]) ** 2) / max(1, n_train - 1)):.4f}'
            eval_metrics[f'{component}_SEP'] = f'{np.sqrt(np.sum((Y_test_multi_original[:, i] - Y_test_pred[:, i]) ** 2) / (n_test - 1)):.4f}'
            for name in base_models:
                eval_metrics[f'{component}_{name}_Val_R²'] = f'{base_model_metrics[name][component]["val_r2"]:.4f}'
                eval_metrics[f'{component}_{name}_Test_R²'] = f'{base_model_metrics[name][component]["test_r2"]:.4f}'
        else:
            eval_metrics = {f'{component}_RMSEC': 'N/A', f'{component}_RMSEP': 'N/A', f'{component}_R²_C': 'N/A',
                           f'{component}_R²_P': 'N/A', f'{component}_SEC': 'N/A', f'{component}_SEP': 'N/A'}
            for name in base_models:
                eval_metrics[f'{component}_{name}_Val_R²'] = 'N/A'
                eval_metrics[f'{component}_{name}_Test_R²'] = 'N/A'

        output_data = {
            'Parameter/Metric': list(common_params.keys()) + list(specific_params.keys()) + list(eval_metrics.keys()),
            'Value': list(common_params.values()) + list(specific_params.values()) + list(eval_metrics.values()),
            'Category': ['Common Parameters'] * len(common_params) + ['Specific Parameters'] * len(specific_params) + ['Evaluation Metrics'] * len(eval_metrics)
        }

        output_df = pd.DataFrame(output_data)
        output_df.to_excel(writer, sheet_name=component, index=False)

print(f"✅ 已生成參數和指標檔案: {output_filename}")