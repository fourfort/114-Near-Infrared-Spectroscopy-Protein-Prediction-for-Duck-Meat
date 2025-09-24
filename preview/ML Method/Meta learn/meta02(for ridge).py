import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter
from scipy.stats import chi2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import pairwise_distances
from datetime import datetime
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 定義統一參數
PARAMS = {
    'WAVELENGTH_RANGE': (700, 1099.5),
    'WAVELENGTH_RANGE_STR': '750-1099.5',
    'MCS_N_ITERATIONS': 1000,
    'MCS_N_COMPONENTS': 7,
    'MAHALANOBIS_THRESHOLD': 70,
    'TRAIN_TEST_RATIO': 0.2,
    'TRAIN_TEST_RATIO_STR': '80:20',
    'K_FOLDS': 10,
    'PCA_N_COMPONENTS': 5,
    'DT_MAX_DEPTH_RANGE': [3, 5, 7, 10, 15],
    'DT_MIN_SAMPLES_SPLIT_RANGE': [2, 5, 10],
    'RF_N_ESTIMATORS': [50, 100, 200],
    'SVM_C_RANGE': [0.01, 0.1, 1, 10, 100],
    'SVM_GAMMA': ['scale', 'auto', 0.01, 0.1],
    'PLS_N_COMPONENTS': [3, 5, 10, 15],
    'RIDGE_ALPHA': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
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
if len(nir_final_df) != 500:
    print(f"⚠️ 波長點數 {len(nir_final_df)} 不等於預期 500，檢查數據格式")

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
def spxy(X, Y, test_size=PARAMS['TRAIN_TEST_RATIO']):
    X = np.array(X)
    Y = np.array(Y)
    n_samples = X.shape[0]
    n_test = int(np.floor(test_size * n_samples))
    if n_test == 0:
        print(f"❌ 樣本數不足以分割：n_samples={n_samples}, n_test={n_test}")
        exit()

    dist_X = pairwise_distances(X)
    dist_Y = pairwise_distances(Y)
    dist_XY = dist_X + dist_Y / np.max(dist_Y)

    selected = []
    remaining = list(range(n_samples))

    i1, i2 = np.unravel_index(np.argmax(dist_XY), dist_XY.shape)
    selected.extend([i1, i2])
    remaining.remove(i1)
    remaining.remove(i2)

    while len(selected) < n_samples - n_test:
        min_distances = np.min(dist_XY[remaining][:, selected], axis=1)
        next_index = remaining[np.argmax(min_distances)]
        selected.append(next_index)
        remaining.remove(next_index)

    selected = np.array(selected)
    test = np.array(remaining)

    X_train = X[selected]
    Y_train = Y[selected]
    X_test = X[test]
    Y_test = Y[test]

    return X_train, X_test, Y_train, Y_test

scaler_x = StandardScaler()
X_scaled = scaler_x.fit_transform(X)
X = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
scaler_y = StandardScaler()
Y_multi_scaled = scaler_y.fit_transform(Y_multi)

X_train, X_test, Y_train_multi, Y_test_multi = spxy(X, Y_multi_scaled)
print(f"✅ SPXY 訓練集形狀: {X_train.shape}, 測試集形狀: {X_test.shape}")
for i, component in enumerate(components):
    print(f"✅ 訓練集 {component} 描述:", pd.Series(scaler_y.inverse_transform(Y_train_multi)[:, i]).describe())
    print(f"✅ 測試集 {component} 描述:", pd.Series(scaler_y.inverse_transform(Y_test_multi)[:, i]).describe())

# === 步驟 3: 訓練基模型並生成 OOF Meta Features ===
X_train_sg = savgol_filter(X_train, window_length=51, polyorder=3, axis=1)
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
    'PLS': {
        'model': PLSRegression(),
        'param_grid': {
            'n_components': PARAMS['PLS_N_COMPONENTS']
        }
    }
}

# 儲存基模型預測
kf = KFold(n_splits=PARAMS['K_FOLDS'], shuffle=True, random_state=42)
meta_features_train = np.zeros((X_train.shape[0], len(components) * len(base_models)))
meta_features_test = np.zeros((X_test.shape[0], len(components) * len(base_models)))
base_model_metrics = {name: {comp: {'train_r2': None, 'test_r2': None} for comp in components} for name in base_models}

best_models = {}
for name, config in base_models.items():
    print(f"\n✅ 訓練基模型: {name}")
    grid_search = RandomizedSearchCV(config['model'], config['param_grid'], n_iter=20,
                                    cv=KFold(n_splits=5, shuffle=True, random_state=42),
                                    scoring='r2', n_jobs=-1, refit=True, random_state=42)
    grid_search.fit(X_train_sg, Y_train_multi)
    best_models[name] = grid_search.best_estimator_
    print(f"✅ {name} 最佳參數: {grid_search.best_params_}")

    # OOF 預測
    train_preds = np.zeros_like(Y_train_multi)
    test_preds = np.zeros_like(Y_test_multi)
    for train_idx, val_idx in kf.split(X_train_sg):
        X_tr, X_val = X_train_sg[train_idx], X_train_sg[val_idx]
        Y_tr = Y_train_multi[train_idx]
        model = config['model'].set_params(**grid_search.best_params_)
        model.fit(X_tr, Y_tr)
        train_preds[val_idx] = model.predict(X_val)
        test_preds += model.predict(X_test_sg) / PARAMS['K_FOLDS']
    
    meta_features_train[:, len(components) * list(base_models.keys()).index(name):len(components) * (list(base_models.keys()).index(name) + 1)] = train_preds
    meta_features_test[:, len(components) * list(base_models.keys()).index(name):len(components) * (list(base_models.keys()).index(name) + 1)] = test_preds
    
    # 計算基模型性能
    for i, comp in enumerate(components):
        train_r2 = safe_r2_score(scaler_y.inverse_transform(Y_train_multi)[:, i], scaler_y.inverse_transform(train_preds)[:, i])
        test_r2 = safe_r2_score(scaler_y.inverse_transform(Y_test_multi)[:, i], scaler_y.inverse_transform(test_preds)[:, i])
        base_model_metrics[name][comp]['train_r2'] = train_r2
        base_model_metrics[name][comp]['test_r2'] = test_r2
        print(f"✅ {name} - {comp} Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")

# === 步驟 4: 檢查 Meta Features 共線性 ===
meta_features_df = pd.DataFrame(meta_features_train, columns=[f"{model}_{comp}" for model in base_models for comp in components])
corr_matrix = meta_features_df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
plt.title('Correlation Heatmap of Meta Features')
plt.tight_layout()
plt.show()

# 計算 VIF
vif_data = pd.DataFrame()
vif_data['Feature'] = meta_features_df.columns
vif_data['VIF'] = [variance_inflation_factor(meta_features_df.values, i) for i in range(meta_features_df.shape[1])]
print("✅ Meta Features VIF:")
print(vif_data)

# 移除高 VIF 特徵（VIF > 10）
selected_features = vif_data[vif_data['VIF'] < 10]['Feature'].tolist()
if len(selected_features) < meta_features_df.shape[1]:
    print(f"✅ 移除高 VIF 特徵，保留: {selected_features}")
    meta_features_train = meta_features_df[selected_features].values
    meta_features_test = pd.DataFrame(meta_features_test, columns=meta_features_df.columns)[selected_features].values
else:
    print("✅ 無高 VIF 特徵，保留所有 meta features")
    selected_features = meta_features_df.columns.tolist()

# 重新映射特徵索引到 VIF 篩選後的列
meta_features_df_selected = pd.DataFrame(meta_features_train, columns=selected_features)
model_feature_indices = {}
for model in base_models:
    model_cols = [col for col in selected_features if col.startswith(model)]
    model_feature_indices[model] = [meta_features_df_selected.columns.get_loc(col) for col in model_cols]
print(f"✅ 每個基模型的保留特徵索引: {model_feature_indices}")

# === 步驟 5: 對 Meta Features 降維 ===
pca_meta = PCA(n_components=PARAMS['PCA_N_COMPONENTS'])
meta_features_train_pca = pca_meta.fit_transform(meta_features_train)
meta_features_test_pca = pca_meta.transform(meta_features_test)
print(f"✅ 降維後 Meta Features 訓練集形狀: {meta_features_train_pca.shape}")
print(f"✅ 降維後 Meta Features 測試集形狀: {meta_features_test_pca.shape}")

# === 步驟 6: 訓練元學習器（Ridge） ===
meta_param_grid = {
    'estimator__alpha': PARAMS['RIDGE_ALPHA']
}
meta_ridge = MultiOutputRegressor(Ridge(random_state=42))
meta_grid_search = RandomizedSearchCV(meta_ridge, meta_param_grid, n_iter=20,
                                     cv=KFold(n_splits=5, shuffle=True, random_state=42),
                                     scoring='r2', n_jobs=-1, refit=True, random_state=42)
meta_grid_search.fit(meta_features_train_pca, Y_train_multi)
best_meta_ridge = meta_grid_search.best_estimator_
print(f"✅ 元學習器最佳參數: {meta_grid_search.best_params_}")

# 簡單平均集成作為基準
avg_preds_train = np.zeros_like(Y_train_multi)
avg_preds_test = np.zeros_like(Y_test_multi)
for i, comp in enumerate(components):
    valid_cols = [idx for model in base_models for idx in model_feature_indices[model] if meta_features_df_selected.columns[idx].endswith(comp)]
    if valid_cols:
        avg_preds_train[:, i] = np.mean(meta_features_train[:, valid_cols], axis=1)
        avg_preds_test[:, i] = np.mean(meta_features_test[:, valid_cols], axis=1)
    else:
        print(f"⚠️ 成分 {comp} 無有效 meta features，使用 RandomForest 預測作為備用")
        rf_idx = list(base_models.keys()).index('RandomForest')
        avg_preds_train[:, i] = meta_features_train[:, rf_idx * len(components) + i]
        avg_preds_test[:, i] = meta_features_test[:, rf_idx * len(components) + i]

# === 步驟 7: 測試與評估 ===
Y_train_pred_multi = best_meta_ridge.predict(meta_features_train_pca)
Y_test_pred_multi = best_meta_ridge.predict(meta_features_test_pca)
Y_train_pred = scaler_y.inverse_transform(Y_train_pred_multi)
Y_test_pred = scaler_y.inverse_transform(Y_test_pred_multi)
Y_train_multi_original = scaler_y.inverse_transform(Y_train_multi)
Y_test_multi_original = scaler_y.inverse_transform(Y_test_multi)
avg_preds_train_original = scaler_y.inverse_transform(avg_preds_train)
avg_preds_test_original = scaler_y.inverse_transform(avg_preds_test)

# 計算 SEC 和 SEP
n_train = X_train.shape[0]
n_test = X_test.shape[0]
n_components = Y_multi.shape[1]
actual_components = components[:n_components]
print(f"✅ 實際檢測成分數: {n_components}, 成分列表: {actual_components}")

# 評估每個成分（Ridge 和平均集成）
for i, component in enumerate(actual_components):
    # Ridge 模型
    train_mse = mean_squared_error(Y_train_pred[:, i], Y_train_multi_original[:, i])
    test_mse = mean_squared_error(Y_test_pred[:, i], Y_test_multi_original[:, i])
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    train_r2 = safe_r2_score(Y_train_pred[:, i], Y_train_multi_original[:, i])
    test_r2 = safe_r2_score(Y_test_pred[:, i], Y_test_multi_original[:, i])
    sec = np.sqrt(np.sum((Y_train_multi_original[:, i] - Y_train_pred[:, i]) ** 2) / max(1, n_train - 1))
    sep = np.sqrt(np.sum((Y_test_multi_original[:, i] - Y_test_pred[:, i]) ** 2) / (n_test - 1))
    print(f"✅ {component} (Ridge) - Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
    print(f"✅ {component} (Ridge) - Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
    print(f"✅ {component} (Ridge) - SEC: {sec:.4f}, SEP: {sep:.4f}")

    # 平均集成
    avg_train_r2 = safe_r2_score(Y_train_multi_original[:, i], avg_preds_train_original[:, i])
    avg_test_r2 = safe_r2_score(Y_test_multi_original[:, i], avg_preds_test_original[:, i])
    print(f"✅ {component} (Average Ensemble) - Train R²: {avg_train_r2:.4f}, Test R²: {avg_test_r2:.4f}")

# 5-fold 交叉驗證評估
cv_scores = []
for i in range(n_components):
    scores = cross_val_score(Ridge(alpha=meta_grid_search.best_params_['estimator__alpha']), 
                             meta_features_train_pca, Y_train_multi[:, i], cv=5, scoring='r2')
    cv_scores.append(scores.mean())
print(f"\n✅ 5-fold Cross-Validation Mean R² (per component): {np.mean(cv_scores):.4f} (± {np.std(cv_scores) * 2:.4f})")

# 測試集交叉驗證
test_cv_scores = []
for i in range(n_components):
    scores = cross_val_score(Ridge(alpha=meta_grid_search.best_params_['estimator__alpha']),
                             meta_features_test_pca, Y_test_multi[:, i], cv=3, scoring='r2')
    test_cv_scores.append(scores.mean())
print(f"✅ 測試集 3-fold Cross-Validation Mean R²: {np.mean(test_cv_scores):.4f} (± {np.std(test_cv_scores) * 2:.4f})")

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
    ax.scatter(Y_test_multi_original[:, i], Y_test_pred[:, i], alpha=0.7, color='blue', label='Meta-Learner (Ridge)')
    ax.scatter(Y_test_multi_original[:, i], avg_preds_test_original[:, i], alpha=0.7, color='green', label='Average Ensemble')
    for name in base_models:
        test_preds = scaler_y.inverse_transform(meta_features_test[:, len(components) * list(base_models.keys()).index(name):len(components) * (list(base_models.keys()).index(name) + 1)])
        ax.scatter(Y_test_multi_original[:, i], test_preds[:, i], alpha=0.5, label=f'Base Model ({name})')
    ax.plot([min(Y_test_multi_original[:, i]), max(Y_test_multi_original[:, i])],
            [min(Y_test_multi_original[:, i]), max(Y_test_multi_original[:, i])], 'r--', label='Ideal (y=x)')
    ax.set_xlabel(f'Actual {component} Content (%)')
    ax.set_ylabel(f'Predicted {component} Content (%)')
    ax.set_title(f'Actual vs Predicted {component}: Meta-Learner vs Base Models vs Average Ensemble')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# === 步驟 8: 生成參數和指標的 Excel 檔案 ===
common_params = {
    'Wavelength Range (nm)': PARAMS['WAVELENGTH_RANGE_STR'],
    'Train-Test Split Ratio': PARAMS['TRAIN_TEST_RATIO_STR'],
    'K-Folds for OOF': PARAMS['K_FOLDS'],
    'PCA Components for Meta Features': PARAMS['PCA_N_COMPONENTS']
}

specific_params = {}
for name, model in best_models.items():
    if name == 'RandomForest':
        specific_params['RF Optimal N Estimators'] = model.estimators_[0].n_estimators
        specific_params['RF Optimal Max Depth'] = model.estimators_[0].max_depth
    elif name == 'PLS':
        specific_params['PLS Optimal N Components'] = model.n_components

specific_params['Ridge Optimal Alpha'] = meta_grid_search.best_params_['estimator__alpha']

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_filename = f'Meta_Learning_Ridge_OOF_{timestamp}.xlsx'
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
            eval_metrics[f'{component}_Avg_Train_R²'] = f'{safe_r2_score(Y_train_multi_original[:, i], avg_preds_train_original[:, i]):.4f}'
            eval_metrics[f'{component}_Avg_Test_R²'] = f'{safe_r2_score(Y_test_multi_original[:, i], avg_preds_test_original[:, i]):.4f}'
            for name in base_models:
                eval_metrics[f'{component}_{name}_Train_R²'] = f'{base_model_metrics[name][component]["train_r2"]:.4f}'
                eval_metrics[f'{component}_{name}_Test_R²'] = f'{base_model_metrics[name][component]["test_r2"]:.4f}'
        else:
            eval_metrics = {f'{component}_RMSEC': 'N/A', f'{component}_RMSEP': 'N/A', f'{component}_R²_C': 'N/A',
                           f'{component}_R²_P': 'N/A', f'{component}_SEC': 'N/A', f'{component}_SEP': 'N/A',
                           f'{component}_Avg_Train_R²': 'N/A', f'{component}_Avg_Test_R²': 'N/A'}
            for name in base_models:
                eval_metrics[f'{component}_{name}_Train_R²'] = 'N/A'
                eval_metrics[f'{component}_{name}_Test_R²'] = 'N/A'

        output_data = {
            'Parameter/Metric': list(common_params.keys()) + list(specific_params.keys()) + list(eval_metrics.keys()),
            'Value': list(common_params.values()) + list(specific_params.values()) + list(eval_metrics.values()),
            'Category': ['Common Parameters'] * len(common_params) + ['Specific Parameters'] * len(specific_params) + ['Evaluation Metrics'] * len(eval_metrics)
        }

        output_df = pd.DataFrame(output_data)
        output_df.to_excel(writer, sheet_name=component, index=False)

print(f"✅ 已生成參數和指標檔案: {output_filename}")