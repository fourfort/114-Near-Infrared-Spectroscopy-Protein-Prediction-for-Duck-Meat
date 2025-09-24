import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter
from scipy.stats import chi2
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from datetime import datetime

# 定義統一參數
PARAMS = {
    'WAVELENGTH_RANGE': (750, 1099.5),  # 波長範圍 (min, max)
    'WAVELENGTH_RANGE_STR': '750-1099.5',  # 波長範圍字串表示
    'MCS_N_ITERATIONS': 1000,  # MCS 抽樣次數
    'MCS_N_COMPONENTS': 7,  # MCS PLS 主成分數（用於異常值檢測）
    'MAHALANOBIS_THRESHOLD': 80,  # 馬氏距離閾值 (%)
    'TRAIN_TEST_RATIO': 0.2,  # 測試集比例 (0.2 表示 80:20)
    'TRAIN_TEST_RATIO_STR': '80:20',  # 訓練-測試比例字串表示
    'MAX_DEPTH_RANGE': [3, 5, 7, 10],  # 決策樹最大深度範圍
    'MIN_SAMPLES_SPLIT_RANGE': [2, 5, 10, 20]  # 最小分割樣本數範圍
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

# 定義相關性數據
correlation_data = {
    ('Moisture', 'Protein'): (0.32 + 0.48) / 2,
    ('Moisture', 'Fat'): (-0.50 + -0.68) / 2,
    ('Moisture', 'SFA'): (-0.34 + -0.49) / 2,
    ('Moisture', 'Ash'): (-0.51 + -0.71) / 2,
    ('Moisture', 'Collagen'): (-0.08 + -0.12) / 2,
    ('Moisture', 'Salt'): (0.18 + 0.25) / 2,
    ('Protein', 'Fat'): (-0.55 + -0.74) / 2,
    ('Protein', 'SFA'): (-0.34 + -0.48) / 2,
    ('Protein', 'Ash'): (-0.57 + -0.79) / 2,
    ('Protein', 'Collagen'): (0.07 + 0.09) / 2,
    ('Protein', 'Salt'): (0.24 + 0.32) / 2,
    ('Fat', 'SFA'): (0.44 + 0.62) / 2,
    ('Fat', 'Ash'): (0.50 + 0.70) / 2,
    ('Fat', 'Collagen'): (-0.04 + -0.04) / 2,
    ('Fat', 'Salt'): (-0.21 + -0.28) / 2,
    ('SFA', 'Ash'): (0.22 + 0.31) / 2,
    ('SFA', 'Collagen'): (-0.34 + -0.49) / 2,
    ('SFA', 'Salt'): (-0.37 + -0.51) / 2,
    ('Ash', 'Collagen'): (0.07 + 0.10) / 2,
    ('Ash', 'Salt'): (-0.13 + -0.11) / 2,
    ('Collagen', 'Salt'): (0.38 + 0.53) / 2
}

components = ['Moisture', 'Protein', 'Fat', 'SFA', 'Ash', 'Collagen', 'Salt']

# 構建相關性矩陣
correlation_matrix = pd.DataFrame(0.0, index=components, columns=components)
for (comp1, comp2), corr in correlation_data.items():
    correlation_matrix.at[comp1, comp2] = corr
    correlation_matrix.at[comp2, comp1] = corr

# 識別強相關組 (abs >= 0.7)
strong_correlation_threshold = 0.7
groups = {}
for i, comp1 in enumerate(components):
    for j, comp2 in enumerate(components):
        if i < j and abs(correlation_matrix.at[comp1, comp2]) >= strong_correlation_threshold:
            if comp1 not in groups and comp2 not in groups:
                groups[comp1] = [comp2]
            elif comp1 in groups:
                groups[comp1].append(comp2)
            elif comp2 in groups:
                groups[comp2].append(comp1)

independent_components = [comp for comp in components if comp not in groups and all(comp not in group for group in groups.values())]
print(f"✅ 強相關分組: {groups}")
print(f"✅ 獨立成分: {independent_components}")

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

# 檢查缺失值
print("\n🔍 NIR 缺失值檢查:\n", nir_df.isnull().sum())
print("\n🔍 傳統分析缺失值檢查:\n", trad_df.isnull().sum())

def find_sample_column(df, name='nir'):
    for col in df.columns:
        if 'sample number' in col.lower() or 'sample no' in col.lower():
            print(f"✅ 在 {name} 資料中找到樣本編號欄位: {col}")
            return col
    print(f"⚠️ {name} 資料中未找到樣本編號欄位")
    return None

nir_sample_col = find_sample_column(nir_df, 'NIR')
trad_sample_col = find_sample_column(trad_df, '傳統分析')

if not nir_sample_col:
    nir_sample_col = 'Sample_Number'
    nir_df[nir_sample_col] = [f'1-{i+1}' for i in range(len(nir_df))]

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

if trad_sample_col:
    trad_df[trad_sample_col] = [f'1-{i+1}' for i in range(len(trad_df))]
else:
    print("⚠️ 無法進行樣本對齊，缺少樣本編號欄")
    exit()

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

# === 步驟 2: 去除異常值 ===
pca = PCA(n_components=min(X.shape[0], X.shape[1], 20))
X_pca = pca.fit_transform(X)
print(f"✅ PCA 降維後資料形狀: {X_pca.shape}")

def mahalanobis_distance(X):
    cov_matrix = np.cov(X, rowvar=False)
    cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6
    try:
        cov_inv = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        print("❌ 協方差矩陣無法求逆，嘗試進一步正則化")
        cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-3
        cov_inv = np.linalg.inv(cov_matrix)
    mean = np.mean(X, axis=0)
    md = np.sqrt(np.sum((X - mean) @ cov_inv * (X - mean), axis=1))
    return md

n_simulations = PARAMS['MCS_N_ITERATIONS']
np.random.seed(42)
md_thresholds = []
for _ in range(n_simulations):
    sim_data = np.random.multivariate_normal(np.mean(X_pca, axis=0), np.cov(X_pca, rowvar=False), size=X_pca.shape[0])
    cov_sim = np.cov(sim_data, rowvar=False)
    if np.any(np.isnan(cov_sim)) or np.any(np.isinf(cov_sim)):
        print("⚠️ 模擬數據協方差矩陣包含無效值")
        continue
    md_sim = mahalanobis_distance(sim_data)
    md_thresholds.append(np.percentile(md_sim, PARAMS['MAHALANOBIS_THRESHOLD']))
md_threshold = np.mean(md_thresholds)

md = mahalanobis_distance(X_pca)
mask_md = md < md_threshold
print(f"✅ 馬氏距離法檢測到 {np.sum(~mask_md)} 個異常值，剩餘樣本數：{np.sum(mask_md)}")

def monte_carlo_outlier_detection(X, y, n_iterations=PARAMS['MCS_N_ITERATIONS'], 
                                 sample_ratio=0.8, n_components=PARAMS['MCS_N_COMPONENTS'], 
                                 threshold_percentile=95):
    n_samples = X.shape[0]
    n_subset = int(n_samples * sample_ratio)
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

scaler_y_temp = StandardScaler()
Y_multi_scaled_temp = scaler_y_temp.fit_transform(Y_multi)
Y_mean_scaled = np.mean(Y_multi_scaled_temp, axis=1)
mask_mcs = monte_carlo_outlier_detection(X, Y_mean_scaled)
mask = mask_md & mask_mcs
X = X[mask]
Y_multi = Y_multi[mask]
print(f"✅ 結合馬氏距離和 MCS 後，剩餘樣本數：{X.shape[0]}")
for component in components:
    print(f"✅ 剔除後 {component} 描述:", pd.Series(Y_multi[:, components.index(component)]).describe())

mask_range = ((Y_multi >= 0) & (Y_multi <= 100)).all(axis=1)
X = X[mask_range]
Y_multi = Y_multi[mask_range]
print(f"✅ 移除異常範圍值後，剩餘樣本數：{X.shape[0]}")
for component in components:
    print(f"✅ 移除後 {component} 描述:", pd.Series(Y_multi[:, components.index(component)]).describe())

# === 步驟 3: NIR 光譜標準化 ===
scaler_x = StandardScaler()
X_scaled = scaler_x.fit_transform(X)
X = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
print("✅ NIR 光譜標準化完成")

# === 步驟 4: 分配樣本集（SPXY） ===
def spxy(X, Y, test_size=PARAMS['TRAIN_TEST_RATIO']):
    X = np.array(X)
    Y = np.array(Y)
    n_samples = X.shape[0]
    n_test = int(np.floor(test_size * n_samples))

    dist_X = pairwise_distances(X)
    dist_Y = pairwise_distances(Y)
    dist_XY = dist_X + dist_Y

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
    remaining = np.array(remaining)

    X_train = X[selected]
    Y_train = Y[selected]
    X_test = X[remaining]
    Y_test = Y[remaining]

    return X_train, X_test, Y_train, Y_test

scaler_y = StandardScaler()
Y_multi_scaled = scaler_y.fit_transform(Y_multi)
X_train, X_test, Y_train_multi, Y_test_multi = spxy(X, Y_multi_scaled)

print(f"✅ SPXY 訓練集形狀: {X_train.shape}, 測試集形狀: {X_test.shape}")
for i, component in enumerate(components):
    print(f"✅ 訓練集 {component} 描述:", pd.Series(scaler_y.inverse_transform(Y_train_multi)[:, i]).describe())
    print(f"✅ 測試集 {component} 描述:", pd.Series(scaler_y.inverse_transform(Y_test_multi)[:, i]).describe())

# === 步驟 5: 光譜預處理與降維 ===
X_train_sg = savgol_filter(X_train, window_length=51, polyorder=3, axis=1)
X_test_sg = savgol_filter(X_test, window_length=51, polyorder=3, axis=1)
print("✅ Savitzky-Golay 濾波完成")

pca = PCA(n_components=20)
X_train_pca = pca.fit_transform(X_train_sg)
X_test_pca = pca.transform(X_test_sg)
print(f"✅ PCA 降維後訓練集形狀: {X_train_pca.shape}, 測試集形狀: {X_test_pca.shape}")

# === 步驟 6: 模型訓練 ===
models = {}
param_grid_multi = {
    'estimator__max_depth': PARAMS['MAX_DEPTH_RANGE'],
    'estimator__min_samples_split': PARAMS['MIN_SAMPLES_SPLIT_RANGE']
}
param_grid_single = {
    'max_depth': PARAMS['MAX_DEPTH_RANGE'],
    'min_samples_split': PARAMS['MIN_SAMPLES_SPLIT_RANGE']
}

for group_name, components_list in groups.items():
    target_indices = [components.index(comp) for comp in [group_name] + components_list]
    Y_train_group = Y_train_multi[:, target_indices]
    Y_test_group = Y_test_multi[:, target_indices]
    
    dt = DecisionTreeRegressor(random_state=42)
    multi_dt = MultiOutputRegressor(dt)
    grid_search = GridSearchCV(multi_dt, param_grid_multi, cv=KFold(n_splits=5, shuffle=True, random_state=42),
                               scoring='r2', n_jobs=-1, refit=True)
    grid_search.fit(X_train_pca, Y_train_group)
    
    models[group_name] = {
        'model': grid_search.best_estimator_,
        'params': grid_search.best_params_,
        'indices': target_indices
    }
    print(f"✅ {group_name} 組最佳參數: max_depth={grid_search.best_params_['estimator__max_depth']}, min_samples_split={grid_search.best_params_['estimator__min_samples_split']}")

for comp in independent_components:
    target_index = components.index(comp)
    Y_train_single = Y_train_multi[:, target_index].reshape(-1, 1)
    Y_test_single = Y_test_multi[:, target_index].reshape(-1, 1)
    
    dt = DecisionTreeRegressor(random_state=42)
    grid_search = GridSearchCV(dt, param_grid_single, cv=KFold(n_splits=5, shuffle=True, random_state=42),
                              scoring='r2', n_jobs=-1, refit=True)
    grid_search.fit(X_train_pca, Y_train_single)
    
    models[comp] = {
        'model': grid_search.best_estimator_,
        'params': grid_search.best_params_,
        'indices': [target_index]
    }
    print(f"✅ {comp} 最佳參數: max_depth={grid_search.best_params_['max_depth']}, min_samples_split={grid_search.best_params_['min_samples_split']}")

# === 步驟 7: 預測與評估 ===
Y_train_pred = np.zeros_like(Y_train_multi)
Y_test_pred = np.zeros_like(Y_test_multi)

for group_name, data in models.items():
    model = data['model']
    indices = data['indices']
    if len(indices) > 1:  # MultiOutputRegressor
        Y_train_pred[:, indices] = model.predict(X_train_pca)
        Y_test_pred[:, indices] = model.predict(X_test_pca)
    else:  # Single DecisionTreeRegressor
        Y_train_pred[:, indices] = model.predict(X_train_pca).reshape(-1, 1)
        Y_test_pred[:, indices] = model.predict(X_test_pca).reshape(-1, 1)

Y_train_pred = scaler_y.inverse_transform(Y_train_pred)
Y_test_pred = scaler_y.inverse_transform(Y_test_pred)
Y_train_multi_original = scaler_y.inverse_transform(Y_train_multi)
Y_test_multi_original = scaler_y.inverse_transform(Y_test_multi)

# 計算評估指標
n_train = X_train.shape[0]
n_test = X_test.shape[0]
n_components = Y_multi.shape[1]
actual_components = components[:n_components]
print(f"✅ 實際檢測成分數: {n_components}, 成分列表: {actual_components}")

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

for group_name, data in models.items():
    model = data['model']
    indices = data['indices']
    if len(indices) > 1:
        cv_scores = cross_val_score(model, X_train_pca, Y_train_multi[:, indices], cv=5, scoring='r2')
        print(f"\n✅ {group_name} 組 5-fold Cross-Validation Mean R²: {cv_scores.mean():.4f} (± {cv_scores.std() * 2:.4f})")
    else:
        cv_scores = cross_val_score(model, X_train_pca, Y_train_multi[:, indices].ravel(), cv=5, scoring='r2')
        print(f"\n✅ {group_name} 5-fold Cross-Validation Mean R²: {cv_scores.mean():.4f} (± {cv_scores.std() * 2:.4f})")

# 指標表格數據
metrics_data_rmse = {'Component': actual_components * 2, 'Metric': ['RMSE'] * n_components * 2,
                     'Training Set': [f'{np.sqrt(mean_squared_error(Y_train_pred[:, i], Y_train_multi_original[:, i])):.4f}' for i in range(n_components)] + ['N/A'] * n_components,
                     'Test Set': ['N/A'] * n_components + [f'{np.sqrt(mean_squared_error(Y_test_pred[:, i], Y_test_multi_original[:, i])):.4f}' for i in range(n_components)]}
metrics_df_rmse = pd.DataFrame(metrics_data_rmse)
print("\n✅ RMSE Evaluation Metrics:")
print(metrics_df_rmse.to_string(index=False))

metrics_data_se = {'Component': actual_components * 2, 'Metric': ['SE'] * n_components * 2,
                   'Training Set': [f'{np.sqrt(np.sum((Y_train_multi_original[:, i] - Y_train_pred[:, i]) ** 2) / max(1, n_train - 1)):.4f}' for i in range(n_components)] + ['N/A'] * n_components,
                   'Test Set': ['N/A'] * n_components + [f'{np.sqrt(np.sum((Y_test_multi_original[:, i] - Y_test_pred[:, i]) ** 2) / (n_test - 1)):.4f}' for i in range(n_components)]}
metrics_df_se = pd.DataFrame(metrics_data_se)
print("\n✅ SE Evaluation Metrics:")
print(metrics_df_se.to_string(index=False))

metrics_data_r2 = {'Component': actual_components * 2, 'Metric': ['R² Score'] * n_components * 2,
                   'Training Set': [f'{safe_r2_score(Y_train_pred[:, i], Y_train_multi_original[:, i]):.4f}' for i in range(n_components)] + ['N/A'] * n_components,
                   'Test Set': ['N/A'] * n_components + [f'{safe_r2_score(Y_test_pred[:, i], Y_test_multi_original[:, i]):.4f}' for i in range(n_components)]}
metrics_df_r2 = pd.DataFrame(metrics_data_r2)
print("\n✅ R² Score Evaluation Metrics:")
print(metrics_df_r2.to_string(index=False))

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

for i, component in enumerate(actual_components):
    plt.figure(figsize=(6, 4))
    plt.hist(Y_multi[:, i], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title(f'Histogram of {component} Content')
    plt.xlabel(f'{component} Content (%)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.scatter(Y_test_multi_original[:, i], Y_test_pred[:, i], alpha=0.7, color='blue', label='Predicted vs Actual')
    plt.plot([min(Y_test_multi_original[:, i]), max(Y_test_multi_original[:, i])],
             [min(Y_test_multi_original[:, i]), max(Y_test_multi_original[:, i])], 'r--', label='Ideal (y=x)')
    plt.xlabel(f'Actual {component} Content (%)')
    plt.ylabel(f'Predicted {component} Content (%)')
    plt.title(f'Decision Tree Regression: Actual vs Predicted {component}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

common_params = {
    'Wavelength Range (nm)': PARAMS['WAVELENGTH_RANGE_STR'],
    'MCS Sampling Iterations': PARAMS['MCS_N_ITERATIONS'],
    'MCS PLS Components': PARAMS['MCS_N_COMPONENTS'],
    'Mahalanobobis Distance Threshold (%)': PARAMS['MAHALANOBIS_THRESHOLD'],
    'Train-Test Split Ratio': PARAMS['TRAIN_TEST_RATIO_STR']
}

specific_params = {f"{k}_Optimal Max Depth": v['params'].get('estimator__max_depth', v['params'].get('max_depth')) for k, v in models.items()}
specific_params.update({f"{k}_Optimal Min Samples Split": v['params'].get('estimator__min_samples_split', v['params'].get('min_samples_split')) for k, v in models.items()})

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_filename = f'Multi_DT_{timestamp}.xlsx'
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
        else:
            eval_metrics = {f'{component}_RMSEC': 'N/A', f'{component}_RMSEP': 'N/A', f'{component}_R²_C': 'N/A',
                           f'{component}_R²_P': 'N/A', f'{component}_SEC': 'N/A', f'{component}_SEP': 'N/A'}

        output_data = {
            'Parameter/Metric': list(common_params.keys()) + list(specific_params.keys()) + list(eval_metrics.keys()),
            'Value': list(common_params.values()) + list(specific_params.values()) + list(eval_metrics.values()),
            'Category': ['Common Parameters'] * len(common_params) + ['Specific Parameters'] * len(specific_params) + ['Evaluation Metrics'] * len(eval_metrics)
        }
        output_df = pd.DataFrame(output_data)
        output_df.to_excel(writer, sheet_name=component, index=False)

print(f"✅ 已生成參數和指標檔案: {output_filename}")