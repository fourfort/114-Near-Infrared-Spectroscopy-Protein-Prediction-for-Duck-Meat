import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter
from scipy.stats import chi2
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import pairwise_distances
from datetime import datetime

# 定義統一參數
PARAMS = {
    'WAVELENGTH_RANGE': (750, 1099.5),
    'WAVELENGTH_RANGE_STR': '750-1099.5',
    'MCS_N_ITERATIONS': 1000,
    'MCS_N_COMPONENTS': 7,
    'MAHALANOBIS_THRESHOLD': 80,
    'TRAIN_TEST_RATIO': 0.2,
    'TRAIN_TEST_RATIO_STR': '80:20',
    'MAX_DEPTH_RANGE': [3,5,7,10,11],  # 優化範圍
    'MIN_SAMPLES_SPLIT_RANGE': [1,2, 5, 10, 20],
    'MIN_SAMPLES_LEAF_RANGE': [1,2,3,5]  # 限制過擬合
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

# 範圍檢查
mask_range = ((Y_multi >= 0) & (Y_multi <= 100)).all(axis=1)
X = X[mask_range]
Y_multi = Y_multi[mask_range]
print(f"✅ 移除異常範圍值後，剩餘樣本數：{X.shape[0]}")
for component in components:
    print(f"✅ 移除後 {component} 描述:", pd.Series(Y_multi[:, components.index(component)]).describe())

# === 步驟 2.5: 成分相關性分析 ===
Y_df = pd.DataFrame(Y_multi, columns=components)
print("\n✅ 計算成分相關性矩陣...")
print(f"✅ Y_multi 形狀: {Y_multi.shape}, 成分數: {Y_multi.shape[1]}")

# 更新 components 以匹配 Y_multi
if Y_multi.shape[1] != len(components):
    print(f"⚠️ Y_multi 成分數 ({Y_multi.shape[1]}) 與 components 長度 ({len(components)}) 不匹配，更新 components 列表")
    components = components[:Y_multi.shape[1]]
    Y_df = pd.DataFrame(Y_multi, columns=components)
    print(f"✅ 更新後的 components: {components}")

# 標準化 Y_df 用於 VIF 計算
scaler_y_vif = StandardScaler()
Y_df_scaled = scaler_y_vif.fit_transform(Y_df)
Y_df_scaled = pd.DataFrame(Y_df_scaled, columns=Y_df.columns)

# Pearson 相關性
corr_pearson = Y_df.corr(method='pearson')
plt.figure(figsize=(10, 8))
sns.heatmap(corr_pearson, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
plt.title("Pearson Correlation Matrix of Components")
plt.show()
print("✅ Pearson 相關性矩陣:\n", corr_pearson)

# Spearman 相關性
corr_spearman = Y_df.corr(method='spearman')
plt.figure(figsize=(10, 8))
sns.heatmap(corr_spearman, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
plt.title("Spearman Correlation Matrix of Components")
plt.show()
print("✅ Spearman 相關性矩陣:\n", corr_spearman)

# Kendall’s Tau 相關性
corr_kendall = Y_df.corr(method='kendall')
plt.figure(figsize=(10, 8))
sns.heatmap(corr_kendall, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
plt.title("Kendall’s Tau Correlation Matrix of Components")
plt.show()
print("✅ Kendall’s Tau 相關性矩陣:\n", corr_kendall)

# 檢查多重共線性 (VIF)
vif_data = pd.DataFrame()
vif_data['Component'] = Y_df_scaled.columns
vif_data['VIF'] = [variance_inflation_factor(Y_df_scaled.values, i) for i in range(Y_df_scaled.shape[1])]
print("\n✅ VIF 检查 (多重共線性):\n", vif_data)

# 基於 VIF 移除高共線成分 (VIF > 50)
high_vif = vif_data[vif_data['VIF'] > 50]['Component'].tolist()
if high_vif:
    print(f"⚠️ 檢測到高 VIF 成分: {high_vif}")
    priority_components = ['Moisture', 'Protein', 'Fat']  # 優先保留
    components_to_remove = [c for c in high_vif if c not in priority_components][:1]  # 最多移除 1 個
    if components_to_remove:
        print(f"✅ 移除高共線成分: {components_to_remove}")
        components_to_keep = [c for c in components if c not in components_to_remove]
        Y_multi = Y_df[components_to_keep].values
        components = components_to_keep
        Y_df = pd.DataFrame(Y_multi, columns=components)
        print(f"✅ 保留成分: {components}")
    else:
        print("✅ 無需移除，所有高 VIF 成分為優先成分")
else:
    print("✅ 無高 VIF 成分，保留所有成分")

# 檢查 Y_multi 是否為空
if Y_multi.shape[1] == 0:
    print("❌ Y_multi 為空，恢復原始成分並禁用 VIF 移除")
    Y_multi = Y_df[components].values
    components = Y_df.columns.tolist()
    print(f"✅ 恢復 components: {components}")

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
print(f"✅ Y_train_multi 成分數: {Y_train_multi.shape[1]}, components: {components}")
for i, component in enumerate(components):
    if i < Y_train_multi.shape[1]:
        print(f"✅ 訓練集 {component} 描述:", pd.Series(scaler_y.inverse_transform(Y_train_multi)[:, i]).describe())
        print(f"✅ 測試集 {component} 描述:", pd.Series(scaler_y.inverse_transform(Y_test_multi)[:, i]).describe())
    else:
        print(f"⚠️ 成分 {component} 不存在於 Y_train_multi，跳過")

# === 步驟 5: 光譜預處理（Savitzky-Golay 濾波） ===
X_train_sg = savgol_filter(X_train, window_length=51, polyorder=3, axis=1)
X_test_sg = savgol_filter(X_test, window_length=51, polyorder=3, axis=1)
print("✅ Savitzky-Golay 濾波完成")

# === 步驟 6: 決策樹參數調優 ===
# 根據相關性調整 max_depth 範圍
avg_corr = corr_pearson.abs().mean().mean()
if avg_corr > 0.6:  # 根據先前 Protein-Ash |r|=0.75–0.79
    max_depth_range = [3,5,7,10]
else:
    max_depth_range = PARAMS['MAX_DEPTH_RANGE']
print(f"✅ 基於相關性調整 max_depth 範圍: {max_depth_range}")

param_grid = {
    'estimator__max_depth': max_depth_range,
    'estimator__min_samples_split': PARAMS['MIN_SAMPLES_SPLIT_RANGE'],
    'estimator__min_samples_leaf': PARAMS['MIN_SAMPLES_LEAF_RANGE']  # 新增
}
dt = DecisionTreeRegressor(random_state=42)
multi_dt = MultiOutputRegressor(dt)
grid_search = GridSearchCV(multi_dt, param_grid, cv=KFold(n_splits=5, shuffle=True, random_state=42),
                           scoring='r2', n_jobs=-1, refit=True)
grid_search.fit(X_train_sg, Y_train_multi)

# 最佳模型與評估
best_multi_dt = grid_search.best_estimator_
best_params = grid_search.best_params_
print(f"✅ 最佳參數: max_depth={best_params['estimator__max_depth']}, min_samples_split={best_params['estimator__min_samples_split']}, min_samples_leaf={best_params['estimator__min_samples_leaf']}")

# === 步驟 7: 模型訓練 ===
best_multi_dt.fit(X_train_sg, Y_train_multi)
print(f"✅ 模型訓練完成，使用的參數: max_depth={best_params['estimator__max_depth']}, min_samples_split={best_params['estimator__min_samples_split']}, min_samples_leaf={best_params['estimator__min_samples_leaf']}")

# === 步驟 8: 評估結果 ===
Y_train_pred_multi = best_multi_dt.predict(X_train_sg)
Y_test_pred_multi = best_multi_dt.predict(X_test_sg)
Y_train_pred = scaler_y.inverse_transform(Y_train_pred_multi)
Y_test_pred = scaler_y.inverse_transform(Y_test_pred_multi)
Y_train_multi_original = scaler_y.inverse_transform(Y_train_multi)
Y_test_multi_original = scaler_y.inverse_transform(Y_test_multi)

# 計算 SEC 和 SEP
n_train = X_train.shape[0]
n_test = X_test.shape[0]

# 動態獲取實際成分數
n_components = Y_multi.shape[1]
actual_components = components[:n_components]
print(f"✅ 實際檢測成分數: {n_components}, 成分列表: {actual_components}")

# 評估每個成分並提取特徵重要性
feature_importances = []
for i, (component, estimator) in enumerate(zip(actual_components, best_multi_dt.estimators_)):
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

    # 特徵重要性
    importance = estimator.feature_importances_
    feature_importances.append((component, importance))
    top_features = np.argsort(importance)[-5:]  # 前 5 個重要波長
    print(f"✅ {component} - Top 5 Feature Importances (wavelength indices): {top_features}")

# 繪製特徵重要性圖
for component, importance in feature_importances:
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importance)), importance)
    plt.title(f'Feature Importance for {component}')
    plt.xlabel('Wavelength Index')
    plt.ylabel('Importance')
    plt.grid(True, alpha=0.3)
    plt.show()

# 5-fold 交叉驗證評估
cv_scores = cross_val_score(best_multi_dt, X_train_sg, Y_train_multi, cv=5, scoring='r2')
print(f"\n✅ 5-fold Cross-Validation Mean R²: {cv_scores.mean():.4f} (± {cv_scores.std() * 2:.4f})")

# 指標表格數據（分為三個表格）
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

# 繪製三個獨立指標表格
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

# 繪製每個成分的含量直方圖
for i, component in enumerate(actual_components):
    plt.figure(figsize=(6, 4))
    plt.hist(Y_multi[:, i], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title(f'Histogram of {component} Content')
    plt.xlabel(f'{component} Content (%)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.show()

# 繪製每個成分的預測散點圖
for i, component in enumerate(actual_components):
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

# === 步驟 9: 生成參數和指標的 Excel 檔案 ===
common_params = {
    'Wavelength Range (nm)': PARAMS['WAVELENGTH_RANGE_STR'],
    'MCS Sampling Iterations': PARAMS['MCS_N_ITERATIONS'],
    'MCS PLS Components': PARAMS['MCS_N_COMPONENTS'],
    'Mahalanobis Distance Threshold (%)': PARAMS['MAHALANOBIS_THRESHOLD'],
    'Train-Test Split Ratio': PARAMS['TRAIN_TEST_RATIO_STR']
}

specific_params = {
    'Optimal Max Depth': best_params['estimator__max_depth'],
    'Optimal Min Samples Split': best_params['estimator__min_samples_split'],
    'Optimal Min Samples Leaf': best_params['estimator__min_samples_leaf']
}

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

    corr_pearson.to_excel(writer, sheet_name='Pearson_Correlation', float_format='%.2f')
    corr_spearman.to_excel(writer, sheet_name='Spearman_Correlation', float_format='%.2f')
    corr_kendall.to_excel(writer, sheet_name='Kendall_Correlation', float_format='%.2f')
    vif_data.to_excel(writer, sheet_name='VIF_Check', index=False)

print(f"✅ 已生成參數、指標和相關性檔案: {output_filename}")