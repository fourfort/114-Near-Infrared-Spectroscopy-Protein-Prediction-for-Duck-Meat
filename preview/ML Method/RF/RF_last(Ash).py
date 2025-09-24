import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter
from scipy.stats import chi2
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.metrics import pairwise_distances
from datetime import datetime

# 定義統一參數
PARAMS = {
    'WAVELENGTH_RANGE': (850, 1099.5),  # 波長範圍（600 點）
    'WAVELENGTH_RANGE_STR': '650-1099.5',
    'MCS_N_ITERATIONS': 200,  # MCS 迭代次數
    'MCS_N_COMPONENTS': 7,  # MCS 主成分數
    'MAHALANOBIS_THRESHOLD': 65,  # 馬氏距離閾值
    'TRAIN_TEST_RATIO': 0.2,  # 測試集比例 (80:20)
    'TRAIN_TEST_RATIO_STR': '80:20',
    'N_ESTIMATORS_RANGE': [50, 100, 200],  # 隨機森林樹數量
    'MAX_DEPTH_RANGE': [3, 5, 7],  # 最大深度
    'MIN_SAMPLES_SPLIT_RANGE': [5, 10]  # 最小分割樣本數
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

# 多重散射校正（MSC）
def msc(input_data):
    input_data = np.array(input_data)
    mean_spectrum = np.mean(input_data, axis=0)
    output_data = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        coeffs = np.polyfit(mean_spectrum, input_data[i], 1)
        output_data[i] = (input_data[i] - coeffs[1]) / coeffs[0]
    return output_data

# === 步驟 1: 數據樣本對齊獲取 ===
try:
    nir_df = pd.read_csv(r'D:\Users\wedke\document\extra document\大三上\113專題研究之製作\論文人生\紅外線\red_coding main\preview\nir_data.csv')
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

# 對齊樣本
if trad_sample_col:
    trad_df[trad_sample_col] = [f'1-{i+1}' for i in range(len(trad_df))]
else:
    print("⚠️ 無法進行樣本對齊，缺少樣本編號欄")
    exit()

if 'Ash' in trad_df.columns:
    trad_df = trad_df.dropna(subset=['Ash'])
    Y = trad_df['Ash'].values
    print("\n🎯 目標值 Y (Ash) 前五筆:", Y[:5])
    print("\n🎯 目標值 Y (Ash) 描述:", pd.Series(Y).describe())
else:
    print("❌ 缺少 'Ash' 欄位，無法建立模型")
    exit()

nir_transposed_df = nir_final_df.set_index('Wavelength').T
nir_transposed_df = nir_transposed_df.apply(pd.to_numeric, errors='coerce')
nir_transposed_df = nir_transposed_df.dropna(axis=1)
print("\n✅ 轉換後 NIR 資料形狀:", nir_transposed_df.shape)

sample_ids = trad_df[trad_sample_col].astype(str).tolist()
nir_transposed_df.index = nir_transposed_df.index.astype(str)
common_ids = list(set(sample_ids).intersection(set(nir_transposed_df.index)))
if len(common_ids) != len(sample_ids) or len(common_ids) != len(Y):
    print(f"❌ 對齊後樣本數不一致：X: {len(common_ids)}, Y: {len(Y)}")
    trad_df = trad_df[trad_df[trad_sample_col].astype(str).isin(common_ids)]
    Y = trad_df['Ash'].values
    X = nir_transposed_df.loc[common_ids]
else:
    X = nir_transposed_df.loc[sample_ids]
print(f"✅ 成功對齊資料，樣本數: {X.shape[0]}")

# === 步驟 2: 異常值處理和剔除 ===
# 馬氏距離法（內部標準化）
scaler_temp = StandardScaler()
X_scaled_temp = scaler_temp.fit_transform(X)
pca = PCA(n_components=min(X.shape[0], X.shape[1], 20))
X_pca = pca.fit_transform(X_scaled_temp)
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

# 蒙特卡洛採樣（MCS）異常值檢測
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
Y_scaled_temp = scaler_y_temp.fit_transform(Y.reshape(-1, 1)).ravel()
mask_mcs = monte_carlo_outlier_detection(X, Y_scaled_temp)
mask = mask_md & mask_mcs
X = X[mask]
Y = Y[mask]
print(f"✅ 結合馬氏距離和 MCS 後，剩餘樣本數：{X.shape[0]}")
print(f"✅ 剔除後目標值 Y (Ash) 描述:", pd.Series(Y).describe())

# Ash 含量範圍檢查（0-10%）
ash_mask = (Y >= 0) & (Y <= 10)
if not np.all(ash_mask):
    print(f"⚠️ 檢測到灰分含量異常值（應在 0-10%）：{Y[~ash_mask]}")
    X = X[ash_mask]
    Y = Y[ash_mask]
    print(f"✅ 移除灰分異常值後，剩餘樣本數：{X.shape[0]}")
    print(f"✅ 移除後目標值 Y (Ash) 描述:", pd.Series(Y).describe())

# 繪製灰分含量分佈直方圖
plt.figure(figsize=(8, 6))
plt.hist(Y, bins=20, edgecolor='black')
plt.xlabel('Ash Content (%)')
plt.ylabel('Frequency')
plt.title('Distribution of Ash Content')
plt.grid(True)
plt.tight_layout()
plt.show()

# === 步驟 3: 樣本標準化 ===
scaler_x = StandardScaler()
X_scaled = scaler_x.fit_transform(X)
X = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
print("✅ NIR 光譜標準化完成")

# === 步驟 4: 樣本集分配（SPXY） ===
def spxy(X, Y, test_size=PARAMS['TRAIN_TEST_RATIO']):
    X = np.array(X)
    Y = np.array(Y).reshape(-1, 1)
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
    Y_train = Y[selected].ravel()
    X_test = X[remaining]
    Y_test = Y[remaining].ravel()

    return X_train, X_test, Y_train, Y_test

scaler_y = StandardScaler()
Y_scaled = scaler_y.fit_transform(Y.reshape(-1, 1)).ravel()
X_train, X_test, Y_train, Y_test = spxy(X, Y_scaled)
print(f"✅ SPXY 訓練集形狀: {X_train.shape}, 測試集形狀: {X_test.shape}")
print(f"✅ 訓練集目標值 Y (Ash) 描述:", pd.Series(scaler_y.inverse_transform(Y_train.reshape(-1, 1)).ravel()).describe())
print(f"✅ 測試集目標值 Y (Ash) 描述:", pd.Series(scaler_y.inverse_transform(Y_test.reshape(-1, 1)).ravel()).describe())

# === 步驟 5: 光譜預處理 ===
# Savitzky-Golay 濾波（分別應用於訓練集和測試集）
X_train_sg = savgol_filter(X_train, window_length=11, polyorder=2, deriv=0)
X_test_sg = savgol_filter(X_test, window_length=11, polyorder=2, deriv=0)
print("✅ 訓練集和測試集 Savitzky-Golay 濾波完成（窗口大小=11，二次多項式）")

# 多重散射校正（MSC）
X_train_msc = msc(X_train_sg)
X_test_msc = msc(X_test_sg)
print("✅ 訓練集和測試集多重散射校正（MSC）完成")

# === 步驟 6: PCA 降維 ===
pca_model = PCA(n_components=20)
X_train_pca = pca_model.fit_transform(X_train)
X_test_pca = pca_model.transform(X_test)
print(f"✅ PCA 降維後訓練集形狀: {X_train_pca.shape}, 測試集形狀: {X_test_pca.shape}")

# === 步驟 7: 隨機森林參數調優 ===
param_grid = {
    'n_estimators': PARAMS['N_ESTIMATORS_RANGE'],
    'max_depth': PARAMS['MAX_DEPTH_RANGE'],
    'min_samples_split': PARAMS['MIN_SAMPLES_SPLIT_RANGE']
}
rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=KFold(n_splits=5, shuffle=True, random_state=42), 
                           scoring='neg_root_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train_pca, Y_train)
best_rf = grid_search.best_estimator_
best_params = grid_search.best_params_
print(f"✅ 最佳參數: n_estimators={best_params['n_estimators']}, max_depth={best_params['max_depth']}, min_samples_split={best_params['min_samples_split']}")

# === 步驟 8: 模型訓練 ===
best_rf.fit(X_train_pca, Y_train)
print(f"✅ 模型訓練完成，使用的參數: n_estimators={best_params['n_estimators']}, max_depth={best_params['max_depth']}, min_samples_split={best_params['min_samples_split']}")

# === 步驟 9: 繪製特徵重要性圖 ===
X_full_scaled = scaler_x.fit_transform(nir_transposed_df.loc[common_ids][mask][ash_mask])
X_full_sg = savgol_filter(X_full_scaled, window_length=11, polyorder=2, deriv=0)
X_full_msc = msc(X_full_sg)
full_rf = RandomForestRegressor(**best_params, random_state=42)
full_rf.fit(X_full_msc, Y_scaled)
feature_importances = full_rf.feature_importances_
wavelengths = nir_final_df['Wavelength'].values
sorted_idx = np.argsort(feature_importances)[::-1][:30]
top_features = feature_importances[sorted_idx]
top_wavelengths = wavelengths[sorted_idx]

plt.figure(figsize=(10, 6))
plt.bar(top_wavelengths, top_features, edgecolor='black')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Feature Importance')
plt.title('Top 30 Feature Importance for Ash Prediction')
plt.grid(True)
plt.tight_layout()
plt.show()
print("✅ 已顯示特徵重要性圖（前 30 個波長）")

# === 步驟 10: 評估結果 ===
Y_train_pred_scaled = best_rf.predict(X_train_pca)
Y_test_pred_scaled = best_rf.predict(X_test_pca)
Y_train_pred = scaler_y.inverse_transform(Y_train_pred_scaled.reshape(-1, 1)).ravel()
Y_test_pred = scaler_y.inverse_transform(Y_test_pred_scaled.reshape(-1, 1)).ravel()
Y_train_original = scaler_y.inverse_transform(Y_train.reshape(-1, 1)).ravel()
Y_test_original = scaler_y.inverse_transform(Y_test.reshape(-1, 1)).ravel()

train_mse = mean_squared_error(Y_train_original, Y_train_pred)
test_mse = mean_squared_error(Y_test_original, Y_test_pred)
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)
train_r2 = safe_r2_score(Y_train_original, Y_train_pred)
test_r2 = safe_r2_score(Y_test_original, Y_test_pred)

n_train = len(Y_train)
n_test = len(Y_test)
sec = np.sqrt(np.sum((Y_train_original - Y_train_pred) ** 2) / max(1, n_train - 1))
sep = np.sqrt(np.sum((Y_test_original - Y_test_pred) ** 2) / (n_test - 1))

# 5-fold 交叉驗證評估
cv_scores = cross_val_score(best_rf, X_train_pca, Y_train, cv=5, scoring='r2')
print(f"\n✅ 5-fold Cross-Validation R² Scores: {cv_scores}")
print(f"✅ Mean CV R²: {cv_scores.mean():.4f} (± {cv_scores.std() * 2:.4f})")

# 指標表格數據
metrics_data = {
    'Metric': ['MSE', 'RMSE', 'R² Score', 'SEC', 'SEP'],
    'Training Set': [f'{train_mse:.4f}', f'{train_rmse:.4f}', f'{train_r2:.4f}', f'{sec:.4f}', 'N/A'],
    'Test Set': [f'{test_mse:.4f}', f'{test_rmse:.4f}', f'{test_r2:.4f}', 'N/A', f'{sep:.4f}']
}
metrics_df = pd.DataFrame(metrics_data)
print("\n✅ Evaluation Metrics:")
print(metrics_df.to_string(index=False))

# 繪製指標表格
fig, ax = plt.subplots(figsize=(6, 4))
ax.axis('off')
table = ax.table(cellText=metrics_df.values, colLabels=metrics_df.columns, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.2)
plt.title('Evaluation Metrics for Training and Test Sets (Ash)')
plt.show()

# 繪製預測散點圖
print(f"Test Set True Values Range: Min {Y_test_original.min():.2f}, Max {Y_test_original.max():.2f}")
x_min = min(Y_test_original.min(), Y_train_original.min())
x_max = max(Y_test_original.max(), Y_train_original.max())
plt.figure(figsize=(6, 6))
plt.scatter(Y_test_original, Y_test_pred, alpha=0.7, label='Predicted vs Actual')
plt.plot([x_min, x_max], [x_min, x_max], 'r--', label='Ideal (y=x)')
plt.xlim(x_min, x_max)
plt.ylim(x_min, x_max)
plt.xlabel('Actual Ash Content (%)')
plt.ylabel('Predicted Ash Content (%)')
plt.title('Random Forest Regression: Actual vs Predicted Ash')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === 步驟 11: 生成參數和指標的 Excel 檔案 ===
common_params = {
    'Wavelength Range (nm)': PARAMS['WAVELENGTH_RANGE_STR'],
    'MCS Sampling Iterations': PARAMS['MCS_N_ITERATIONS'],
    'MCS PLS Components': PARAMS['MCS_N_COMPONENTS'],
    'Mahalanobis Distance Threshold (%)': PARAMS['MAHALANOBIS_THRESHOLD'],
    'Train-Test Split Ratio': PARAMS['TRAIN_TEST_RATIO_STR']
}

specific_params = {
    'Optimal n_estimators': best_params['n_estimators'],
    'Optimal Max Depth': best_params['max_depth'],
    'Optimal Min Samples Split': best_params['min_samples_split'],
    'PCA Components': X_train_pca.shape[1]
}

eval_metrics = {
    'RMSEC': f'{train_rmse:.4f}',
    'RMSEP': f'{test_rmse:.4f}',
    'R² C': f'{train_r2:.4f}',
    'R² P': f'{test_r2:.4f}',
    'SEC': f'{sec:.4f}',
    'SEP': f'{sep:.4f}'
}

output_data = {
    'Parameter/Metric': list(common_params.keys()) + list(specific_params.keys()) + list(eval_metrics.keys()),
    'Value': list(common_params.values()) + list(specific_params.values()) + list(eval_metrics.values()),
    'Category': ['Common Parameters'] * len(common_params) + ['Specific Parameters'] * len(specific_params) + ['Evaluation Metrics'] * len(eval_metrics)
}

output_df = pd.DataFrame(output_data)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_filename = f'RF_Ash_Reordered_{timestamp}.xlsx'
output_df.to_excel(output_filename, index=False)
print(f"✅ 已生成參數和指標檔案: {output_filename}")