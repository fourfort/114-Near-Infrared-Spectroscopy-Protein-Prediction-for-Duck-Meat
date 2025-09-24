import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
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
    'WAVELENGTH_RANGE': (600, 1099.5),  # 波長範圍 (min, max)
    'WAVELENGTH_RANGE_STR': '600-1099.5',  # 波長範圍字串表示
    'MCS_N_ITERATIONS': 1000,  # MCS 抽樣次數
    'MCS_N_COMPONENTS': 7,  # MCS PLS 主成分數
    'MAHALANOBIS_THRESHOLD': 75,  # 馬氏距離閾值 (%)
    'TRAIN_TEST_RATIO': 0.2,  # 測試集比例 (0.2 表示 80:20)
    'TRAIN_TEST_RATIO_STR': '80:20',  # 訓練-測試比例字串表示
    'UVE_NOISE_VARIABLES': 500,  # UVE 隨機噪聲變量數量
    'UVE_CV_FOLDS': 5,  # UVE 交叉驗證折數
}

# 設置 Times New Roman 字體
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# 自定義 R² 計算函數，處理變異為零的情況
def safe_r2_score(y_true, y_pred):
    if len(y_true) <= 1 or np.var(y_true) == 0:
        return 0.0
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

# UVE 波長選擇函數
def uve_wavelength_selection(X, y, n_components, n_noise_vars=PARAMS['UVE_NOISE_VARIABLES'], cv_folds=PARAMS['UVE_CV_FOLDS']):
    """
    無信息變量消除 (UVE) 用於選擇重要波長
    X: 光譜數據 (樣本 x 波長)
    y: 目標值
    n_components: PLS 主成分數
    n_noise_vars: 添加的隨機噪聲變量數
    cv_folds: 交叉驗證折數
    返回: 選中的波長索引
    """
    n_samples, n_features = X.shape
    
    # 生成隨機噪聲變量，範圍與標準化後的光譜數據一致
    noise = np.random.normal(0, 1e-6, size=(n_samples, n_noise_vars))
    X_augmented = np.hstack((X, noise))
    
    # 初始化 PLS 模型
    pls = PLSRegression(n_components=n_components)
    
    # 進行交叉驗證以計算回歸係數
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    coefs = np.zeros((cv_folds, X_augmented.shape[1]))
    
    for fold, (train_idx, _) in enumerate(kf.split(X_augmented)):
        X_train = X_augmented[train_idx]
        y_train = y[train_idx]
        pls.fit(X_train, y_train)
        coefs[fold] = pls.coef_.ravel()
    
    # 計算穩定性指標 (回歸係數均值除以標準差)
    mean_coefs = np.mean(coefs, axis=0)
    std_coefs = np.std(coefs, axis=0)
    stability = np.abs(mean_coefs) / (std_coefs + 1e-10)  # 避免除零
    
    # 分離原始變量和噪聲變量的穩定性
    stability_real = stability[:n_features]
    stability_noise = stability[n_features:]
    
    # 使用噪聲變量的穩定性閾值選擇重要波長
    threshold = np.percentile(stability_noise, 80)  # 使用 95% 分位數作為閾值
    selected_indices = np.where(stability_real > threshold)[0]
    
    print(f"✅ UVE 選擇了 {len(selected_indices)} 個波長（共 {n_features} 個波長）")
    
    return selected_indices

# === 步驟 1: 數據獲取與處理 ===
try:
    nir_df = pd.read_csv(r'nir_data.csv')
    print("✅ nir_data.csv 前五行:")
    print(nir_df.head())
except FileNotFoundError:
    print("❌ 無法找到 nir_data.csv，請確認檔案存在於當前目錄。")
    exit()

try:
    trad_df = pd.read_excel(r'trad_data.xlsx')
    print("\n✅ trad_data.xlsx 前五行:")
    print(trad_df.head())
except FileNotFoundError:
    print("❌ 無法找到 trad_data.xlsx，請確認檔案存在於當前目錄。")
    exit()

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

if 'Protein' in trad_df.columns:
    trad_df = trad_df.dropna(subset=['Protein'])
    Y = trad_df['Protein'].values
    print("\n🎯 目標值 Y 前五筆:", Y[:5])
    print("\n🎯 目標值 Y 描述:", pd.Series(Y).describe())
else:
    print("❌ 缺少 'Protein' 欄位，無法建立模型")
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
    Y = trad_df['Protein'].values
    X = nir_transposed_df.loc[common_ids]
else:
    X = nir_transposed_df.loc[sample_ids]
print(f"✅ 成功對齊資料，樣本數: {X.shape[0]}")

# === 步驟 2: 去除異常值 ===
# 馬氏距離法
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
        
        pls = PLSRegression(n_components=min(n_components, X_subset_scaled.shape[1], n_subset))
        pls.fit(X_subset_scaled, y_subset)
        
        X_scaled = scaler.transform(X)
        y_pred = pls.predict(X_scaled).ravel()
        residuals += (y - y_pred) ** 2
    
    residuals = residuals / n_iterations
    threshold = np.percentile(residuals, threshold_percentile)
    mask_mcs = residuals < threshold
    print(f"✅ MCS 檢測到 {np.sum(~mask_mcs)} 個異常值，剩餘樣本數：{np.sum(mask_mcs)}")
    return mask_mcs

# 結合馬氏距離和 MCS（交集）
scaler_y_temp = StandardScaler()
Y_scaled_temp = scaler_y_temp.fit_transform(Y.reshape(-1, 1)).ravel()
mask_mcs = monte_carlo_outlier_detection(X, Y_scaled_temp)
mask = mask_md & mask_mcs
X = X[mask]
Y = Y[mask]
print(f"✅ 結合馬氏距離和 MCS 後，剩餘樣本數：{X.shape[0]}")
print(f"✅ 剔除後目標值 Y 描述:", pd.Series(Y).describe())

# 蛋白質含量範圍檢查（0-50%）
protein_mask = (Y >= 0) & (Y <= 50)
if not np.all(protein_mask):
    print(f"⚠️ 檢測到蛋白質含量異常值（應在 0-50%）：{Y[~protein_mask]}")
    X = X[protein_mask]
    Y = Y[protein_mask]
    print(f"✅ 移除蛋白質異常值後，剩餘樣本數：{X.shape[0]}")
    print(f"✅ 移除後目標值 Y 描述:", pd.Series(Y).describe())

# 繪製蛋白質含量分佈直方圖
plt.figure(figsize=(8, 6))
plt.hist(Y, bins=20, edgecolor='black')
plt.xlabel('Protein Content (%)')
plt.ylabel('Frequency')
plt.title('Distribution of Protein Content')
plt.grid(True)
plt.tight_layout()
plt.show()

# === 步驟 3: NIR 光譜標準化 ===
scaler_x = StandardScaler()
X_scaled = scaler_x.fit_transform(X)
X = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
print("✅ NIR 光譜標準化完成")

# === 步驟 4: 分配樣本集（SPXY） ===
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
print(f"✅ 訓練集目標值 Y 描述:", pd.Series(scaler_y.inverse_transform(Y_train.reshape(-1, 1)).ravel()).describe())
print(f"✅ 測試集目標值 Y 描述:", pd.Series(scaler_y.inverse_transform(Y_test.reshape(-1, 1)).ravel()).describe())

# === 步驟 5: 光譜預處理（Savitzky-Golay 濾波） ===
X_train_sg = savgol_filter(X_train, window_length=51, polyorder=3, axis=1)
X_test_sg = savgol_filter(X_test, window_length=51, polyorder=3, axis=1)
print("✅ Savitzky-Golay 濾波完成")

# === 步驟 6: 無信息變量消除（UVE）波長選擇 ===
# 使用 UVE 選擇重要波長
selected_wavelength_indices = uve_wavelength_selection(X_train_sg, Y_train, n_components=PARAMS['MCS_N_COMPONENTS'])

# 根據選中的波長索引提取訓練集和測試集數據
X_train_selected = X_train_sg[:, selected_wavelength_indices]
X_test_selected = X_test_sg[:, selected_wavelength_indices]
print(f"✅ UVE 波長選擇後，訓練集形狀: {X_train_selected.shape}, 測試集形狀: {X_test_selected.shape}")

# === 步驟 7: 交叉驗證找最佳 n_components ===
max_lv = min(10, X_train_selected.shape[1], X_train_selected.shape[0])
rmse_cv = []

for lv in range(1, max_lv + 1):
    pls_cv = PLSRegression(n_components=lv)
    scores = cross_val_score(
        pls_cv,
        X_train_selected,
        Y_train,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring='neg_root_mean_squared_error'
    )
    rmse_cv.append(-scores.mean())

best_lv = np.argmin(rmse_cv) + 1
print(f"🔍 使用交叉驗證選擇最佳潛在變數數量為: {best_lv}")

# === 步驟 8: 模型訓練（使用最佳 n_components） ===
pls = PLSRegression(n_components=best_lv)
pls.fit(X_train_selected, Y_train)
print(f"✅ 模型訓練完成，使用的潛在變數數量 (n_components): {best_lv}")

# === 步驟 9: 評估結果 ===
Y_train_pred_scaled = pls.predict(X_train_selected)
Y_test_pred_scaled = pls.predict(X_test_selected)
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
sec = np.sqrt(np.sum((Y_train_original - Y_train_pred) ** 2) / max(1, n_train - best_lv - 1))
sep = np.sqrt(np.sum((Y_test_original - Y_test_pred) ** 2) / (n_test - 1))

# 5-fold 交叉驗證評估
cv_scores = cross_val_score(pls, X_train_selected, Y_train, cv=5, scoring='r2')
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
plt.title('Evaluation Metrics for Training and Test Sets (Protein)')
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
plt.xlabel('Actual Protein Content (%)')
plt.ylabel('Predicted Protein Content (%)')
plt.title('PLS Regression: Actual vs Predicted Protein')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === 步驟 10: 生成參數和指標的 Excel 檔案 ===
# 統一參數
common_params = {
    'Wavelength Range (nm)': PARAMS['WAVELENGTH_RANGE_STR'],
    'MCS Sampling Iterations': PARAMS['MCS_N_ITERATIONS'],
    'MCS PLS Components': PARAMS['MCS_N_COMPONENTS'],
    'Mahalanobis Distance Threshold (%)': PARAMS['MAHALANOBIS_THRESHOLD'],
    'Train-Test Split Ratio': PARAMS['TRAIN_TEST_RATIO_STR'],
    'UVE Noise Variables': PARAMS['UVE_NOISE_VARIABLES'],
    'UVE CV Folds': PARAMS['UVE_CV_FOLDS']
}

# 特有參數
specific_params = {
    'Optimal PLS Latent Variables': best_lv,
    'Number of Selected Wavelengths (UVE)': len(selected_wavelength_indices)
}

# 評估指標
eval_metrics = {
    'RMSEC': f'{train_rmse:.4f}',
    'RMSEP': f'{test_rmse:.4f}',
    'R² C': f'{train_r2:.4f}',
    'R² P': f'{test_r2:.4f}',
    'SEC': f'{sec:.4f}',
    'SEP': f'{sep:.4f}'
}

# 合併所有參數和指標
output_data = {
    'Parameter/Metric': list(common_params.keys()) + list(specific_params.keys()) + list(eval_metrics.keys()),
    'Value': list(common_params.values()) + list(specific_params.values()) + list(eval_metrics.values()),
    'Category': ['Common Parameters'] * len(common_params) + ['Specific Parameters'] * len(specific_params) + ['Evaluation Metrics'] * len(eval_metrics)
}

# 創建 DataFrame
output_df = pd.DataFrame(output_data)

# 生成帶時間戳的檔案名
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_filename = f'PLS_Protein_UVE_{timestamp}.xlsx'

# 輸出到 Excel
output_df.to_excel(output_filename, index=False)
print(f"✅ 已生成參數和指標檔案: {output_filename}")