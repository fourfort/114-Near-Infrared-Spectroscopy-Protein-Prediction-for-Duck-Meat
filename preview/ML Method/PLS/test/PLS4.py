import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter
from scipy.stats import chi2, f
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization

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

# === 步驟 1: 數據獲取與處理 ===
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

# 處理光譜資料（波長範圍 850-1099.5 nm）
nir_df.columns = ['Wavelength'] + [f'1-{i}' for i in range(1, len(nir_df.columns))]
nir_df = nir_df.drop(0).reset_index(drop=True)
nir_df['Wavelength'] = nir_df['Wavelength'].astype(float)
nir_filtered_df = nir_df[(nir_df['Wavelength'] >= 850) & (nir_df['Wavelength'] <= 1099.5)]
selected_columns = ['Wavelength'] + [col for col in nir_df.columns if col != 'Wavelength']
nir_final_df = nir_filtered_df[selected_columns]
print(f"✅ 過濾後 NIR 數據波長範圍: {nir_final_df['Wavelength'].min()} - {nir_final_df['Wavelength'].max()}, 共 {len(nir_final_df)} 個波長點")

# 對齊樣本
if trad_sample_col:
    trad_df = trad_df.dropna(subset=[trad_sample_col])
    trad_df[trad_sample_col] = [f'1-{i+1}' for i in range(len(trad_df))]
else:
    print("⚠️ 無法進行樣本對齊，缺少樣本編號欄")
    exit()

if 'Moisture' in trad_df.columns:
    trad_df = trad_df.dropna(subset=['Moisture'])
    Y = trad_df['Moisture'].values
    print("\n🎯 目標值 Y 前五筆:", Y[:5])
    print("\n🎯 目標值 Y 描述:", pd.Series(Y).describe())
else:
    print("❌ 缺少 'Moisture' 欄位，無法建立模型")
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
    Y = trad_df['Moisture'].values
    X = nir_transposed_df.loc[common_ids]
else:
    X = nir_transposed_df.loc[sample_ids]
print(f"✅ 成功對齊資料，樣本數: {X.shape[0]}")

# === 步驟 2: 去除異常值（SPE 和 Hotelling’s T²） ===
def detect_outliers_spe_t2(X, n_components=15, alpha=0.05):
    X_np = X.to_numpy()
    pca = PCA(n_components=min(X_np.shape[0], X_np.shape[1], n_components))
    X_pca = pca.fit_transform(X_np)
    loadings = pca.components_.T
    residuals = X_np - np.dot(X_pca, loadings.T)
    
    spe = np.sum(residuals ** 2, axis=1)
    spe_threshold = np.percentile(spe, 100 * (1 - alpha))
    
    cov_matrix = np.cov(X_pca.T)
    cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6
    try:
        cov_inv = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-3
        cov_inv = np.linalg.inv(cov_matrix)
    t2 = np.sum((X_pca @ cov_inv) * X_pca, axis=1)
    t2_threshold = (n_components * (X_np.shape[0] - 1) * f.ppf(1 - alpha, n_components, X_np.shape[0] - n_components)) / (X_np.shape[0] - n_components)
    
    mask = (spe < spe_threshold) & (t2 < t2_threshold)
    print(f"✅ SPE 閾值: {spe_threshold:.4f}, Hotelling’s T² 閾值: {t2_threshold:.4f}")
    return mask

mask = detect_outliers_spe_t2(X, n_components=15, alpha=0.05)
X = X[mask]
Y = Y[mask]
print(f"✅ SPE 和 Hotelling’s T² 法剔除異常值後，剩餘樣本數: {X.shape[0]}")
print(f"✅ 剔除後目標值 Y 描述:", pd.Series(Y).describe())

# 水分含量範圍檢查（60-80%）
moisture_mask = (Y >= 60) & (Y <= 80)
if not np.all(moisture_mask):
    print(f"⚠️ 檢測到水分含量異常值（應在 60-80%）：{Y[~moisture_mask]}")
    X = X[moisture_mask]
    Y = Y[moisture_mask]
    print(f"✅ 移除水分含量異常值後，剩餘樣本數: {X.shape[0]}")
    print(f"✅ 移除後目標值 Y 描述:", pd.Series(Y).describe())

# 繪製水分含量分佈直方圖
plt.figure(figsize=(8, 6))
plt.hist(Y, bins=20, edgecolor='black')
plt.xlabel('Moisture Content (%)')
plt.ylabel('Frequency')
plt.title('Distribution of Moisture Content')
plt.grid(True)
plt.tight_layout()
plt.show()

# === 步驟 3: NIR 光譜標準化 ===
scaler_x = StandardScaler()
X_scaled = scaler_x.fit_transform(X)
X = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
print("✅ NIR 光譜標準化完成")

# === 步驟 4: 分配樣本集（Kennard-Stone 法） ===
def kennard_stone(X, test_size=0.2):
    X_np = X.to_numpy()
    n_samples = X_np.shape[0]
    if n_samples < 10:
        raise ValueError("樣本數過少，無法進行 Kennard-Stone 分割")
    n_test = int(n_samples * test_size)
    remaining_indices = list(range(n_samples))
    selected_indices = []
    
    dist_matrix = np.sqrt(((X_np[:, np.newaxis] - X_np) ** 2).sum(axis=2))
    
    max_dist_idx = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
    selected_indices.extend(max_dist_idx)
    remaining_indices.remove(max_dist_idx[0])
    if max_dist_idx[0] != max_dist_idx[1]:
        remaining_indices.remove(max_dist_idx[1])
    
    while len(selected_indices) < n_samples - n_test:
        min_distances = np.min(dist_matrix[remaining_indices][:, selected_indices], axis=1)
        max_min_dist_idx = np.argmax(min_distances)
        selected_idx = remaining_indices[max_min_dist_idx]
        selected_indices.append(selected_idx)
        remaining_indices.remove(selected_idx)
    
    test_indices = remaining_indices
    train_indices = selected_indices
    return train_indices, test_indices

scaler_y = StandardScaler()
Y_scaled = scaler_y.fit_transform(Y.reshape(-1, 1)).ravel()
train_indices, test_indices = kennard_stone(X, test_size=0.2)
X_train = X.iloc[train_indices]
X_test = X.iloc[test_indices]
Y_train = Y_scaled[train_indices]
Y_test = Y_scaled[test_indices]
print(f"✅ 訓練集形狀: {X_train.shape}, 測試集形狀: {X_test.shape}")
print(f"✅ 訓練集目標值 Y 描述:", pd.Series(scaler_y.inverse_transform(Y_train.reshape(-1, 1)).ravel()).describe())
print(f"✅ 測試集目標值 Y 描述:", pd.Series(scaler_y.inverse_transform(Y_test.reshape(-1, 1)).ravel()).describe())

# === 步驟 5: 光譜預處理（Savitzky-Golay 濾波） ===
X_train_sg = savgol_filter(X_train, window_length=51, polyorder=3, axis=1)
X_test_sg = savgol_filter(X_test, window_length=51, polyorder=3, axis=1)
print("✅ Savitzky-Golay 濾波完成")

# === 步驟 6: 特徵波長提取（CARS） ===
def cars(X, y, n_features=30, n_iterations=50):
    X_np = X if isinstance(X, np.ndarray) else X.to_numpy()
    n_samples, n_wavelengths = X_np.shape
    selected_indices = list(range(n_wavelengths))
    weights = np.ones(n_wavelengths)
    selected_history = []

    for _ in range(n_iterations):
        n_components = min(7, len(selected_indices), n_samples)
        if n_components < 1 or len(selected_indices) < n_features:
            break
        pls = PLSRegression(n_components=n_components)
        pls.fit(X_np[:, selected_indices], y)
        coef = np.abs(pls.coef_.ravel())
        weights[selected_indices] = coef / np.sum(coef)
        n_select = max(n_features, int(len(selected_indices) * 0.95))
        selected_indices = np.random.choice(selected_indices, size=n_select, replace=False, p=weights[selected_indices])
        selected_history.append(selected_indices.copy())

    final_indices = selected_history[-1]
    if len(final_indices) > n_features:
        final_indices = final_indices[:n_features]
    elif len(final_indices) < n_features:
        final_indices = selected_history[np.argmin([abs(len(indices) - n_features) for indices in selected_history])]
    return final_indices

selected_wavelengths = cars(X_train_sg, Y_train, n_features=30, n_iterations=50)
X_train_selected = X_train_sg[:, selected_wavelengths]
X_test_selected = X_test_sg[:, selected_wavelengths]
print(f"✅ CARS 選擇 {len(selected_wavelengths)} 個特徵波長")
print(f"✅ CARS 選擇的波長: {nir_final_df['Wavelength'].values[selected_wavelengths]}")

# 繪製特徵波長選擇圖
wavelengths = nir_final_df['Wavelength'].values
mean_spectrum = scaler_x.inverse_transform(X.mean().values.reshape(1, -1))[0]
plt.figure(figsize=(10, 6))
plt.plot(wavelengths, mean_spectrum, label='Mean Spectrum')
plt.scatter(wavelengths[selected_wavelengths], mean_spectrum[selected_wavelengths], 
            color='red', label='Selected Wavelengths', zorder=5)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Absorbance (Log(1/R))')
plt.title('CARS Selected Wavelengths')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === 步驟 7: 貝葉斯優化選擇最佳 n_components ===
def pls_cv_score(n_components, X, y):
    pls = PLSRegression(n_components=int(n_components))
    scores = []
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    for train_idx, test_idx in cv.split(X):
        X_train_cv, X_test_cv = X[train_idx], X[test_idx]
        y_train_cv, y_test_cv = y[train_idx], y[test_idx]
        pls.fit(X_train_cv, y_train_cv)
        y_pred_cv = pls.predict(X_test_cv)
        mse = mean_squared_error(y_test_cv, y_pred_cv)
        scores.append(np.sqrt(mse))
    return -np.mean(scores)

pbounds = {'n_components': (2, min(10, X_train_selected.shape[1], X_train_selected.shape[0]))}
optimizer = BayesianOptimization(
    f=lambda n_components: pls_cv_score(n_components, X_train_selected, Y_train),
    pbounds=pbounds,
    random_state=42,
    verbose=2
)
optimizer.maximize(init_points=10, n_iter=15)
best_lv = int(optimizer.max['params']['n_components'])
best_rmse = optimizer.max['target']
print(f"🔍 貝葉斯優化選擇最佳潛在變數數量: {best_lv}, 最佳 RMSE: {best_rmse:.4f}")

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
sec = np.sqrt(np.sum((Y_train_original - Y_train_pred) ** 2) / (n_train - best_lv - 1))
sep = np.sqrt(np.sum((Y_test_original - Y_test_pred) ** 2) / (n_test - 1))

# K-fold 交叉驗證評估
cv_scores = []
cv = KFold(n_splits=10, shuffle=True, random_state=42)
for train_idx, test_idx in cv.split(X_train_selected):
    X_train_cv, X_test_cv = X_train_selected[train_idx], X_train_selected[test_idx]
    y_train_cv, y_test_cv = Y_train[train_idx], Y_train[test_idx]
    pls_cv = PLSRegression(n_components=best_lv)
    pls_cv.fit(X_train_cv, y_train_cv)
    y_pred_cv = pls_cv.predict(X_test_cv)
    r2_cv = safe_r2_score(y_test_cv, y_pred_cv)
    cv_scores.append(r2_cv)
cv_scores = np.array(cv_scores)
print(f"\n✅ 10-Fold CV R² Scores: Mean {np.nanmean(cv_scores):.4f} (± {np.nanstd(cv_scores) * 2:.4f})")

metrics_data = {
    'Metric': ['MSE', 'RMSE', 'R² Score', 'SEC', 'SEP'],
    'Training Set': [f'{train_mse:.4f}', f'{train_rmse:.4f}', f'{train_r2:.4f}', f'{sec:.4f}', 'N/A'],
    'Test Set': [f'{test_mse:.4f}', f'{test_rmse:.4f}', f'{test_r2:.4f}', 'N/A', f'{sep:.4f}']
}
metrics_df = pd.DataFrame(metrics_data)
print("\n✅ Evaluation Metrics:")
print(metrics_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(6, 4))
ax.axis('off')
table = ax.table(cellText=metrics_df.values, colLabels=metrics_df.columns, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.2)
plt.title('Evaluation Metrics for Training and Test Sets')
plt.show()

print(f"Test Set True Values Range: Min {Y_test_original.min():.2f}, Max {Y_test_original.max():.2f}")
x_min = min(Y_test_original.min(), Y_train_original.min())
x_max = max(Y_test_original.max(), Y_train_original.max())
plt.figure(figsize=(6, 6))
plt.scatter(Y_test_original, Y_test_pred, alpha=0.7, label='Predicted vs Actual')
plt.plot([x_min, x_max], [x_min, x_max], 'r--', label='Ideal (y=x)')
plt.xlim(x_min, x_max)
plt.ylim(x_min, x_max)
plt.xlabel('Actual Moisture Content (%)')
plt.ylabel('Predicted Moisture Content (%)')
plt.title('PLS Regression: Actual vs Predicted Moisture')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()