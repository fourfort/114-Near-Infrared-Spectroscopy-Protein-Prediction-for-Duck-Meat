import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter
from scipy.stats import chi2
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 設置 Times New Roman 字體
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# === 步驟 1: 數據獲取與處理 ===
try:
    nir_df = pd.read_csv(r'D:\Users\wedke\document\extra document\大三上\113專題研究之製作\論文人生\紅外線\red_coding main\preview\nir_data.csv')
    print("✅ nir_data.csv 前五行:")
    print(nir_df.head())
except FileNotFoundError:
    print("❌ 無法找到 nir_data.csv，請確認路徑。")

try:
    trad_df = pd.read_excel(r'D:\Users\wedke\document\extra document\大三上\113專題研究之製作\論文人生\紅外線\red_coding main\preview\trad_data.xlsx')
    print("\n✅ trad_data.xlsx 前五行:")
    print(trad_df.head())
except FileNotFoundError:
    print("❌ 無法找到 trad_data.xlsx，請確認路徑。")

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
nir_filtered_df = nir_df[(nir_df['Wavelength'] >= 750) & (nir_df['Wavelength'] <= 1099.5)]
selected_columns = ['Wavelength'] + [col for col in nir_df.columns if col != 'Wavelength']
nir_final_df = nir_filtered_df[selected_columns]
print(f"✅ 過濾後 NIR 數據波長範圍: {nir_final_df['Wavelength'].min()} - {nir_final_df['Wavelength'].max()}, 共 {len(nir_final_df)} 個波長點")

# 對齊樣本
if trad_sample_col:
    trad_df[trad_sample_col] = [f'1-{i+1}' for i in range(len(trad_df))]
else:
    print("⚠️ 無法進行樣本對齊，缺少樣本編號欄")
    exit()

if 'Salt' in trad_df.columns:
    trad_df = trad_df.dropna(subset=['Salt'])
    Y = trad_df['Salt'].values
    print("\n🎯 目標值 Y 前五筆:", Y[:5])
    print("\n🎯 目標值 Y 描述:", pd.Series(Y).describe())
else:
    print("❌ 缺少 'Salt' 欄位，無法建立模型")
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
    Y = trad_df['Salt'].values
    X = nir_transposed_df.loc[common_ids]
else:
    X = nir_transposed_df.loc[sample_ids]
print(f"✅ 成功對齊資料，樣本數: {X.shape[0]}")

# === 步驟 2: 去除異常值 ===
pca = PCA(n_components=min(X.shape[0], X.shape[1], 8))
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

n_simulations = 1000
np.random.seed(42)
md_thresholds = []
for _ in range(n_simulations):
    sim_data = np.random.multivariate_normal(np.mean(X_pca, axis=0), np.cov(X_pca, rowvar=False), size=X_pca.shape[0])
    md_sim = mahalanobis_distance(sim_data)
    md_thresholds.append(np.percentile(md_sim, 85))
md_threshold = np.mean(md_thresholds)

md = mahalanobis_distance(X_pca)
mask = md < md_threshold
X = X[mask]
Y = Y[mask]
print(f"✅ 馬氏距離法剔除異常值後，剩餘樣本數: {X.shape[0]}")
print(f"✅ 剔除後目標值 Y 描述:", pd.Series(Y).describe())

# === 步驟 3: 分配樣本集 ===
scaler_y = StandardScaler()
Y_scaled = scaler_y.fit_transform(Y.reshape(-1, 1)).ravel()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_scaled, test_size=0.25, random_state=42, stratify=np.digitize(Y_scaled, np.percentile(Y_scaled, [5, 20, 40, 60, 80, 95])))
print(f"✅ 訓練集形狀: {X_train.shape}, 測試集形狀: {X_test.shape}")
print(f"✅ 訓練集目標值 Y 描述:", pd.Series(scaler_y.inverse_transform(Y_train.reshape(-1, 1)).ravel()).describe())
print(f"✅ 測試集目標值 Y 描述:", pd.Series(scaler_y.inverse_transform(Y_test.reshape(-1, 1)).ravel()).describe())

# === 步驟 4: 光譜預處理（Savitzky-Golay 濾波） ===
X_train_sg = savgol_filter(X_train, window_length=61, polyorder=3, axis=1)
X_test_sg = savgol_filter(X_test, window_length=61, polyorder=3, axis=1)
print("✅ Savitzky-Golay 濾波完成")

# === 步驟 5: 特徵波長提取（CARS） ===
def cars(X, y, n_features=25, n_iterations=20):
    n_samples, n_wavelengths = X.shape
    selected_indices = list(range(n_wavelengths))
    weights = np.ones(n_wavelengths)
    selected_history = []

    for _ in range(n_iterations):
        n_components = min(7, len(selected_indices), n_samples)
        if n_components < 1 or len(selected_indices) < 20:
            break
        pls = PLSRegression(n_components=n_components)
        pls.fit(X[:, selected_indices], y)
        coef = np.abs(pls.coef_.ravel())
        weights[selected_indices] = coef / np.sum(coef)
        n_select = max(20, int(len(selected_indices) * 0.98))
        selected_indices = np.random.choice(selected_indices, size=n_select, replace=False, p=weights[selected_indices])
        selected_history.append(selected_indices.copy())

    final_indices = selected_history[-1]
    if len(final_indices) > n_features:
        final_indices = final_indices[:n_features]
    elif len(final_indices) < n_features:
        final_indices = selected_history[np.argmin([abs(len(indices) - n_features) for indices in selected_history])]
    return final_indices

selected_wavelengths = cars(X_train_sg, Y_train, n_features=25)
X_train_selected = X_train_sg[:, selected_wavelengths]
X_test_selected = X_test_sg[:, selected_wavelengths]
print(f"✅ CARS 選擇 {len(selected_wavelengths)} 個特徵波長")

# 繪製特徵波長選擇圖
plt.figure(figsize=(10, 6))
plt.plot(X_train.columns.astype(float), X_train.mean(), label='Mean Spectrum')
plt.scatter(X_train.columns[selected_wavelengths].astype(float), X_train.mean().iloc[selected_wavelengths], 
            color='red', label='Selected Wavelengths', zorder=5)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Absorbance (Log(1/R))')
plt.title('CARS Selected Wavelengths')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === 步驟 6: 模型訓練 ===
n_components = min(12, X_train_selected.shape[1], X_train_selected.shape[0])
pls = PLSRegression(n_components=n_components)
pls.fit(X_train_selected, Y_train)
print(f"✅ 模型參數數量 (n_components): {n_components}")

# === 步驟 7: 評估結果 ===
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
train_r2 = r2_score(Y_train_original, Y_train_pred)
test_r2 = r2_score(Y_test_original, Y_test_pred)

# 計算 SEC 和 SEP
n_train = len(Y_train)
n_test = len(Y_test)
sec = np.sqrt(np.sum((Y_train_original - Y_train_pred) ** 2) / (n_train - n_components - 1))
sep = np.sqrt(np.sum((Y_test_original - Y_test_pred) ** 2) / (n_test - 1))

# 15-fold 交叉驗證評估
cv_scores = cross_val_score(pls, X_train_selected, Y_train, cv=15, scoring='r2')
print(f"\n✅ 15-fold Cross-Validation R² Scores: {cv_scores}")
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
plt.title('Evaluation Metrics for Training and Test Sets')
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
plt.title('PLS Regression: Actual vs Predicted Salt')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()