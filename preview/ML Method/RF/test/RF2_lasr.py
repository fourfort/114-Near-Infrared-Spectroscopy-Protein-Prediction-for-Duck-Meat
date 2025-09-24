import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.inspection import permutation_importance
import time
from datetime import datetime

# 設置 Times New Roman 字體
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# === 定義統一參數 ===
PARAMS = {
    'WAVELENGTH_RANGE': (850, 1099.5),  # 波長範圍
    'WAVELENGTH_RANGE_STR': '850-1099.5',
    'MCS_N_ITERATIONS': 200,
    'MCS_N_COMPONENTS': 7,
    'MAHALANOBIS_THRESHOLD': 80,
    'TRAIN_TEST_RATIO': 0.2,
    'TRAIN_TEST_RATIO_STR': '80:20',
    'SVR_C_RANGE': [10, 100, 500, 1000, 2000, 5000],  # 擴展 C 範圍
    'SVR_EPSILON_RANGE': [0.0001, 0.001, 0.01, 0.1],  # 擴展 epsilon 範圍
    'SVR_GAMMA_RANGE': [0.001, 0.01, 0.1, 'scale'],  # 新增 gamma 範圍
    'N_TOP_FEATURES': 50  # 選擇前 50 個重要波長
}

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

# 移除缺失值
trad_df = trad_df.dropna(subset=['Sample Number', 'Ash'])
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

# 處理光譜資料
nir_df.columns = ['Wavelength'] + [f'1-{i}' for i in range(1, len(nir_df.columns))]
nir_df = nir_df.drop(0).reset_index(drop=True)
nir_df['Wavelength'] = nir_df['Wavelength'].astype(float)
nir_filtered_df = nir_df[(nir_df['Wavelength'] >= PARAMS['WAVELENGTH_RANGE'][0]) & (nir_df['Wavelength'] <= PARAMS['WAVELENGTH_RANGE'][1])]
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
    Y = trad_df['Ash'].values
    print("\n🎯 目標值 Y 前五筆:", Y[:5])
    print("\n🎯 目標值 Y 描述:", pd.Series(Y).describe())
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

# === 步驟 2: 異常值處理（馬氏距離 + MCS） ===
pls = PLSRegression(n_components=PARAMS['MCS_N_COMPONENTS'])
X_pls = pls.fit_transform(X, Y)[0]
print(f"✅ PLS 降維後資料形狀: {X_pls.shape}")

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

np.random.seed(42)
md_thresholds = []
for i in range(PARAMS['MCS_N_ITERATIONS']):
    sim_data = np.random.multivariate_normal(np.mean(X_pls, axis=0), np.cov(X_pls, rowvar=False), size=X_pls.shape[0])
    md_sim = mahalanobis_distance(sim_data)
    md_thresholds.append(np.percentile(md_sim, PARAMS['MAHALANOBIS_THRESHOLD']))
    if (i + 1) % 50 == 0:
        print(f"✅ MCS 模擬進度: {i + 1}/{PARAMS['MCS_N_ITERATIONS']}")
md_threshold = np.mean(md_thresholds)

md = mahalanobis_distance(X_pls)
mask = md < md_threshold
X = X[mask]
Y = Y[mask]
print(f"✅ 馬氏距離 + MCS 剔除異常值後，剩餘樣本數: {X.shape[0]}")
print(f"✅ 剔除後目標值 Y 描述:", pd.Series(Y).describe())

# === 步驟 3: 樣本集分配（SPXY 演算法） ===
def spxy(X, y, test_size=0.2):
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    
    dist_X = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            dist_X[i, j] = np.sqrt(np.sum((X_scaled[i] - X_scaled[j]) ** 2))
            dist_X[j, i] = dist_X[i, j]
    
    dist_y = np.abs(y_scaled[:, None] - y_scaled[None, :])
    
    dist_X = dist_X / np.max(dist_X)
    dist_y = dist_y / np.max(dist_y)
    
    dist = dist_X + 2 * dist_y  # 增加 y 的權重
    
    max_dist_idx = np.unravel_index(np.argmax(dist), dist.shape)
    selected = list(max_dist_idx)
    remaining = list(set(range(n_samples)) - set(selected))
    
    while len(selected) < n_test:
        min_dist_to_selected = np.min([dist[i, selected] for i in remaining], axis=0)
        next_sample = remaining[np.argmax(min_dist_to_selected)]
        selected.append(next_sample)
        remaining.remove(next_sample)
    
    test_idx = selected
    train_idx = remaining
    
    return train_idx, test_idx

train_idx, test_idx = spxy(X, Y, test_size=PARAMS['TRAIN_TEST_RATIO'])
X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]
Y_train = Y[train_idx]
Y_test = Y[test_idx]

scaler_y = StandardScaler()
Y_train_scaled = scaler_y.fit_transform(Y_train.reshape(-1, 1)).ravel()
Y_test_scaled = scaler_y.transform(Y_test.reshape(-1, 1)).ravel()
print(f"✅ SPXY 分割完成，訓練集形狀: {X_train.shape}, 測試集形狀: {X_test.shape}")
print(f"✅ 訓練集目標值 Y 描述:", pd.Series(Y_train).describe())
print(f"✅ 測試集目標值 Y 描述:", pd.Series(Y_test).describe())

# === 步驟 4: 光譜預處理（Savitzky-Golay + MSC） ===
def msc(X):
    mean_spectrum = np.mean(X, axis=0)
    X_msc = np.zeros_like(X)
    for i in range(X.shape[0]):
        a, b = np.polyfit(mean_spectrum, X[i], 1)
        X_msc[i] = (X[i] - b) / a
    return X_msc

# 可視化 MSC 前後的光譜
plt.figure(figsize=(10, 6))
plt.plot(X_train.columns, X_train.iloc[0], label='Raw Spectrum (Sample 1)')
X_train_msc = msc(X_train.values)
plt.plot(X_train.columns, X_train_msc[0], label='MSC Spectrum (Sample 1)')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity')
plt.title('Raw vs MSC Preprocessed Spectrum')
plt.legend()
plt.tight_layout()
plt.show()

X_train_msc = msc(X_train.values)
X_test_msc = msc(X_test.values)
X_train_sg = savgol_filter(X_train_msc, window_length=21, polyorder=2, axis=1)  # 優化窗口大小和多項式階數
X_test_sg = savgol_filter(X_test_msc, window_length=21, polyorder=2, axis=1)
feature_names = X_train.columns
print("✅ Savitzky-Golay + MSC 預處理完成")

# === 步驟 5: 簡單特徵選擇 ===
pls = PLSRegression(n_components=PARAMS['MCS_N_COMPONENTS'])
pls.fit(X_train_sg, Y_train_scaled)
feature_importance = np.abs(pls.coef_)
sorted_idx = np.argsort(feature_importance)[::-1][:PARAMS['N_TOP_FEATURES']]
X_train_sg = X_train_sg[:, sorted_idx]
X_test_sg = X_test_sg[:, sorted_idx]
feature_names = feature_names[sorted_idx]
print(f"✅ 選擇前 {PARAMS['N_TOP_FEATURES']} 個重要波長，形狀: {X_train_sg.shape}")

# === 步驟 6: 模型訓練 ===
param_grid = {
    'C': PARAMS['SVR_C_RANGE'],
    'epsilon': PARAMS['SVR_EPSILON_RANGE'],
    'gamma': PARAMS['SVR_GAMMA_RANGE']
}
svm = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=5, scoring='r2')
start_train_time = time.time()
svm.fit(X_train_sg, Y_train_scaled)
train_time = time.time() - start_train_time
best_params = svm.best_params_
print(f"✅ SVM 模型訓練完成，最佳參數: {best_params}, 訓練時間: {train_time:.2f} 秒")

# === 步驟 7: 評估結果 ===
start_pred_time = time.time()
Y_train_pred_scaled = svm.predict(X_train_sg)
Y_test_pred_scaled = svm.predict(X_test_sg)
pred_time = time.time() - start_pred_time
Y_train_pred = scaler_y.inverse_transform(Y_train_pred_scaled.reshape(-1, 1)).ravel()
Y_test_pred = scaler_y.inverse_transform(Y_test_pred_scaled.reshape(-1, 1)).ravel()
Y_train_original = scaler_y.inverse_transform(Y_train_scaled.reshape(-1, 1)).ravel()
Y_test_original = scaler_y.inverse_transform(Y_test_scaled.reshape(-1, 1)).ravel()

train_mse = mean_squared_error(Y_train_original, Y_train_pred)
test_mse = mean_squared_error(Y_test_original, Y_test_pred)
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)
train_r2 = r2_score(Y_train_original, Y_train_pred)
test_r2 = r2_score(Y_test_original, Y_test_pred)

n_train = len(Y_train)
n_test = len(Y_test)
n_params = 1
sec = np.sqrt(np.sum((Y_train_original - Y_train_pred) ** 2) / (n_train - n_params - 1))
sep = np.sqrt(np.sum((Y_test_original - Y_test_pred) ** 2) / (n_test - 1))

cv_scores = cross_val_score(svm.best_estimator_, X_train_sg, Y_train_scaled, cv=5, scoring='r2')
print(f"\n✅ 5-fold Cross-Validation R² Scores: {cv_scores}")
print(f"✅ Mean CV R²: {cv_scores.mean():.4f} (± {cv_scores.std() * 2:.4f})")

# 指標表格
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

# 特徵重要性
perm_importance = permutation_importance(svm.best_estimator_, X_test_sg, Y_test_scaled, n_repeats=5, random_state=42, scoring='r2')  # 減少重複次數
feature_importance = perm_importance.importances_mean
feature_names_list = feature_names.tolist() if hasattr(feature_names, 'tolist') else feature_names
sorted_idx = np.argsort(feature_importance)[::-1][:20]
top_wavelengths = [float(feature_names_list[i]) for i in sorted_idx]
top_importance = [feature_importance[i] for i in sorted_idx]
plt.figure(figsize=(12, 6))
plt.bar([str(w) for w in top_wavelengths], top_importance)
plt.xticks(rotation=90)
plt.ylabel("Feature Importance")
plt.title("Top 20 Important Wavelengths in SVM for Ash")
plt.tight_layout()
plt.show()

# 預測散點圖
x_min = min(Y_test_original.min(), Y_train_original.min())
x_max = max(Y_test_original.max(), Y_train_original.max())
plt.figure(figsize=(6, 6))
plt.scatter(Y_test_original, Y_test_pred, alpha=0.7, label='Predicted vs Actual')
plt.plot([x_min, x_max], [x_min, x_max], 'r--', label='Ideal (y=x)')
plt.xlim(x_min, x_max)
plt.ylim(x_min, x_max)
plt.xlabel('Actual Ash Content (%)')
plt.ylabel('Predicted Ash Content (%)')
plt.title('SVM Regression: Actual vs Predicted Ash')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"✅ 總訓練時間: {train_time:.2f} 秒，總預測時間: {pred_time:.2f} 秒")

# === 步驟 8: 生成參數和指標的 Excel 檔案 ===
try:
    common_params = {
        'Wavelength Range (nm)': PARAMS['WAVELENGTH_RANGE_STR'],
        'MCS Sampling Iterations': PARAMS['MCS_N_ITERATIONS'],
        'MCS PLS Components': PARAMS['MCS_N_COMPONENTS'],
        'Mahalanobis Distance Threshold (%)': PARAMS['MAHALANOBIS_THRESHOLD'],
        'Train-Test Split Ratio': PARAMS['TRAIN_TEST_RATIO_STR']
    }

    specific_params = {
        'Optimal C': best_params['C'],
        'Optimal Epsilon': best_params['epsilon'],
        'Optimal Gamma': best_params['gamma'],
        'Top 20 Wavelengths by Importance': ', '.join([f'{w:.1f} (Imp: {i:.4f})' for w, i in zip(top_wavelengths, top_importance)])
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
    output_filename = f'SVR_Ash_No_CARS_{timestamp}.xlsx'
    output_df.to_excel(output_filename, index=False)
    print(f"✅ 已生成參數和指標檔案: {output_filename}")
except Exception as e:
    print(f"❌ Excel 檔案生成失敗: {str(e)}")