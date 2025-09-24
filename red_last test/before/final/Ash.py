import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import chi2
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import time
import xlsxwriter
from sklearn.linear_model import Ridge
import os

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

# 中心化（Centering）預處理函數
def center(X):
    X = np.array(X)
    means = np.mean(X, axis=1, keepdims=True)
    X_centered = X - means
    return X_centered

# IUVE 波長選擇函數
def iuve_wavelength_selection(X, y, n_components, n_noise_vars= 150, cv_folds= 8, alpha=1.0):
    n_samples, n_features = X.shape
    noise = np.random.normal(0, 1e-6, size=(n_samples, n_noise_vars))
    X_augmented = np.hstack((X, noise))
    ridge = Ridge(alpha=alpha, fit_intercept=True, random_state=42)
    
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    coefs = np.zeros((cv_folds, X_augmented.shape[1]))
    for fold, (train_idx, _) in enumerate(kf.split(X_augmented)):
        X_train = X_augmented[train_idx]
        y_train = y[train_idx]
        ridge.fit(X_train, y_train)
        coefs[fold] = ridge.coef_.ravel()

    mean_coefs = np.mean(coefs, axis=0)
    std_coefs = np.std(coefs, axis=0)
    stability = np.abs(mean_coefs) / (std_coefs + 1e-10)
    stability_real = stability[:n_features]
    stability_noise = stability[n_features:]

    threshold = np.max(np.abs(stability_noise)) * 0.99
    selected_indices = np.where(stability_real > threshold)[0]
    print(f"✅ IUVE 選擇了 {len(selected_indices)} 個波長（共 {n_features} 個波長）")
    return selected_indices

# === 模型參數定義 ===
PARAMS = {
    'PLS': {
        'WAVELENGTH_RANGE': (650, 1099.5),
        'WAVELENGTH_RANGE_STR': '650-1099.5',
        'MCS_N_ITERATIONS': 1000,
        'MCS_N_COMPONENTS': 10,
        'MAHALANOBIS_THRESHOLD': 65,  # 調整為更寬鬆的閾值
        'TRAIN_TEST_RATIO': 0.2,
        'TRAIN_TEST_RATIO_STR': '80:20',
        'UVE_NOISE_VARIABLES': 700,
        'UVE_CV_FOLDS': 5
    },
    'DT': {
        'WAVELENGTH_RANGE': (650, 1099.5),
        'WAVELENGTH_RANGE_STR': '650-1099.5',
        'MCS_N_ITERATIONS': 1000,
        'MCS_N_COMPONENTS': 10,
        'MAHALANOBIS_THRESHOLD': 65,
        'TRAIN_TEST_RATIO': 0.2,
        'TRAIN_TEST_RATIO_STR': '80:20',
        'MAX_DEPTH_RANGE': [30,40,50],  # 縮小範圍以減少過擬合
        'MIN_SAMPLES_SPLIT_RANGE':[3,4]
    },
    'SVM': {
        'WAVELENGTH_RANGE': (650, 1099.5),
        'WAVELENGTH_RANGE_STR': '650-1099.5',
        'MCS_N_ITERATIONS': 1000,
        'MCS_N_COMPONENTS': 10,
        'MAHALANOBIS_THRESHOLD': 65,
        'TRAIN_TEST_RATIO': 0.2,
        'TRAIN_TEST_RATIO_STR': '80:20',
        'SVR_C_RANGE': [1,65],
        'SVR_EPSILON_RANGE': [0.122,0.261,0.5,1]
    },
    'RF': {
        'WAVELENGTH_RANGE': (650, 1099.5),
        'WAVELENGTH_RANGE_STR': '650-1099.5',
        'MCS_N_ITERATIONS': 500,
        'MCS_N_COMPONENTS': 10,
        'MAHALANOBIS_THRESHOLD': 65,
        'TRAIN_TEST_RATIO': 0.2,
        'TRAIN_TEST_RATIO_STR': '80:20',
        'MAX_DEPTH_RANGE': [3,7,10],
        'MIN_SAMPLES_SPLIT_RANGE': [2,3,5,10],
        'N_ESTIMATORS_RANGE': [20,30,60],
        'MAX_FEATURES_RANGE': ['sqrt', 'log2']
    }
}

# === 步驟 1: 數據獲取與處理 ===
try:
    nir_df = pd.read_csv(r'D:\meta\nir_data.csv')
    print("✅ nir_data.csv 前五行:")
    print(nir_df.head())
except FileNotFoundError:
    print("❌ 無法找到 nir_data.csv，請確認路徑。")
    exit()

try:
    trad_df = pd.read_excel(r'D:\meta\trad_data.xlsx')
    print("\n✅ trad_data.xlsx 前五行:")
    print(trad_df.head())
except FileNotFoundError:
    print("❌ 無法找到 trad_data.xlsx，請確認路徑。")
    exit()

# 目標成分
component = 'Ash'

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

# 處理光譜資料
nir_df.columns = ['Wavelength'] + [f'1-{i}' for i in range(1, len(nir_df.columns))]
nir_df = nir_df.drop(0).reset_index(drop=True)
nir_df['Wavelength'] = nir_df['Wavelength'].astype(float)

# 波長範圍過濾
wavelength_range = PARAMS['PLS']['WAVELENGTH_RANGE']  # (850, 1099.5)
nir_filtered_df = nir_df[(nir_df['Wavelength'] >= wavelength_range[0]) & 
                         (nir_df['Wavelength'] <= wavelength_range[1])]
selected_columns = ['Wavelength'] + [col for col in nir_df.columns if col != 'Wavelength']
nir_final_df = nir_filtered_df[selected_columns]
print(f"✅ 過濾後 NIR 數據波長範圍: {nir_final_df['Wavelength'].min()} - {nir_final_df['Wavelength'].max()}, 共 {len(nir_final_df)} 個波長點")

# 檢查波長點數（可選）
expected_wavelength_points = 500
if len(nir_final_df) != expected_wavelength_points:
    print(f"⚠️ 波長點數 {len(nir_final_df)} 不等於預期 {expected_wavelength_points}，請檢查數據格式")

# 更新 nir_df
nir_df = nir_final_df

# 對齊樣本
if trad_sample_col:
    trad_df[trad_sample_col] = [f'1-{i+1}' for i in range(len(trad_df))]
else:
    print("⚠️ 無法進行樣本對齊，缺少樣本編號欄")
    exit()

if component in trad_df.columns:
    trad_df = trad_df.dropna(subset=[component])
    Y = trad_df[component].values
    print(f"\n🎯 {component} 目標值 Y 前五筆:", Y[:5])
    print(f"\n🎯 {component} 目標值 Y 描述:", pd.Series(Y).describe())
else:
    print(f"❌ 缺少 '{component}' 欄位，無法建立模型")
    exit()

nir_transposed_df = nir_df.set_index('Wavelength').T
nir_transposed_df = nir_transposed_df.apply(pd.to_numeric, errors='coerce')
nir_transposed_df = nir_transposed_df.dropna(axis=1)
print("\n✅ 轉換後 NIR 資料形狀:", nir_transposed_df.shape)

sample_ids = trad_df[trad_sample_col].astype(str).tolist()
nir_transposed_df.index = nir_transposed_df.index.astype(str)
common_ids = list(set(sample_ids).intersection(set(nir_transposed_df.index)))
if len(common_ids) != len(sample_ids) or len(common_ids) != len(Y):
    print(f"❌ 對齊後樣本數不一致：X: {len(common_ids)}, Y: {len(Y)}")
    trad_df = trad_df[trad_df[trad_sample_col].astype(str).isin(common_ids)]
    Y = trad_df[component].values
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

# 使用 PLS 的參數作為異常值檢測的默認參數
n_simulations = PARAMS['PLS']['MCS_N_ITERATIONS']
np.random.seed(42)
md_thresholds = []
for _ in range(n_simulations):
    sim_data = np.random.multivariate_normal(np.mean(X_pca, axis=0), np.cov(X_pca, rowvar=False), size=X_pca.shape[0])
    cov_sim = np.cov(sim_data, rowvar=False)
    if np.any(np.isnan(cov_sim)) or np.any(np.isinf(cov_sim)):
        print("⚠️ 模擬數據協方差矩陣包含無效值")
        continue
    md_sim = mahalanobis_distance(sim_data)
    md_thresholds.append(np.percentile(md_sim, PARAMS['PLS']['MAHALANOBIS_THRESHOLD']))
md_threshold = np.mean(md_thresholds)

md = mahalanobis_distance(X_pca)
mask_md = md < md_threshold
print(f"✅ 馬氏距離法檢測到 {np.sum(~mask_md)} 個異常值，剩餘樣本數：{np.sum(mask_md)}")

# 蒙特卡洛採樣（MCS）異常值檢測
def monte_carlo_outlier_detection(X, y, n_iterations=PARAMS['PLS']['MCS_N_ITERATIONS'], 
                                 sample_ratio=0.8, n_components=PARAMS['PLS']['MCS_N_COMPONENTS'], 
                                 threshold_percentile=90):  # 降低閾值
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

# 範圍檢查
mask_range = (Y >= 0) & (Y <= 100)
X = X[mask_range]
Y = Y[mask_range]
print(f"✅ 移除異常範圍值後，剩餘樣本數：{X.shape[0]}")
print(f"✅ 移除後目標值 Y 描述:", pd.Series(Y).describe())

# === 步驟 3: NIR 光譜標準化與分配樣本集（SPXY） ===


scaler_x = StandardScaler()
X_scaled = scaler_x.fit_transform(X)
X_processed = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
print(f"✅ NIR 光譜標準化完成")

# 儲存結果的字典
results = {}

for model_name in ['PLS', 'DT', 'SVM', 'RF']:
    params = PARAMS[model_name]

    def spxy(X, Y, test_size=params['TRAIN_TEST_RATIO']):
        if isinstance(X, pd.DataFrame):
            indices = X.index
            columns = X.columns
            X = np.array(X)
        else:
            indices = np.arange(X.shape[0])
            columns = None
        
        Y = np.array(Y).reshape(-1, 1)
        n_samples = X.shape[0]
        n_test = int(np.floor(test_size * n_samples))
        if n_test < 1 or n_samples - n_test < 1:
            print(f"❌ 樣本數 {n_samples} 不足以分割訓練和測試集，需至少 {int(1/test_size)+1} 個樣本")
            exit()

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

        X_train = pd.DataFrame(X_train, index=indices[selected], columns=columns)
        X_test = pd.DataFrame(X_test, index=indices[remaining], columns=columns)

        return X_train, X_test, Y_train, Y_test
    
    scaler_y = StandardScaler()
    Y_scaled_filtered = scaler_y.fit_transform(Y.reshape(-1, 1)).ravel()
    X_train, X_test, Y_train, Y_test = spxy(X_processed, Y_scaled_filtered)
    print(f"✅ {model_name} SPXY 訓練集形狀: {X_train.shape}, 測試集形狀: {X_test.shape}")
    print(f"✅ {model_name} 訓練集 {component} 描述:", pd.Series(scaler_y.inverse_transform(Y_train.reshape(-1, 1)).ravel()).describe())
    print(f"✅ {model_name} 測試集 {component} 描述:", pd.Series(scaler_y.inverse_transform(Y_test.reshape(-1, 1)).ravel()).describe())

    # === 步驟 4: 光譜預處理（Centering） ===
    X_train_centered = center(X_train)
    X_train = pd.DataFrame(X_train_centered, index=X_train.index, columns=X_train.columns)
    X_test_centered = center(X_test)
    X_test = pd.DataFrame(X_test_centered, index=X_test.index, columns=X_test.columns)
    print(f"✅ {model_name} NIR 光譜中心化完成")

    # === 步驟 5: 特徵選擇與模型訓練 ===
    if model_name == 'PLS':
        selected_wavelength_indices = iuve_wavelength_selection(X_train, Y_train, n_components=params['MCS_N_COMPONENTS'])
        X_train_selected = X_train.iloc[:, selected_wavelength_indices]
        X_test_selected = X_test.iloc[:, selected_wavelength_indices]
        print(f"✅ {model_name} IUVE 波長選擇後，訓練集形狀: {X_train_selected.shape}, 測試集形狀: {X_test_selected.shape}")

        # === 步驟 6: PLS 參數調優 ===
        param_grid = {
            'n_components': range(2, min(X_train_selected.shape[1], 20) + 1)
        }
        pls = PLSRegression()
        grid_search = GridSearchCV(pls, param_grid, cv=KFold(n_splits=5, shuffle=True, random_state=42),
                                  scoring='r2', n_jobs=1, refit=True, error_score='raise')  # 單線程
        start_time = time.time()
        grid_search.fit(X_train_selected, Y_train)
        train_time = time.time() - start_time
    else:
        X_train_selected = X_train
        X_test_selected = X_test
        if model_name == 'DT':
            param_grid = {
                'max_depth': params['MAX_DEPTH_RANGE'],
                'min_samples_split': params['MIN_SAMPLES_SPLIT_RANGE']
            }
            model = DecisionTreeRegressor(random_state=42)
        elif model_name == 'SVM':
            param_grid = {
                'C': params['SVR_C_RANGE'],
                'epsilon': params['SVR_EPSILON_RANGE']
            }
            model = SVR(kernel='rbf', gamma='scale')
        else:  # RF
            param_grid = {
                'max_depth': params['MAX_DEPTH_RANGE'],
                'min_samples_split': params['MIN_SAMPLES_SPLIT_RANGE'],
                'n_estimators': params['N_ESTIMATORS_RANGE'],
                'max_features': params['MAX_FEATURES_RANGE']
            }
            model = RandomForestRegressor(random_state=42)

        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=1, refit=True, error_score='raise')  # 單線程
        start_time = time.time()
        grid_search.fit(X_train_selected, Y_train)
        train_time = time.time() - start_time

    print(f"✅ {model_name} 模型訓練完成，最佳參數: {grid_search.best_params_}, 訓練時間: {train_time:.2f} 秒")

    # === 步驟 7: 評估結果 ===
    start_pred_time = time.time()
    Y_train_pred = grid_search.predict(X_train_selected)
    Y_test_pred = grid_search.predict(X_test_selected)
    pred_time = time.time() - start_pred_time
    Y_train_pred_original = scaler_y.inverse_transform(Y_train_pred.reshape(-1, 1)).ravel()
    Y_test_pred_original = scaler_y.inverse_transform(Y_test_pred.reshape(-1, 1)).ravel()
    Y_train_original = scaler_y.inverse_transform(Y_train.reshape(-1, 1)).ravel()
    Y_test_original = scaler_y.inverse_transform(Y_test.reshape(-1, 1)).ravel()

    train_mse = mean_squared_error(Y_train_original, Y_train_pred_original)
    test_mse = mean_squared_error(Y_test_original, Y_test_pred_original)
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    train_r2 = safe_r2_score(Y_train_original, Y_train_pred_original)
    test_r2 = safe_r2_score(Y_test_original, Y_test_pred_original)
    sec = np.sqrt(np.sum((Y_train_original - Y_train_pred_original) ** 2) / max(1, len(Y_train) - 1))
    sep = np.sqrt(np.sum((Y_test_original - Y_test_pred_original) ** 2) / (len(Y_test) - 1))

    cv_scores = cross_val_score(grid_search.best_estimator_, X_train_selected, Y_train, cv=5, scoring='r2')
    print(f"\n✅ {model_name} 5-fold Cross-Validation Mean R²: {cv_scores.mean():.4f} (± {cv_scores.std() * 2:.4f})")
    print(f"✅ {model_name} Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
    print(f"✅ {model_name} Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
    print(f"✅ {model_name} SEC: {sec:.4f}, SEP: {sep:.4f}")

    # 繪製預測散點圖
    plt.figure(figsize=(6, 6))
    plt.scatter(Y_test_original, Y_test_pred_original, alpha=0.7, label='Predicted vs Actual')
    plt.plot([min(Y_test_original), max(Y_test_original)], [min(Y_test_original), max(Y_test_original)], 'r--', label='Ideal (y=x)')
    plt.xlabel(f'Actual {component} Content (%)')
    plt.ylabel(f'Predicted {component} Content (%)')
    plt.title(f'{model_name} Regression: Actual vs Predicted {component}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    print(f"✅ {model_name} 總訓練時間: {train_time:.2f} 秒，總預測時間: {pred_time:.2f} 秒")

    # 儲存結果
    results[model_name] = {
        'Train RMSE': train_rmse,
        'Test RMSE': test_rmse,
        'Train R²': train_r2,
        'Test R²': test_r2,
        'CV Mean R²': cv_scores.mean(),
        'CV Std R²': cv_scores.std() * 2,
        'SEC': sec,
        'SEP': sep,
        'Best Params': grid_search.best_params_,
        'Train Time (s)': train_time,
        'Pred Time (s)': pred_time,
        'Unique Params': {k: v for k, v in params.items() if k not in ['MCS_N_ITERATIONS', 'MCS_N_COMPONENTS', 'MAHALANOBIS_THRESHOLD', 'TRAIN_TEST_RATIO']},
        'Common Params': {
            'MCS_N_ITERATIONS': params['MCS_N_ITERATIONS'],
            'MCS_N_COMPONENTS': params['MCS_N_COMPONENTS'],
            'MAHALANOBIS_THRESHOLD': params['MAHALANOBIS_THRESHOLD'],
            'TRAIN_TEST_RATIO': params['TRAIN_TEST_RATIO']
        }
    }

# === 步驟 8: 將結果匯出到 Excel ===
with pd.ExcelWriter('model_evaluation_results.xlsx', engine='xlsxwriter') as writer:
    for model_name, result in results.items():
        df = pd.DataFrame({
            'Metric': ['Train RMSE', 'Test RMSE', 'Train R²', 'Test R²', 'CV Mean R²', 'CV Std R²', 'SEC', 'SEP', 'Train Time (s)', 'Pred Time (s)'],
            'Value': [
                result['Train RMSE'],
                result['Test RMSE'],
                result['Train R²'],
                result['Test R²'],
                result['CV Mean R²'],
                result['CV Std R²'],
                result['SEC'],
                result['SEP'],
                result['Train Time (s)'],
                result['Pred Time (s)']
            ]
        })
        df.to_excel(writer, sheet_name=model_name, index=False)

        # 將最佳參數寫入同一工作表
        param_df = pd.DataFrame({
            'Parameter': ['Best Params'] + [f'Unique Params - {k}' for k in result['Unique Params'].keys()] + [f'Common Params - {k}' for k in result['Common Params'].keys()],
            'Value': [str(result['Best Params'])] + [str(v) for v in result['Unique Params'].values()] + [str(v) for v in result['Common Params'].values()]
        })
        param_df.to_excel(writer, sheet_name=model_name, startrow=len(df) + 2, index=False)

    # 添加總結表
    summary_df = pd.DataFrame({
        model: {
            'Test R²': results[model]['Test R²'],
            'Test RMSE': results[model]['Test RMSE'],
            'Train Time (s)': results[model]['Train Time (s)']
        } for model in results
    }).T
    summary_df.to_excel(writer, sheet_name='Summary', index=True)
    print("\n✅ 模型比較總結:\n", summary_df)

print("✅ 結果已儲存至 model_evaluation_results.xlsx")