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
from scipy.signal import savgol_filter
from scipy.stats import chi2
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import time
import xlsxwriter

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

# === 模型參數定義 ===
PARAMS = {
    'PLS': {
        'WAVELENGTH_RANGE': (850, 1099.5),
        'WAVELENGTH_RANGE_STR': '850-1099.5',
        'MCS_N_ITERATIONS': 1000,
        'MCS_N_COMPONENTS': 10,
        'MAHALANOBIS_THRESHOLD': 65,
        'TRAIN_TEST_RATIO': 0.2,
        'TRAIN_TEST_RATIO_STR': '80:20',
        'CARS_N_FEATURES': 14,
        'CARS_N_ITERATIONS': 50
    },
    'DT': {
        'WAVELENGTH_RANGE': (850, 1099.5),
        'WAVELENGTH_RANGE_STR': '850-1099.5',
        'MCS_N_ITERATIONS': 1000,
        'MCS_N_COMPONENTS': 10,
        'MAHALANOBIS_THRESHOLD': 65,
        'TRAIN_TEST_RATIO': 0.2,
        'TRAIN_TEST_RATIO_STR': '80:20',
        'MAX_DEPTH_RANGE': [3,20,30],
        'MIN_SAMPLES_SPLIT_RANGE': [2,3,5]
    },
    'SVM': {
        'WAVELENGTH_RANGE': (850, 1099.5),
        'WAVELENGTH_RANGE_STR': '850-1099.5',
        'MCS_N_ITERATIONS': 1000,
        'MCS_N_COMPONENTS': 10,
        'MAHALANOBIS_THRESHOLD': 65,
        'TRAIN_TEST_RATIO': 0.2,
        'TRAIN_TEST_RATIO_STR': '80:20',
        'SVR_C_RANGE': [1,2,3800],
        'SVR_EPSILON_RANGE': [0.1358,0.5,0.4]
    },
    'RF': {
        'WAVELENGTH_RANGE': (850, 1099.5),
        'WAVELENGTH_RANGE_STR': '850-1099.5',
        'MCS_N_ITERATIONS': 500,
        'MCS_N_COMPONENTS': 10,
        'MAHALANOBIS_THRESHOLD': 65,
        'TRAIN_TEST_RATIO': 0.2,
        'TRAIN_TEST_RATIO_STR': '80:20',
        'MAX_DEPTH_RANGE': [3,7,10],
        'MIN_SAMPLES_SPLIT_RANGE': [2,3,5,10],
        'N_ESTIMATORS_RANGE': [30,60],
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
component = 'Moisture'

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

# 範圍檢查
mask_range = (Y >= 0) & (Y <= 100)
X = X[mask_range]
Y = Y[mask_range]
print(f"✅ 移除異常範圍值後，剩餘樣本數：{X.shape[0]}")
print(f"✅ 移除後目標值 Y 描述:", pd.Series(Y).describe())

predictions = {}
# === 步驟 3: NIR 光譜預處理（移除獨立標準化，移到步驟 5 結合 SG 濾波） ===
print("✅ 跳過獨立標準化，將在步驟 5 結合 Savitzky-Golay 濾波與 Autoscaling")

# === 步驟 4: 分配樣本集（SPXY） ===
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

        X_train = pd.DataFrame(X_train, index=indices[selected], columns=columns) if columns is not None else X_train
        X_test = pd.DataFrame(X_test, index=indices[remaining], columns=columns) if columns is not None else X_test

        return X_train, X_test, Y_train, Y_test
    scaler_y = StandardScaler()
    Y_scaled = scaler_y.fit_transform(Y.reshape(-1, 1)).ravel()
    X_train, X_test, Y_train, Y_test = spxy(X, Y_scaled)
    print(f"✅ {model_name} SPXY 訓練集形狀: {X_train.shape}, 測試集形狀: {X_test.shape}")
    print(f"✅ {model_name} 訓練集 {component} 描述:", pd.Series(scaler_y.inverse_transform(Y_train.reshape(-1, 1)).ravel()).describe())
    print(f"✅ {model_name} 測試集 {component} 描述:", pd.Series(scaler_y.inverse_transform(Y_test.reshape(-1, 1)).ravel()).describe())

    # === 步驟 5: 光譜預處理（結合 Savitzky-Golay 濾波與 Autoscaling） ===
    # 首先應用 Savitzky-Golay 濾波進行光譜平滑
    X_train_sg = savgol_filter(X_train, window_length=51, polyorder=3, axis=1)
    X_test_sg = savgol_filter(X_test, window_length=51, polyorder=3, axis=1)

    # 隨後應用 Autoscaling (StandardScaler) 進行標準化
    scaler_x = StandardScaler()
    X_train_preprocessed = scaler_x.fit_transform(X_train_sg)
    X_test_preprocessed = scaler_x.transform(X_test_sg)

    # 將預處理後的數據轉為 DataFrame，保留索引和欄名（如果適用）
    X_train_preprocessed = pd.DataFrame(X_train_preprocessed, index=X_train.index if isinstance(X_train, pd.DataFrame) else None, 
                                        columns=X_train.columns if isinstance(X_train, pd.DataFrame) else None)
    X_test_preprocessed = pd.DataFrame(X_test_preprocessed, index=X_test.index if isinstance(X_test, pd.DataFrame) else None, 
                                       columns=X_test.columns if isinstance(X_test, pd.DataFrame) else None)
    print(f"✅ {model_name} 光譜預處理完成：Savitzky-Golay 濾波與 Autoscaling 結合")

    # === 步驟 6: CARS 特徵選擇 ===
    if model_name == 'PLS':
        def cars_feature_selection(X, y, n_features=params['CARS_N_FEATURES'], n_iterations=params['CARS_N_ITERATIONS'], random_state=42):
            n_samples, n_wavelengths = X.shape
            selected_indices = list(range(n_wavelengths))
            weights = np.ones(n_wavelengths)
            selected_history = []

            np.random.seed(random_state)  # 確保隨機性一致
            for _ in range(n_iterations):
                n_components = min(params['MCS_N_COMPONENTS'], len(selected_indices), n_samples)
                if n_components < 1 or len(selected_indices) < n_features:
                    break
                pls = PLSRegression(n_components=n_components)
                # 處理 DataFrame 或 NumPy 陣列
                X_subset = X.iloc[:, selected_indices] if isinstance(X, pd.DataFrame) else X[:, selected_indices]
                pls.fit(X_subset, y.reshape(-1, 1))
                coef = np.abs(pls.coef_.ravel())
                weights[selected_indices] = coef / np.sum(coef)  # 標準化權重
                n_select = max(n_features, int(len(selected_indices) * 0.98))  # 保留 98% 的波長
                selected_indices = np.random.choice(selected_indices, size=n_select, replace=False, p=weights[selected_indices])
                selected_history.append(selected_indices.copy())

            # 選擇最終波長
            final_indices = selected_history[-1] if selected_history else list(range(n_wavelengths))
            if len(final_indices) > n_features:
                final_indices = final_indices[:n_features]
            elif len(final_indices) < n_features:
                final_indices = selected_history[np.argmin([abs(len(indices) - n_features) for indices in selected_history])]
            if len(final_indices) < 2:
                print(f"⚠️ CARS 選擇的特徵數 ({len(final_indices)}) 過少，回退到全波長")
                final_indices = np.arange(n_wavelengths)
            
            return final_indices

        # 確保參數定義
        if 'CARS_N_FEATURES' not in params:
            params['CARS_N_FEATURES'] = 20  # 預設值，可根據需求調整
        if 'CARS_N_ITERATIONS' not in params:
            params['CARS_N_ITERATIONS'] = 50  # 預設值，可根據需求調整

        # 調用 CARS 特徵選擇
        selected_features = cars_feature_selection(X_train_preprocessed, Y_train, 
                                                n_features=params['CARS_N_FEATURES'], 
                                                n_iterations=params['CARS_N_ITERATIONS'])
        X_train_cars = X_train_preprocessed.iloc[:, selected_features] if isinstance(X_train_preprocessed, pd.DataFrame) else X_train_preprocessed[:, selected_features]
        X_test_cars = X_test_preprocessed.iloc[:, selected_features] if isinstance(X_test_preprocessed, pd.DataFrame) else X_test_preprocessed[:, selected_features]
        print(f"✅ {model_name} CARS 選擇了 {len(selected_features)} 個特徵波長")
        
        # 輸出選擇的波長值
        selected_wavelengths = nir_df['Wavelength'].values[selected_features]
        print(f"✅ {model_name} CARS 選擇的波長: {selected_wavelengths}")

        # === 步驟 7: PLS 參數調優 ===
        param_grid = {
            'n_components': range(2, min(X_train_cars.shape[1], 20) + 1)
        }
        pls = PLSRegression()
        grid_search = GridSearchCV(pls, param_grid, cv=KFold(n_splits=5, shuffle=True, random_state=42),
                                scoring='r2', n_jobs=-1, refit=True, error_score='raise')
        start_time = time.time()
        grid_search.fit(X_train_cars, Y_train)
        train_time = time.time() - start_time
        print(f"✅ {model_name} 模型訓練完成，最佳參數: {grid_search.best_params_}, 訓練時間: {train_time:.2f} 秒")
    else:
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

        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1, refit=True, error_score='raise')
        start_time = time.time()
        grid_search.fit(X_train_preprocessed, Y_train)
        train_time = time.time() - start_time

    print(f"✅ {model_name} 模型訓練完成，最佳參數: {grid_search.best_params_}, 訓練時間: {train_time:.2f} 秒")

    # === 步驟 8: 評估結果 ===
    start_pred_time = time.time()
    if model_name == 'PLS':
        Y_train_pred = grid_search.predict(X_train_cars)
        Y_test_pred = grid_search.predict(X_test_cars)
    else:
        Y_train_pred = grid_search.predict(X_train_preprocessed)
        Y_test_pred = grid_search.predict(X_test_preprocessed)
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

    cv_scores = cross_val_score(grid_search.best_estimator_, 
                               X_train_cars if model_name == 'PLS' else X_train_preprocessed, 
                               Y_train, cv=5, scoring='r2')
    print(f"\n✅ {model_name} 5-fold Cross-Validation Mean R²: {cv_scores.mean():.4f} (± {cv_scores.std() * 2:.4f})")
    print(f"✅ {model_name} Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
    print(f"✅ {model_name} Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
    print(f"✅ {model_name} SEC: {sec:.4f}, SEP: {sep:.4f}")

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
    
    # 儲存預測數據到 predictions 字典
    predictions[model_name] = {
        'Y_test_original': Y_test_original,
        'Y_test_pred_original': Y_test_pred_original
    }

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

# === 步驟 8: 將結果匯出到 Excel ===
with pd.ExcelWriter('model_evaluation_results.xlsx', engine='xlsxwriter') as writer:
    # 定義 Excel 格式
    workbook = writer.book
    format_num = workbook.add_format({'num_format': '0.0000'})  # 四位小數格式
    format_text = workbook.add_format({'text_wrap': True})  # 文字換行
    format_header = workbook.add_format({'bold': True, 'align': 'center', 'border': 1})  # 表頭格式

    for model_name in ['PLS', 'DT', 'SVM', 'RF']:
        # 輸出評估指標
        df = pd.DataFrame({
            'Metric': ['Train RMSE', 'Test RMSE', 'Train R²', 'Test R²', 'CV Mean R²', 'CV Std R²', 'SEC', 'SEP', 'Train Time (s)', 'Pred Time (s)'],
            'Value': [
                results[model_name]['Train RMSE'],
                results[model_name]['Test RMSE'],
                results[model_name]['Train R²'],
                results[model_name]['Test R²'],
                results[model_name]['CV Mean R²'],
                results[model_name]['CV Std R²'],
                results[model_name]['SEC'],
                results[model_name]['SEP'],
                results[model_name]['Train Time (s)'],
                results[model_name]['Pred Time (s)']
            ]
        })
        df.to_excel(writer, sheet_name=model_name, index=False, startrow=1)

        # 設置評估指標工作表的格式
        worksheet = writer.sheets[model_name]
        worksheet.write(0, 0, f'{model_name} Model Evaluation', format_header)
        worksheet.set_column('A:A', 15, format_text)  # Metric 欄寬
        worksheet.set_column('B:B', 12, format_num)   # Value 欄寬，四位小數

        # 輸出最佳參數
        param_df = pd.DataFrame({
            'Parameter': ['Best Params'] + [f'Unique Params - {k}' for k in results[model_name]['Unique Params'].keys()] + [f'Common Params - {k}' for k in results[model_name]['Common Params'].keys()],
            'Value': [str(results[model_name]['Best Params'])] + [str(v) for v in results[model_name]['Unique Params'].values()] + [str(v) for v in results[model_name]['Common Params'].values()]
        })
        param_df.to_excel(writer, sheet_name=model_name, startrow=len(df) + 3, index=False)

        # 設置參數部分的格式
        worksheet.write(len(df) + 2, 0, 'Parameters', format_header)
        worksheet.set_column('A:A', 25, format_text)  # Parameter 欄寬
        worksheet.set_column('B:B', 20, format_text)  # Value 欄寬

print("✅ 評估指標和參數已儲存至 model_evaluation_results.xlsx")

# === 步驟 9: 將散點圖數據匯出到獨立的 Excel 文件 ===
for model_name in ['PLS', 'DT', 'SVM', 'RF']:
    # 創建散點圖數據的 DataFrame
    scatter_df = pd.DataFrame({
        'Actual': predictions[model_name]['Y_test_original'],
        'Predicted': predictions[model_name]['Y_test_pred_original']
    })
    
    # 為每個模型創建獨立的 Excel 文件
    try:
        with pd.ExcelWriter(f'{model_name}_Scatter_Data.xlsx', engine='xlsxwriter') as writer:
            # 定義 Excel 格式
            workbook = writer.book
            format_num = workbook.add_format({'num_format': '0.0000'})  # 四位小數格式
            format_header = workbook.add_format({'bold': True, 'align': 'center', 'border': 1})  # 表頭格式
            
            # 輸出散點圖數據
            scatter_df.to_excel(writer, sheet_name='Scatter_Data', index=False, startrow=1)
            
            # 設置散點圖數據工作表的格式
            worksheet = writer.sheets['Scatter_Data']
            worksheet.write(0, 0, f'{model_name} Scatter Data', format_header)
            worksheet.set_column('A:A', 12, format_num)  # Actual 欄寬
            worksheet.set_column('B:B', 12, format_num)  # Predicted 欄寬
        
        print(f"✅ {model_name} 散點圖數據已儲存至 {model_name}_Scatter_Data.xlsx")
    except Exception as e:
        print(f"❌ 儲存 {model_name}_Scatter_Data.xlsx 時發生錯誤: {e}")
