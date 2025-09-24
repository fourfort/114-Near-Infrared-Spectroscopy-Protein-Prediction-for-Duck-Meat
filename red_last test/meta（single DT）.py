import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import pairwise_distances
from statsmodels.stats.outliers_influence import variance_inflation_factor
from datetime import datetime

# 定義統一參數
PARAMS = {
    'WAVELENGTH_RANGE': (850, 1099.5),
    'WAVELENGTH_RANGE_STR': '800-1099.5',
    'MCS_N_ITERATIONS': 1000,
    'MCS_N_COMPONENTS': 7,
    'MAHALANOBIS_THRESHOLD': 90,
    'TRAIN_VAL_TEST_RATIO': (0.6, 0.2, 0.2),
    'TRAIN_VAL_TEST_RATIO_STR': '60:20:20',
    'K_FOLDS': 10,
    'PCA_N_COMPONENTS': 5,
    'DT_MAX_DEPTH_RANGE': [7, 10, 12, 15],
    'DT_MIN_SAMPLES_SPLIT_RANGE': [2, 5, 10],
    'DT_MIN_SAMPLES_LEAF_RANGE': [5, 10, 20],
    'RF_N_ESTIMATORS': [50, 100, 200],
    'PLS_N_COMPONENTS': [3, 5, 10, 15],
    'R2_THRESHOLD': 0.7,
    'VIF_THRESHOLD': 15
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

# 定義目標成分
components = ['Moisture', 'Protein', 'Fat', 'SFA', 'Ash', 'Collagen', 'Salt']

# 讓使用者選擇單一成分，使用數字
print("\n可用成分:")
for i, comp in enumerate(components, 1):
    print(f"{i}: {comp}")
selected_idx = int(input("請輸入成分編號 (例如: 1 for Moisture): "))
if not 1 <= selected_idx <= len(components):
    print(f"❌ 無效編號: {selected_idx}")
    exit()
selected_component = components[selected_idx - 1]
print(f"✅ 選擇成分: {selected_component}")

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

# 提取單一成分目標值
trad_df = trad_df.dropna(subset=[selected_component])
Y_single = trad_df[selected_component].values
sample_ids = trad_df[trad_sample_col].astype(str).tolist()
nir_transposed_df = nir_final_df.set_index('Wavelength').T
nir_transposed_df = nir_transposed_df.apply(pd.to_numeric, errors='coerce')
nir_transposed_df = nir_transposed_df.dropna(axis=1)
print("\n✅ 轉換後 NIR 資料形狀:", nir_transposed_df.shape)

nir_transposed_df.index = nir_transposed_df.index.astype(str)
common_ids = list(set(sample_ids).intersection(set(nir_transposed_df.index)))
if len(common_ids) != len(sample_ids) or len(common_ids) != len(Y_single):
    print(f"❌ 對齊後樣本數不一致：X: {len(common_ids)}, Y: {len(Y_single)}")
    trad_df = trad_df[trad_df[trad_sample_col].astype(str).isin(common_ids)]
    Y_single = trad_df[selected_component].values
    X = nir_transposed_df.loc[common_ids]
else:
    X = nir_transposed_df.loc[sample_ids]
print(f"✅ 成功對齊資料，樣本數: {X.shape[0]}")

# === 步驟 2: 異常值檢測 ===
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
Y_single = Y_single[mask_md]
print(f"✅ 馬氏距離過濾後樣本數：{X.shape[0]}")
if X.shape[0] == 0:
    print("❌ 馬氏距離過濾後數據為空，無法繼續處理")
    exit()

scaler_y_temp = StandardScaler()
Y_single_scaled_temp = scaler_y_temp.fit_transform(Y_single.reshape(-1, 1)).ravel()
mask_mcs = monte_carlo_outlier_detection(X, Y_single_scaled_temp)
if len(mask_mcs) == 0:
    print("❌ 蒙特卡洛檢測失敗，跳過異常值檢測")
    mask_mcs = np.ones(X.shape[0], dtype=bool)
X = X[mask_mcs]
Y_single = Y_single[mask_mcs]
print(f"✅ 結合馬氏距離和 MCS 後，剩餘樣本數：{X.shape[0]}")

if X.shape[0] == 0:
    print("❌ 異常值檢測後數據為空，無法繼續處理")
    exit()

# === 步驟 3: 數據集分配（SPXY） ===
def spxy(X, Y, test_size):
    X = np.array(X)
    Y = np.array(Y).reshape(-1, 1)
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
    Y_train = Y[selected].ravel()
    X_test = X[test]
    Y_test = Y[test].ravel()

    return X_train, X_test, Y_train, Y_test

scaler_x = StandardScaler()
X_scaled = scaler_x.fit_transform(X)
X = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
scaler_y = StandardScaler()
Y_single_scaled = scaler_y.fit_transform(Y_single.reshape(-1, 1)).ravel()

# 分割訓練+驗證 vs 測試
train_val_ratio, test_ratio = PARAMS['TRAIN_VAL_TEST_RATIO'][0] + PARAMS['TRAIN_VAL_TEST_RATIO'][1], PARAMS['TRAIN_VAL_TEST_RATIO'][2]
X_train_val, X_test, Y_train_val, Y_test = spxy(X, Y_single_scaled, test_size=test_ratio)

# 從訓練+驗證集中再分割訓練 vs 驗證
train_ratio_inner = PARAMS['TRAIN_VAL_TEST_RATIO'][0] / train_val_ratio
X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=1 - train_ratio_inner, random_state=42)

print(f"✅ SPXY 訓練集形狀: {X_train.shape}, 驗證集形狀: {X_val.shape}, 測試集形狀: {X_test.shape}")
print(f"✅ 訓練集 {selected_component} 描述:", pd.Series(scaler_y.inverse_transform(Y_train.reshape(-1, 1)).ravel()).describe())
print(f"✅ 驗證集 {selected_component} 描述:", pd.Series(scaler_y.inverse_transform(Y_val.reshape(-1, 1)).ravel()).describe())
print(f"✅ 測試集 {selected_component} 描述:", pd.Series(scaler_y.inverse_transform(Y_test.reshape(-1, 1)).ravel()).describe())

# === 步驟 4: 光譜預處理（僅 Savitzky-Golay） ===
X_train_sg = savgol_filter(X_train, window_length=51, polyorder=3, axis=1)
X_val_sg = savgol_filter(X_val, window_length=51, polyorder=3, axis=1)
X_test_sg = savgol_filter(X_test, window_length=51, polyorder=3, axis=1)
print("✅ Savitzky-Golay 濾波完成")

# === 步驟 5: 訓練基模型並生成 OOF Meta Features（新增 R² 篩選） ===
base_models = {
    'PLS': {
        'model': PLSRegression(),
        'param_grid': {'n_components': PARAMS['PLS_N_COMPONENTS']}
    },
    'RF': {
        'model': RandomForestRegressor(random_state=42),
        'param_grid': {
            'n_estimators': PARAMS['RF_N_ESTIMATORS'],
            'max_depth': PARAMS['DT_MAX_DEPTH_RANGE'],
            'min_samples_leaf': PARAMS['DT_MIN_SAMPLES_LEAF_RANGE']
        }
    },
    'SVM': {
        'model': SVR(),
        'param_grid': {
            'kernel': ['rbf', 'linear'],
            'C': [0.1, 1, 10],
            'epsilon': [0.01, 0.1, 0.5]
        }
    }
}

# 儲存基模型預測
kf = KFold(n_splits=PARAMS['K_FOLDS'], shuffle=True, random_state=42)
meta_features_train = np.zeros((X_train.shape[0], len(base_models)))
meta_features_val = np.zeros((X_val.shape[0], len(base_models)))
meta_features_test = np.zeros((X_test.shape[0], len(base_models)))
base_model_metrics = {name: {'train_r2': None, 'val_r2': None, 'test_r2': None} for name in base_models}
best_models = {}

# 訓練並生成 meta features
for idx, (name, config) in enumerate(base_models.items()):
    print(f"\n✅ 訓練基模型: {name}")
    grid_search = RandomizedSearchCV(config['model'], config['param_grid'], n_iter=20,
                                    cv=KFold(n_splits=5, shuffle=True, random_state=42),
                                    scoring='r2', n_jobs=-1, refit=True, random_state=42)
    grid_search.fit(X_train_sg, Y_train)
    best_models[name] = grid_search.best_estimator_
    print(f"✅ {name} 最佳參數: {grid_search.best_params_}")

    # OOF 預測 for train
    train_preds = np.zeros_like(Y_train)
    for train_idx, val_idx in kf.split(X_train_sg):
        X_tr, X_v = X_train_sg[train_idx], X_train_sg[val_idx]
        Y_tr = Y_train[train_idx]
        model = config['model'].set_params(**grid_search.best_params_)
        model.fit(X_tr, Y_tr)
        train_preds[val_idx] = model.predict(X_v)
    
    # Predict on val and test
    val_preds = grid_search.predict(X_val_sg)
    test_preds = grid_search.predict(X_test_sg)
    
    meta_features_train[:, idx] = train_preds
    meta_features_val[:, idx] = val_preds
    meta_features_test[:, idx] = test_preds
    
    # 計算基模型性能
    train_r2 = safe_r2_score(scaler_y.inverse_transform(Y_train.reshape(-1, 1)).ravel(), scaler_y.inverse_transform(train_preds.reshape(-1, 1)).ravel())
    val_r2 = safe_r2_score(scaler_y.inverse_transform(Y_val.reshape(-1, 1)).ravel(), scaler_y.inverse_transform(val_preds.reshape(-1, 1)).ravel())
    test_r2 = safe_r2_score(scaler_y.inverse_transform(Y_test.reshape(-1, 1)).ravel(), scaler_y.inverse_transform(test_preds.reshape(-1, 1)).ravel())
    base_model_metrics[name]['train_r2'] = train_r2
    base_model_metrics[name]['val_r2'] = val_r2
    base_model_metrics[name]['test_r2'] = test_r2
    print(f"✅ {name} - {selected_component} Train R²: {train_r2:.4f}, Val R²: {val_r2:.4f}, Test R²: {test_r2:.4f}")

# 篩選 R² > R2_THRESHOLD 的基模型（基於驗證集）
selected_base_models = [name for name, metrics in base_model_metrics.items() if metrics['val_r2'] > PARAMS['R2_THRESHOLD']]
use_meta_features = len(selected_base_models) > 0

if use_meta_features:
    print("\n✅ 篩選後的基模型（驗證集 R² > {}）：".format(PARAMS['R2_THRESHOLD']))
    for name in selected_base_models:
        metrics = base_model_metrics[name]
        print(f"  - {name}: Train R² = {metrics['train_r2']:.4f}, Val R² = {metrics['val_r2']:.4f}, Test R² = {metrics['test_r2']:.4f}")
else:
    print(f"⚠️ 所有基模型的驗證集 R² 均低於 {PARAMS['R2_THRESHOLD']}，將直接使用原始光譜數據訓練元學習器")
    meta_features_train = X_train_sg
    meta_features_val = X_val_sg
    meta_features_test = X_test_sg
    selected_features = ['Raw_Spectral_Data']
    vif_data = pd.DataFrame({'Feature': ['Raw_Spectral_Data'], 'VIF': [1.0]})
    model_feature_indices = {'Raw_Spectral_Data': [0]}

# 更新 meta features，只保留選中的模型（若使用 meta features）
if use_meta_features:
    selected_indices = [list(base_models.keys()).index(name) for name in selected_base_models]
    meta_features_train = meta_features_train[:, selected_indices]
    meta_features_val = meta_features_val[:, selected_indices]
    meta_features_test = meta_features_test[:, selected_indices]
    print(f"✅ 更新後 Meta Features 形狀: 訓練集 {meta_features_train.shape}, 驗證集 {meta_features_val.shape}, 測試集 {meta_features_test.shape}")

    # 更新 best_models，只保留選中的模型
    best_models = {name: model for name, model in best_models.items() if name in selected_base_models}

    # === 步驟 6: 檢查 Meta Features 共線性 ===
    meta_features_df = pd.DataFrame(meta_features_train, columns=selected_base_models)
    corr_matrix = meta_features_df.corr()
    avg_meta_corr = corr_matrix.abs().mean().mean() if meta_features_df.shape[1] > 1 else 1.0
    print(f"✅ Meta Features 平均相關性: {avg_meta_corr:.4f}")

    if meta_features_df.shape[1] > 1:
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
        plt.title('Correlation Heatmap of Meta Features')
        plt.tight_layout()
        plt.show()
    else:
        print("✅ 僅一個 Meta Feature，跳過相關性熱圖")

    # 計算 VIF（僅在有多於一個特徵時執行）
    if meta_features_df.shape[1] > 1:
        vif_data = pd.DataFrame()
        vif_data['Feature'] = meta_features_df.columns
        vif_data['VIF'] = [variance_inflation_factor(meta_features_df.values, i) for i in range(meta_features_df.shape[1])]
        print("✅ Meta Features VIF:")
        print(vif_data)
        
        # 移除高 VIF 特徵（VIF > VIF_THRESHOLD），但至少保留一個特徵
        selected_features = vif_data[vif_data['VIF'] < PARAMS['VIF_THRESHOLD']]['Feature'].tolist()
        if not selected_features:
            # 選擇驗證集 R² 最高的模型
            best_model = max(selected_base_models, key=lambda x: base_model_metrics[x]['val_r2'])
            selected_features = [best_model]
            print(f"⚠️ 所有特徵的 VIF 高於 {PARAMS['VIF_THRESHOLD']}，保留驗證集 R² 最高的模型: {best_model}")
        
        if len(selected_features) < meta_features_df.shape[1]:
            print(f"✅ 移除高 VIF 特徵，保留: {selected_features}")
            meta_features_train = meta_features_df[selected_features].values
            meta_features_val = pd.DataFrame(meta_features_val, columns=meta_features_df.columns)[selected_features].values
            meta_features_test = pd.DataFrame(meta_features_test, columns=meta_features_df.columns)[selected_features].values
        else:
            print("✅ 無高 VIF 特徵，保留所有 meta features")
            selected_features = meta_features_df.columns.tolist()
    else:
        print("✅ 僅一個 Meta Feature，跳過 VIF 計算")
        vif_data = pd.DataFrame({'Feature': [meta_features_df.columns[0]], 'VIF': [1.0]})
        selected_features = meta_features_df.columns.tolist()

    # 重新映射特徵索引到 VIF 篩選後的列
    meta_features_df_selected = pd.DataFrame(meta_features_train, columns=selected_features)
    model_feature_indices = {}
    for model in selected_base_models:
        model_cols = [col for col in selected_features if col.startswith(model)]
        model_feature_indices[model] = [meta_features_df_selected.columns.get_loc(col) for col in model_cols]
    print(f"✅ 每個基模型的保留特徵索引: {model_feature_indices}")

# === 步驟 7: 對 Meta Features 降維 ===
if use_meta_features and len(selected_features) > 1:
    pca_n_components = max(2, min(PARAMS['PCA_N_COMPONENTS'], len(selected_features)))
    print(f"✅ PCA 組分數: {pca_n_components}")
    pca_meta = PCA(n_components=pca_n_components)
    meta_features_train_pca = pca_meta.fit_transform(meta_features_train)
    meta_features_val_pca = pca_meta.transform(meta_features_val)
    meta_features_test_pca = pca_meta.transform(meta_features_test)
else:
    print("✅ 未使用 PCA 降維（無 Meta Features 或僅一個特徵）")
    meta_features_train_pca = meta_features_train
    meta_features_val_pca = meta_features_val
    meta_features_test_pca = meta_features_test
    pca_n_components = meta_features_train.shape[1]

print(f"✅ 降維後 Meta Features 訓練集形狀: {meta_features_train_pca.shape}")
print(f"✅ 降維後 Meta Features 驗證集形狀: {meta_features_val_pca.shape}")
print(f"✅ 降維後 Meta Features 測試集形狀: {meta_features_test_pca.shape}")

# === 步驟 8: 訓練元學習器（DecisionTree） ===
dt_max_depth = [2, 3] if use_meta_features and avg_meta_corr > 0.7 else [3, 5, 7]
print(f"✅ DT max_depth 範圍: {dt_max_depth}")
meta_param_grid = {
    'max_depth': dt_max_depth,
    'min_samples_leaf': [5, 10]
}
meta_dt = DecisionTreeRegressor(random_state=42)
meta_grid_search = RandomizedSearchCV(meta_dt, meta_param_grid, n_iter=20,
                                     cv=KFold(n_splits=5, shuffle=True, random_state=42),
                                     scoring='r2', n_jobs=-1, refit=True, random_state=42)
meta_grid_search.fit(meta_features_train_pca, Y_train)
best_meta_dt = meta_grid_search.best_estimator_
print(f"✅ 元學習器最佳參數: {meta_grid_search.best_params_}")

# 簡單平均集成作為基準（僅當使用 Meta Features 時計算）
if use_meta_features:
    avg_preds_train = np.mean(meta_features_train, axis=1) if meta_features_train.shape[1] > 1 else meta_features_train.ravel()
    avg_preds_val = np.mean(meta_features_val, axis=1) if meta_features_val.shape[1] > 1 else meta_features_val.ravel()
    avg_preds_test = np.mean(meta_features_test, axis=1) if meta_features_test.shape[1] > 1 else meta_features_test.ravel()
else:
    avg_preds_train = np.zeros_like(Y_train)  # 占位符，無平均集成
    avg_preds_val = np.zeros_like(Y_val)
    avg_preds_test = np.zeros_like(Y_test)

# === 步驟 9: 測試與評估 ===
Y_train_pred_single = best_meta_dt.predict(meta_features_train_pca)
Y_val_pred_single = best_meta_dt.predict(meta_features_val_pca)
Y_test_pred_single = best_meta_dt.predict(meta_features_test_pca)
Y_train_pred = scaler_y.inverse_transform(Y_train_pred_single.reshape(-1, 1)).ravel()
Y_val_pred = scaler_y.inverse_transform(Y_val_pred_single.reshape(-1, 1)).ravel()
Y_test_pred = scaler_y.inverse_transform(Y_test_pred_single.reshape(-1, 1)).ravel()
Y_train_original = scaler_y.inverse_transform(Y_train.reshape(-1, 1)).ravel()
Y_val_original = scaler_y.inverse_transform(Y_val.reshape(-1, 1)).ravel()
Y_test_original = scaler_y.inverse_transform(Y_test.reshape(-1, 1)).ravel()
if use_meta_features:
    avg_preds_train_original = scaler_y.inverse_transform(avg_preds_train.reshape(-1, 1)).ravel()
    avg_preds_val_original = scaler_y.inverse_transform(avg_preds_val.reshape(-1, 1)).ravel()
    avg_preds_test_original = scaler_y.inverse_transform(avg_preds_test.reshape(-1, 1)).ravel()
else:
    avg_preds_train_original = np.zeros_like(Y_train_original)  # 占位符
    avg_preds_val_original = np.zeros_like(Y_val_original)
    avg_preds_test_original = np.zeros_like(Y_test_original)

# 計算 SEC 和 SEP
n_train = X_train.shape[0]
n_val = X_val.shape[0]
n_test = X_test.shape[0]

# 評估單一成分（DecisionTree 和平均集成）
train_mse = mean_squared_error(Y_train_pred, Y_train_original)
val_mse = mean_squared_error(Y_val_pred, Y_val_original)
test_mse = mean_squared_error(Y_test_pred, Y_test_original)
train_rmse = np.sqrt(train_mse)
val_rmse = np.sqrt(val_mse)
test_rmse = np.sqrt(test_mse)
train_r2 = safe_r2_score(Y_train_original, Y_train_pred)
val_r2 = safe_r2_score(Y_val_original, Y_val_pred)
test_r2 = safe_r2_score(Y_test_original, Y_test_pred)
sec = np.sqrt(np.sum((Y_train_original - Y_train_pred) ** 2) / max(1, n_train - 1))
sev = np.sqrt(np.sum((Y_val_original - Y_val_pred) ** 2) / max(1, n_val - 1))
sep = np.sqrt(np.sum((Y_test_original - Y_test_pred) ** 2) / (n_test - 1))
print(f"✅ {selected_component} (DecisionTree) - Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
print(f"✅ {selected_component} (DecisionTree) - Train R²: {train_r2:.4f}, Val R²: {val_r2:.4f}, Test R²: {test_r2:.4f}")
print(f"✅ {selected_component} (DecisionTree) - SEC: {sec:.4f}, SEV: {sev:.4f}, SEP: {sep:.4f}")

if use_meta_features:
    avg_train_r2 = safe_r2_score(Y_train_original, avg_preds_train_original)
    avg_val_r2 = safe_r2_score(Y_val_original, avg_preds_val_original)
    avg_test_r2 = safe_r2_score(Y_test_original, avg_preds_test_original)
    print(f"✅ {selected_component} (Average Ensemble) - Train R²: {avg_train_r2:.4f}, Val R²: {avg_val_r2:.4f}, Test R²: {avg_test_r2:.4f}")
else:
    avg_train_r2 = None
    avg_val_r2 = None
    avg_test_r2 = None
    print("✅ 未使用 Meta Features，跳過平均集成評估")

# 5-fold 交叉驗證評估 on train
cv_scores = cross_val_score(DecisionTreeRegressor(**meta_grid_search.best_params_), 
                            meta_features_train_pca, Y_train, cv=5, scoring='r2')
print(f"\n✅ 5-fold Cross-Validation Mean R² (train): {np.mean(cv_scores):.4f} (± {np.std(cv_scores) * 2:.4f})")

# 交叉驗證 on val and test (use 2-fold for validation due to small size)
val_cv_scores = cross_val_score(DecisionTreeRegressor(**meta_grid_search.best_params_),
                                meta_features_val_pca, Y_val, cv=2, scoring='r2')
print(f"✅ 驗證集 2-fold Cross-Validation Mean R²: {np.mean(val_cv_scores):.4f} (± {np.std(val_cv_scores) * 2:.4f})")

test_cv_scores = cross_val_score(DecisionTreeRegressor(**meta_grid_search.best_params_),
                                 meta_features_test_pca, Y_test, cv=3, scoring='r2')
print(f"✅ 測試集 3-fold Cross-Validation Mean R²: {np.mean(test_cv_scores):.4f} (± {np.std(test_cv_scores) * 2:.4f})")

# 指標表格數據
metrics_data_rmse = {
    'Component': [selected_component] * 3,
    'Metric': ['RMSE'] * 3,
    'Set': ['Training', 'Validation', 'Test'],
    'Value': [f'{train_rmse:.4f}', f'{val_rmse:.4f}', f'{test_rmse:.4f}']
}
metrics_df_rmse = pd.DataFrame(metrics_data_rmse)
print("\n✅ RMSE Evaluation Metrics:")
print(metrics_df_rmse.to_string(index=False))

metrics_data_se = {
    'Component': [selected_component] * 3,
    'Metric': ['SE'] * 3,
    'Set': ['Training', 'Validation', 'Test'],
    'Value': [f'{sec:.4f}', f'{sev:.4f}', f'{sep:.4f}']
}
metrics_df_se = pd.DataFrame(metrics_data_se)
print("\n✅ SE Evaluation Metrics:")
print(metrics_df_se.to_string(index=False))

metrics_data_r2 = {
    'Component': [selected_component] * 3,
    'Metric': ['R² Score'] * 3,
    'Set': ['Training', 'Validation', 'Test'],
    'Value': [f'{train_r2:.4f}', f'{val_r2:.4f}', f'{test_r2:.4f}']
}
metrics_df_r2 = pd.DataFrame(metrics_data_r2)
print("\n✅ R² Score Evaluation Metrics:")
print(metrics_df_r2.to_string(index=False))

# 繪製表格
fig_rmse = plt.figure(figsize=(12, 4))
ax1 = fig_rmse.add_subplot(111)
ax1.axis('off')
table_rmse = ax1.table(cellText=metrics_df_rmse.values, colLabels=metrics_df_rmse.columns, cellLoc='center', loc='center')
table_rmse.auto_set_font_size(False)
table_rmse.set_fontsize(10)
table_rmse.scale(1.2, 1.2)
ax1.set_title('RMSE Evaluation Metrics')
plt.show()

fig_se = plt.figure(figsize=(12, 4))
ax2 = fig_se.add_subplot(111)
ax2.axis('off')
table_se = ax2.table(cellText=metrics_df_se.values, colLabels=metrics_df_se.columns, cellLoc='center', loc='center')
table_se.auto_set_font_size(False)
table_se.set_fontsize(10)
table_se.scale(1.2, 1.2)
ax2.set_title('SE Evaluation Metrics')
plt.show()

fig_r2 = plt.figure(figsize=(12, 4))
ax3 = fig_r2.add_subplot(111)
ax3.axis('off')
table_r2 = ax3.table(cellText=metrics_df_r2.values, colLabels=metrics_df_r2.columns, cellLoc='center', loc='center')
table_r2.auto_set_font_size(False)
table_r2.set_fontsize(10)
table_r2.scale(1.2, 1.2)
ax3.set_title('R² Score Evaluation Metrics')
plt.show()

# 繪製基模型與元學習器的比較散點圖 (for test)
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(Y_test_original, Y_test_pred, alpha=0.7, color='blue', label='Meta-Learner (DecisionTree)')
if use_meta_features and meta_features_test.shape[1] > 0:
    ax.scatter(Y_test_original, avg_preds_test_original, alpha=0.7, color='green', label='Average Ensemble')
    for idx, name in enumerate(selected_base_models):
        if idx < meta_features_test.shape[1]:  # 確保索引不超出範圍
            test_preds = scaler_y.inverse_transform(meta_features_test[:, idx].reshape(-1, 1)).ravel()
            ax.scatter(Y_test_original, test_preds, alpha=0.5, label=f'Base Model ({name})')
else:
    print("⚠️ 未使用 Meta Features 或無 Meta Features 可繪製基模型散點圖")
ax.plot([min(Y_test_original), max(Y_test_original)],
        [min(Y_test_original), max(Y_test_original)], 'r--', label='Ideal (y=x)')
ax.set_xlabel(f'Actual {selected_component} Content (%)')
ax.set_ylabel(f'Predicted {selected_component} Content (%)')
ax.set_title(f'Actual vs Predicted {selected_component} (Test Set): Meta-Learner vs Base Models vs Average Ensemble')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# === 步驟 10: 生成參數和指標的 Excel 檔案 ===
common_params = {
    'Wavelength Range (nm)': PARAMS['WAVELENGTH_RANGE_STR'],
    'Train-Val-Test Split Ratio': PARAMS['TRAIN_VAL_TEST_RATIO_STR'],
    'K-Folds for OOF': PARAMS['K_FOLDS'],
    'PCA Components for Meta Features': pca_n_components,
    'R2 Threshold for Base Models': PARAMS['R2_THRESHOLD'],
    'VIF Threshold for Meta Features': PARAMS['VIF_THRESHOLD'],
    'Used Meta Features': 'Yes' if use_meta_features else 'No'
}

specific_params = {}
for name, model in best_models.items():
    if name == 'RF':
        specific_params['RF Optimal N Estimators'] = model.n_estimators
        specific_params['RF Optimal Max Depth'] = model.max_depth
        specific_params['RF Optimal Min Samples Leaf'] = model.min_samples_leaf
    elif name == 'SVM':
        specific_params['SVM Optimal Kernel'] = model.kernel
        specific_params['SVM Optimal C'] = model.C
        specific_params['SVM Optimal Epsilon'] = model.epsilon
    elif name == 'PLS':
        specific_params['PLS Optimal N Components'] = model.n_components

specific_params['Meta DT Optimal Max Depth'] = meta_grid_search.best_params_['max_depth']
specific_params['Meta DT Optimal Min Samples Leaf'] = meta_grid_search.best_params_['min_samples_leaf']

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_filename = f'Single_Component_{selected_component}_Meta_Learning_DecisionTree_OOF_{timestamp}.xlsx'
with pd.ExcelWriter(output_filename) as writer:
    eval_metrics = {}
    eval_metrics[f'{selected_component}_RMSEC'] = f'{train_rmse:.4f}'
    eval_metrics[f'{selected_component}_RMSEV'] = f'{val_rmse:.4f}'
    eval_metrics[f'{selected_component}_RMSEP'] = f'{test_rmse:.4f}'
    eval_metrics[f'{selected_component}_R²_C'] = f'{train_r2:.4f}'
    eval_metrics[f'{selected_component}_R²_V'] = f'{val_r2:.4f}'
    eval_metrics[f'{selected_component}_R²_P'] = f'{test_r2:.4f}'
    eval_metrics[f'{selected_component}_SEC'] = f'{sec:.4f}'
    eval_metrics[f'{selected_component}_SEV'] = f'{sev:.4f}'
    eval_metrics[f'{selected_component}_SEP'] = f'{sep:.4f}'
    if use_meta_features:
        eval_metrics[f'{selected_component}_Avg_Train_R²'] = f'{avg_train_r2:.4f}'
        eval_metrics[f'{selected_component}_Avg_Val_R²'] = f'{avg_val_r2:.4f}'
        eval_metrics[f'{selected_component}_Avg_Test_R²'] = f'{avg_test_r2:.4f}'
    else:
        eval_metrics[f'{selected_component}_Avg_Train_R²'] = 'N/A'
        eval_metrics[f'{selected_component}_Avg_Val_R²'] = 'N/A'
        eval_metrics[f'{selected_component}_Avg_Test_R²'] = 'N/A'
    for name in base_models:
        eval_metrics[f'{selected_component}_{name}_Train_R²'] = f'{base_model_metrics[name]["train_r2"]:.4f}'
        eval_metrics[f'{selected_component}_{name}_Val_R²'] = f'{base_model_metrics[name]["val_r2"]:.4f}'
        eval_metrics[f'{selected_component}_{name}_Test_R²'] = f'{base_model_metrics[name]["test_r2"]:.4f}'
    
    eval_metrics['Selected_Base_Models'] = ', '.join(selected_base_models) if use_meta_features else 'None'

    output_data = {
        'Parameter/Metric': list(common_params.keys()) + list(specific_params.keys()) + list(eval_metrics.keys()),
        'Value': list(common_params.values()) + list(specific_params.values()) + list(eval_metrics.values()),
        'Category': ['Common Parameters'] * len(common_params) + ['Specific Parameters'] * len(specific_params) + ['Evaluation Metrics'] * len(eval_metrics)
    }

    output_df = pd.DataFrame(output_data)
    output_df.to_excel(writer, sheet_name=selected_component, index=False)

    # 保存 VIF
    vif_data.to_excel(writer, sheet_name='VIF_Check', index=False)

    # 保存基模型篩選結果
    base_model_selection = {
        'Base Model': list(base_model_metrics.keys()),
        'Train R²': [base_model_metrics[name]['train_r2'] for name in base_model_metrics],
        'Validation R²': [base_model_metrics[name]['val_r2'] for name in base_model_metrics],
        'Test R²': [base_model_metrics[name]['test_r2'] for name in base_model_metrics],
        'Selected': [name in selected_base_models for name in base_model_metrics]
    }
    base_model_selection_df = pd.DataFrame(base_model_selection)
    base_model_selection_df.to_excel(writer, sheet_name='Base_Model_Selection', index=False)

print(f"✅ 已生成參數、指標檔案: {output_filename}")