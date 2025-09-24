import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 載入 NIR 數據
nir_data = pd.read_csv('1.csv', index_col=0).T
nir_data.index = nir_data.index.str.replace('sample no: ', '', regex=False).str.strip()
print(f"NIR data shape after loading: {nir_data.shape}")

# 清理非數值數據
nir_data = nir_data.apply(pd.to_numeric, errors='coerce')
nir_data = nir_data.dropna(axis=1, how='all')
print(f"NIR data shape after cleaning: {nir_data.shape}")

# 過濾波長範圍 (850-1100 nm)
wavelengths = nir_data.columns.astype(float)
nir_data = nir_data.loc[:, (wavelengths >= 850) & (wavelengths <= 1100)]
print(f"NIR data shape after wavelength filtering (850-1100 nm): {nir_data.shape}")

# 載入化學分析數據
chem_data = pd.read_excel('Samples_91880802_12-06-2025_12-02-58.xlsx', sheet_name='Samples')
chem_data['Sample Number'] = chem_data['Sample Number'].astype(str).str.strip()
chem_data = chem_data.set_index('Sample Number')
print(f"Chem data shape: {chem_data.shape}")

# 對齊數據
common_samples = nir_data.index.intersection(chem_data.index)
print(f"Common samples: {len(common_samples)}")
X = nir_data.loc[common_samples]
y = chem_data.loc[common_samples, 'Protein']
print(f"X shape: {X.shape}, y shape: {y.shape}")

# 異常值處理
y = y.dropna()
X = X.loc[y.index]
outliers = y[y > 30].index
if len(outliers) > 0:
    print(f"Removing {len(outliers)} outliers with Protein > 30%")
    X = X.drop(outliers)
    y = y.drop(outliers)
print(f"X shape after outlier removal: {X.shape}, y shape: {y.shape}")

# 光譜預處理
X_smooth = savgol_filter(X, window_length=7, polyorder=2, axis=1)
X_snv = (X_smooth - X_smooth.mean(axis=1)[:, np.newaxis]) / X_smooth.std(axis=1)[:, np.newaxis]

# 標準化目標變量
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()
print(f"X_snv shape: {X_snv.shape}, y_scaled shape: {y_scaled.shape}")

# 分割數據集
X_train, X_test, y_train, y_test = train_test_split(X_snv, y_scaled, test_size=0.2, random_state=42)
print(f"Training set: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

# PLS 模型訓練
pls = PLSRegression(n_components=10)
pls.fit(X_train, y_train)

# 預測與評估
y_pred_scaled = pls.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
mse = mean_squared_error(y_test_original, y_pred)
r2 = r2_score(y_test_original, y_pred)
print(f'MSE: {mse:.4f}, R²: {r2:.4f}')

# 繪圖
plt.scatter(y_test_original, y_pred, alpha=0.5)
plt.xlabel('Actual Protein Content (%)')
plt.ylabel('Predicted Protein Content (%)')
plt.title('PLS Prediction of Protein Content')
plt.plot([min(y_test_original), max(y_test_original)], [min(y_test_original), max(y_test_original)], 'r--')
plt.show()

# 交叉驗證
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_mse = []
for train_index, val_index in kf.split(X_snv):
    X_cv_train, X_cv_val = X_snv[train_index], X_snv[val_index]
    y_cv_train, y_cv_val = y_scaled[train_index], y_scaled[val_index]
    pls.fit(X_cv_train, y_cv_train)
    y_cv_pred_scaled = pls.predict(X_cv_val)
    y_cv_pred = scaler_y.inverse_transform(y_cv_pred_scaled.reshape(-1, 1)).ravel()
    y_cv_val_original = scaler_y.inverse_transform(y_cv_val.reshape(-1, 1)).ravel()
    cv_mse.append(mean_squared_error(y_cv_val_original, y_cv_pred))
print(f'Cross-Validation MSE: {np.mean(cv_mse):.4f} ± {np.std(cv_mse):.4f}')