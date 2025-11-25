import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
import os

# 設置全局字體為 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# === 步驟 1: 數據載入與處理 ===
try:
    nir_df = pd.read_csv(r'D:\Users\wedke\document\extra document\大三上\113專題研究之製作\論文人生\紅外線\red_coding main\preview\ML Method\nir_data.csv')
    print("\nNIR 數據前五行:")
    print(nir_df.head())
except FileNotFoundError:
    print("無法找到 nir_data.csv，請確認路徑。")
    exit()

try:
    trad_df = pd.read_excel(r'D:\Users\wedke\document\extra document\大三上\113專題研究之製作\論文人生\紅外線\red_coding main\preview\trad_data.xlsx')
    print("\n傳統分析數據前五行:")
    print(trad_df.head())
except FileNotFoundError:
    print("無法找到 trad_data.xlsx，請確認路徑。")
    exit()

# 定義目標成分
components = ['Moisture', 'Protein', 'Fat', 'Ash']  # 僅分析這四個
print(f"\n目標成分: {components}")

# 自動偵測樣本編號欄位
def find_sample_column(df, name='data'):
    for col in df.columns:
        if 'sample' in col.lower() and ('number' in col.lower() or 'no' in col.lower()):
            print(f"在 {name} 資料中找到樣本編號欄位: {col}")
            return col
    print(f"{name} 資料中未找到樣本編號欄位")
    return None

trad_sample_col = find_sample_column(trad_df, '傳統分析')

# === 步驟 2: NIR 資料處理 ===
nir_df.columns = ['Wavelength'] + [f'S{i}' for i in range(1, len(nir_df.columns))]
nir_df = nir_df.iloc[1:].reset_index(drop=True)
nir_df['Wavelength'] = nir_df['Wavelength'].astype(float)

# 波長範圍過濾
wavelength_range = (750, 1099.5)
nir_filtered = nir_df[(nir_df['Wavelength'] >= wavelength_range[0]) & 
                     (nir_df['Wavelength'] <= wavelength_range[1])]
X_nir = nir_filtered.set_index('Wavelength').T
wavelengths = nir_filtered['Wavelength'].values

print(f"\nNIR 資料形狀: {X_nir.shape} (樣本 × 波長)")
print(f"波長範圍: {wavelengths.min():.1f} ~ {wavelengths.max():.1f} nm, 共 {len(wavelengths)} 點")

# === 步驟 3: 傳統分析資料處理與對齊 ===
if trad_sample_col:
    trad_df[trad_sample_col] = [f'S{i+1}' for i in range(len(trad_df))]
else:
    print("未找到樣本欄位，使用預設 S1, S2, ...")
    trad_sample_col = 'Sample_ID'
    trad_df[trad_sample_col] = [f'S{i+1}' for i in range(len(trad_df))]

# 提取成分
df_components = trad_df[components].copy()
trad_df_full = trad_df[[trad_sample_col] + components].copy()
trad_df_full.set_index(trad_sample_col, inplace=True)

# 對齊 NIR 與傳統分析樣本
common_samples = X_nir.index.intersection(trad_df_full.index)
if len(common_samples) == 0:
    print("樣本對齊失敗！嘗試去除前綴...")
    X_nir_clean = X_nir.index.str.lstrip('S').str.lstrip('1-')
    trad_clean = trad_df_full.index.str.lstrip('S').str.lstrip('1-')
    mapping = dict(zip(X_nir_clean, X_nir.index))
    common_idx = set(X_nir_clean) & set(trad_clean)
    common_samples = [mapping[idx] for idx in common_idx]

print(f"\n對齊後樣本數: {len(common_samples)}")

X_nir_aligned = X_nir.loc[common_samples].astype(float)
Y_aligned = trad_df_full.loc[common_samples][components]

print(f"最終 X 形狀: {X_nir_aligned.shape}, Y 形狀: {Y_aligned.shape}")

# === 步驟 4: 負值與極端值處理 ===
# 負值 → 設為 0
Y_clean = Y_aligned.where(Y_aligned >= 0, 0)
print("\n負值已設為 0")

# 標準化（RobustScaler）
scaler_y = RobustScaler()
Y_scaled = pd.DataFrame(scaler_y.fit_transform(Y_clean), columns=components, index=Y_clean.index)

# 移除極端值（2*IQR）
Y_cleaned = Y_scaled.copy()
for col in components:
    q1 = Y_scaled[col].quantile(0.25)
    q3 = Y_scaled[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 2 * iqr
    upper = q3 + 2 * iqr
    outliers = (Y_scaled[col] < lower) | (Y_scaled[col] > upper)
    print(f"{col} 移除 {outliers.sum()} 個極端值")
    Y_cleaned.loc[outliers, col] = np.nan

Y_final = Y_cleaned.dropna()
X_final = X_nir_aligned.loc[Y_final.index]

print(f"\n清理後資料形狀: X={X_final.shape}, Y={Y_final.shape}")

# === 步驟 5: 波長 vs 成分 相關性分析 ===
correlation_results = {}
for comp in components:
    corr_per_wavelength = X_final.corrwith(Y_final[comp])
    correlation_results[comp] = corr_per_wavelength

corr_df = pd.DataFrame(correlation_results, index=wavelengths)

# 創建 Excel 寫入器
excel_writer = pd.ExcelWriter('NIR_Wavelength_Component_Correlation.xlsx', engine='xlsxwriter')

# 繪製相關性光譜圖
plt.figure(figsize=(14, 8))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
for i, comp in enumerate(components):
    plt.plot(wavelengths, corr_df[comp], label=f'{comp}', color=colors[i], linewidth=2)

plt.axhline(y=0.6, color='gray', linestyle='--', alpha=0.7, label='|r| = 0.6')
plt.axhline(y=-0.6, color='gray', linestyle='--', alpha=0.7)
plt.axhline(y=0, color='black', linewidth=0.8)
plt.title('NIR Wavelength vs. Component Correlation Spectrum', fontsize=16, fontweight='bold')
plt.xlabel('Wavelength (nm)', fontsize=14)
plt.ylabel('Pearson Correlation Coefficient (r)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlim(wavelength_range[0], wavelength_range[1])
plt.ylim(-1, 1)
plt.tight_layout()
plt.show()

# 儲存完整相關性矩陣
corr_df.round(4).to_excel(excel_writer, sheet_name='Full_Correlation', float_format='%.4f')

# === 步驟 6: 高相關波長區間分析 ===
print("\n" + "="*60)
print("高相關波長區間 (|r| > 0.6)")
print("="*60)
high_corr_intervals = {}
for comp in components:
    high = corr_df.index[np.abs(corr_df[comp]) > 0.6].tolist()
    if high:
        intervals = []
        start = high[0]
        for i in range(1, len(high)):
            if high[i] - high[i-1] > 2:  # 間隔 > 2nm 視為新區間
                intervals.append(f"{start:.1f}–{high[i-1]:.1f}")
                start = high[i]
        intervals.append(f"{start:.1f}–{high[-1]:.1f}")
        high_corr_intervals[comp] = intervals
        print(f"{comp:8}: {', '.join(intervals)}")
    else:
        print(f"{comp:8}: 無")
        high_corr_intervals[comp] = []

# 儲存高相關波長
high_corr_df = pd.DataFrame(high_corr_intervals)
high_corr_df.to_excel(excel_writer, sheet_name='High_Correlation_Intervals', index=False)

# === 步驟 7: PCA + Biplot（NIR 光譜）===
scaler_x = RobustScaler()
X_scaled = scaler_x.fit_transform(X_final)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Biplot
plt.figure(figsize=(12, 9))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, c='lightgray', label='Samples')

# 成分載荷向量
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
scale = 5
for i, comp in enumerate(components):
    plt.arrow(0, 0, loadings[i, 0]*scale, loadings[i, 1]*scale,
              color=colors[i], width=0.02, head_width=0.2, alpha=0.8)
    plt.text(loadings[i, 0]*scale*1.2, loadings[i, 1]*scale*1.2,
             comp, color=colors[i], fontsize=14, fontweight='bold')

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=13)
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=13)
plt.title('PCA Biplot: NIR Spectra with Component Loadings', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# PCA 結果
pca_summary = pd.DataFrame({
    'PC': ['PC1', 'PC2'],
    'Explained Variance Ratio': pca.explained_variance_ratio_,
    'Explained Variance (%)': pca.explained_variance_ratio_ * 100
})
pca_summary.to_excel(excel_writer, sheet_name='PCA_Summary', index=False)

# 儲存載荷矩陣
loadings_df = pd.DataFrame(pca.components_.T, index=wavelengths, columns=['PC1', 'PC2'])
loadings_df['Abs_PC1'] = np.abs(loadings_df['PC1'])
top_wavelengths = loadings_df.nlargest(10, 'Abs_PC1').index.tolist()
print(f"\nPC1 載荷最高的前10個波長: {top_wavelengths}")

loadings_df.to_excel(excel_writer, sheet_name='PCA_Loadings', float_format='%.6f')

excel_writer.close()
print("\n所有結果已儲存至 'NIR_Wavelength_Component_Correlation.xlsx'")