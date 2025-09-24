import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

# 設置全局字體為 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# === 步驟 1: 數據載入與處理 ===
try:
    trad_df = pd.read_excel(r'D:\Users\wedke\document\extra document\大三上\113專題研究之製作\論文人生\紅外線\red_coding main\preview\trad_data.xlsx')
    print("\n✅ trad_data.xlsx 前五行:")
    print(trad_df.head())
except FileNotFoundError:
    print("❌ 無法找到 trad_data.xlsx，請確認路徑。")
    exit()

# 定義成分欄位
components = ['Moisture', 'Protein', 'Fat', 'SFA', 'Ash', 'Collagen', 'Salt']

# 自動偵測樣本編號欄位名稱
def find_sample_column(df, name='data'):
    for col in df.columns:
        if 'sample number' in col.lower():
            print(f"✅ 在 {name} 資料中找到樣本編號欄位: {col}")
            return col
    print(f"⚠️ {name} 資料中未找到樣本編號欄位")
    return None

trad_sample_col = find_sample_column(trad_df, '傳統分析')

# 檢查缺失值並移除 Sample Number 缺失的樣本
print("\n🔍 傳統分析缺失值檢查:\n", trad_df.isnull().sum())

#重新生成編號（Sample Number）
if trad_sample_col:
    trad_df[trad_sample_col] = [f'1-{i+1}' for i in range(len(trad_df))]

# 提取成分數據
df_components = trad_df[components]
print(f"\n✅ 成分數據形狀: {df_components.shape}")

# 檢查負值
print("\n🔍 負值檢查:")
for col in components:
    neg_count = (df_components[col] < 0).sum()
    if neg_count > 0:
        print(f"⚠️ {col} 包含 {neg_count} 個負值")

# 選項 1: 將負值替換為 0
df_components_zero = df_components.where(df_components >= 0, 0)
print("\n✅ 負值已替換為0，成分數據描述:")
print(df_components_zero.describe())

# 選項 2: 移除包含負值的樣本
df_components_no_neg = df_components[(df_components >= 0).all(axis=1)]
print(f"\n✅ 移除負值樣本後的數據形狀: {df_components_no_neg.shape}")
print("\n✅ 移除負值樣本後數據描述:")
print(df_components_no_neg.describe())

# 比較描述統計
print("\n✅ df_components_zero 描述統計:")
print(df_components_zero.describe())
print("\n✅ df_components_no_neg 描述統計:")
print(df_components_no_neg.describe())

# 計算平均值和標準差的相對變化(兩個處理方式之比較)
for col in components:
    zero_mean = df_components_zero[col].mean()
    no_neg_mean = df_components_no_neg[col].mean()
    zero_std = df_components_zero[col].std()
    no_neg_std = df_components_no_neg[col].std()
    mean_diff = (zero_mean - no_neg_mean) / no_neg_mean * 100 if no_neg_mean != 0 else float('inf')
    std_diff = (zero_std - no_neg_std) / no_neg_std * 100 if no_neg_std != 0 else float('inf')
    print(f"\n✅ {col} 平均值變化: {mean_diff:.2f}%")
    print(f"✅ {col} 標準差變化: {std_diff:.2f}%")

# === 步驟 2: 分組分析 ===
# 根據 Sample Number 前綴分組
trad_df['Group'] = trad_df[trad_sample_col].str.extract(r'(\D+\d*-?\w*)')[0]
groups = trad_df['Group'].unique()
print(f"\n✅ 檢測到的分組: {groups}")

# === 步驟 3: 繪製分組散點圖 ===
print("\n✅ 繪製分組散點圖以檢查群集效應")
sns.pairplot(trad_df, vars=components, hue='Group', diag_kind='kde')
plt.suptitle("Pairplot of Components by Group", fontsize=14, family='Times New Roman', y=1.02)
plt.show()

# === 步驟 4: 標準化與移除極端值（使用移除負值樣本的數據爲佳） ===
scaler = RobustScaler()
df_components_standardized = pd.DataFrame(scaler.fit_transform(df_components_no_neg), 
                                         columns=df_components_no_neg.columns, 
                                         index=df_components_no_neg.index)
print("\n✅ 標準化後數據描述:")
print(df_components_standardized.describe())

# 移除極端值（基於 IQR，2*IQR）
df_components_cleaned = df_components_standardized.copy()
for col in components:
    q1 = df_components_standardized[col].quantile(0.25)
    q3 = df_components_standardized[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 2 * iqr
    upper_bound = q3 + 2 * iqr
    outliers = (df_components_standardized[col] < lower_bound) | (df_components_standardized[col] > upper_bound)
    print(f"✅ {col} 移除 {outliers.sum()} 個極端值 (基於 2*IQR)")
    df_components_cleaned.loc[outliers, col] = np.nan

print("\n✅ 移除極端值後數據形狀: {df_components_cleaned.shape}")
print("\n✅ 移除極端值後缺失值檢查:\n", df_components_cleaned.isnull().sum())

# === 步驟 5: 相關性分析 ===
# 創建 Excel 寫入器
excel_writer = pd.ExcelWriter('correlation_and_pca_results.xlsx', engine='xlsxwriter')

# 分組相關性分析
for group in groups:
    group_mask = trad_df['Group'] == group
    group_data = df_components_cleaned.loc[group_mask]
    if len(group_data.dropna()) > 1:
        # 計算 Pearson 和 Spearman 相關性
        corr_pearson = group_data.corr(method='pearson', min_periods=1)
        corr_spearman = group_data.corr(method='spearman', min_periods=1)
        
        # 輸出到控制台
        print(f"\n✅ 分組 {group} 的皮爾森相關性:")
        print(corr_pearson)
        print(f"\n✅ 分組 {group} 的斯皮爾曼相關性:")
        print(corr_spearman)
        
        # 保存到 Excel
        corr_pearson.to_excel(excel_writer, sheet_name=f'Pearson_Group_{group}', float_format='%.2f')
        corr_spearman.to_excel(excel_writer, sheet_name=f'Spearman_Group_{group}', float_format='%.2f')

# 整體相關性分析
# Pearson 相關性
corr_matrix_pearson = df_components_cleaned.corr(method='pearson', min_periods=1)
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix_pearson, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f', 
            annot_kws={'family': 'Times New Roman', 'size': 12})
plt.title("Pearson Correlation Matrix of Components (Cleaned)", fontsize=14, family='Times New Roman')
plt.xlabel("Components", fontsize=12, family='Times New Roman')
plt.ylabel("Components", fontsize=12, family='Times New Roman')
plt.show()
corr_matrix_pearson.to_excel(excel_writer, sheet_name='Pearson_Overall', float_format='%.2f')

# Spearman 相關性
corr_matrix_spearman = df_components_cleaned.corr(method='spearman', min_periods=1)
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix_spearman, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f', 
            annot_kws={'family': 'Times New Roman', 'size': 12})
plt.title("Spearman Correlation Matrix of Components (Cleaned)", fontsize=14, family='Times New Roman')
plt.xlabel("Components", fontsize=12, family='Times New Roman')
plt.ylabel("Components", fontsize=12, family='Times New Roman')
plt.show()
corr_matrix_spearman.to_excel(excel_writer, sheet_name='Spearman_Overall', float_format='%.2f')

# Kendall’s Tau 相關性
corr_matrix_kendall = df_components_cleaned.corr(method='kendall', min_periods=1)
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix_kendall, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f', 
            annot_kws={'family': 'Times New Roman', 'size': 12})
plt.title("Kendall’s Tau Correlation Matrix of Components (Cleaned)", fontsize=14, family='Times New Roman')
plt.xlabel("Components", fontsize=12, family='Times New Roman')
plt.ylabel("Components", fontsize=12, family='Times New Roman')
plt.show()
corr_matrix_kendall.to_excel(excel_writer, sheet_name='Kendall_Overall', float_format='%.2f')

# === 步驟 6: PCA與Biplot分析 ===
df_components_pca = df_components_cleaned.dropna()
if df_components_pca.empty:
    print("❌ 移除缺失值後無有效數據，無法進行PCA分析")
    exit()
print(f"✅ PCA分析的數據形狀（移除缺失值後）: {df_components_pca.shape}")

X_scaled = scaler.fit_transform(df_components_pca)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, label='Samples')
for i, component in enumerate(components):
    plt.arrow(0, 0, pca.components_[0, i]*3, pca.components_[1, i]*3, color='r', alpha=0.5, head_width=0.1)
    plt.text(pca.components_[0, i]*3.5, pca.components_[1, i]*3.5, component, 
             color='r', fontsize=12, family='Times New Roman')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% variance)', 
           fontsize=12, family='Times New Roman')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% variance)', 
           fontsize=12, family='Times New Roman')
plt.title("PCA Biplot of Components (Cleaned)", fontsize=14, family='Times New Roman')
plt.grid(True)
plt.legend(prop={'family': 'Times New Roman', 'size': 12})
plt.show()

print("\n✅ PCA解釋方差比例:", pca.explained_variance_ratio_)
print(f"✅ PC1解釋方差: {pca.explained_variance_ratio_[0]*100:.2f}%")
print(f"✅ PC2解釋方差: {pca.explained_variance_ratio_[1]*100:.2f}%")

# 保存 PCA 結果到 Excel
pca_results = pd.DataFrame({
    'Principal Component': ['PC1', 'PC2'],
    'Explained Variance Ratio': [pca.explained_variance_ratio_[0], pca.explained_variance_ratio_[1]],
    'Explained Variance (%)': [pca.explained_variance_ratio_[0]*100, pca.explained_variance_ratio_[1]*100]
})
pca_results.to_excel(excel_writer, sheet_name='PCA_Results', index=False, float_format='%.4f')

# 關閉 Excel 寫入器，保存文件
excel_writer.close()
print("\n✅ 相關性和 PCA 結果已保存至 'correlation_and_pca_results.xlsx'")