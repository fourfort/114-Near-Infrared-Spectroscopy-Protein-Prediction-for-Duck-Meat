import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# === 步驟 1: 讀取資料 ===
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

# === 步驟 2: 檢查缺失值 ===
print("\n🔍 NIR 缺失值檢查:\n", nir_df.isnull().sum())
print("\n🔍 傳統分析缺失值檢查:\n", trad_df.isnull().sum())

# === 步驟 3: 自動偵測樣本編號欄位名稱 ===
def find_sample_column(df, name='nir'):
    for col in df.columns:
        if 'sample number' in col.lower() or 'sample no' in col.lower():
            print(f"✅ 在 {name} 資料中找到樣本編號欄位: {col}")
            return col
    print(f"⚠️ {name} 資料中未找到樣本編號欄位")
    return None

nir_sample_col = find_sample_column(nir_df, 'NIR')
trad_sample_col = find_sample_column(trad_df, '傳統分析')

# === 步驟 4: 處理光譜資料 ===
nir_df.columns = [''] + [f'1-{i}' for i in range(1, len(nir_df.columns))]
nir_df = nir_df.drop(0).reset_index(drop=True)
nir_df[''] = nir_df[''].astype(float)
nir_filtered_df = nir_df[(nir_df[''] >= 850) & (nir_df[''] <= 1100)]
selected_columns = [''] + [f'1-{i}' for i in range(1, 151)]
nir_final_df = nir_filtered_df[[col for col in selected_columns if col in nir_filtered_df.columns]]

# === 步驟 5: 資料轉置並進行 SNV 光譜預處理 ===
nir_transposed_df = nir_final_df.set_index('').T
# 嘗試將所有資料轉成 float，並移除有 NaN 的欄位
nir_transposed_df = nir_transposed_df.apply(pd.to_numeric, errors='coerce')  # 無法轉成 float 的設為 NaN
nir_transposed_df = nir_transposed_df.dropna(axis=1)  # 移除有 NaN 的樣本
print("\n✅ 轉換後 NIR 資料形狀:", nir_transposed_df.shape)

nir_snv = (nir_transposed_df - nir_transposed_df.mean(axis=1).values[:, np.newaxis]) / \
           nir_transposed_df.std(axis=1).values[:, np.newaxis]

# === 步驟 6: 對傳統樣本重新編號 ===
if trad_sample_col:
    trad_df[trad_sample_col] = [f'1-{i+1}' for i in range(len(trad_df))]
else:
    print("⚠️ 無法進行樣本對齊，缺少樣本編號欄")

# === 步驟 7: 處理目標變數與異常值 ===
if 'Protein' in trad_df.columns:
    trad_df = trad_df.dropna(subset=['Protein'])
    trad_df = trad_df[trad_df['Protein'] <= 30]  # 移除 Protein > 30 的異常值
    Y = trad_df['Protein'].values
    print("\n🎯 目標值 Y 前五筆:", Y[:5])
else:
    print("❌ 缺少 'Protein' 欄位，無法建立模型")

# === 步驟 8: 對齊樣本 ===
if trad_sample_col:
    sample_ids = trad_df[trad_sample_col].astype(str).tolist()
    nir_snv.index = nir_snv.index.astype(str)
    X = nir_snv.loc[sample_ids]

    if X.shape[0] != len(Y):
        print(f"❌ 對齊後樣本數不一致：X: {X.shape[0]}, Y: {len(Y)}")
    else:
        print(f"✅ 成功對齊資料，樣本數: {X.shape[0]}")

# === 步驟 9: 標準化目標變數並分割資料 ===
scaler_y = StandardScaler()
Y_scaled = scaler_y.fit_transform(Y.reshape(-1, 1)).ravel()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_scaled, test_size=0.2, random_state=42)

# === 步驟 10: 建立與訓練 PLS 模型 ===
pls = PLSRegression(n_components=5)
pls.fit(X_train, Y_train)

Y_pred_scaled = pls.predict(X_test)
Y_pred = scaler_y.inverse_transform(Y_pred_scaled.reshape(-1, 1)).ravel()
Y_test_original = scaler_y.inverse_transform(Y_test.reshape(-1, 1))

mse = mean_squared_error(Y_test_original, Y_pred)
r2 = r2_score(Y_test_original, Y_pred)

print("\n📈 模型測試結果：")
print(f"均方誤差 (MSE): {mse:.4f}")
print(f"決定係數 R²: {r2:.4f}")

# === 步驟 11: 繪圖分析 ===
plt.figure(figsize=(8, 6))
plt.scatter(Y_test_original, Y_pred, alpha=0.7, color='blue', label='Predicted vs Actual')
plt.plot([Y_test_original.min(), Y_test_original.max()], [Y_test_original.min(), Y_test_original.max()],
         'r--', label='Ideal (y = x)')
plt.xlabel('Actual Protein Content (%)')
plt.ylabel('Predicted Protein Content (%)')
plt.title('PLS Regression: Actual vs Predicted Protein')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
