import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.colors as mcolors

# 假設 wavelengths, mean_spectrum 已有
wavelengths = nir_final_df['Wavelength'].values
mean_spectrum = scaler_x.inverse_transform(X.mean().values.reshape(1, -1))[0]

# 假設這是每個波長的「選中機率」(0~1之間)，例如 feature importance
importance = np.zeros_like(wavelengths, dtype=float)
importance[selected_wavelengths] = np.random.rand(len(selected_wavelengths))  # 範例隨機

# 平滑處理，讓顏色有漸層而非突兀
importance_smooth = gaussian_filter1d(importance, sigma=8)

# 自訂顏色映射 (藍-綠-黃-紅)
cmap = mcolors.LinearSegmentedColormap.from_list(
    "blue_green_yellow_red", ["blue", "green", "yellow", "red"]
)

plt.figure(figsize=(12, 6))

# 畫熱力背景
plt.imshow(
    [importance_smooth], 
    extent=[wavelengths.min(), wavelengths.max(), mean_spectrum.min(), mean_spectrum.max()],
    aspect='auto',
    cmap=cmap,
    alpha=0.5
)

# 疊加光譜曲線
plt.plot(wavelengths, mean_spectrum, color='black', linewidth=1.5, label='Mean Spectrum')

plt.xlabel('Wavelength (nm)')
plt.ylabel('Absorbance (Log(1/R))')
plt.title('Heatmap-style Feature Wavelength Importance')
plt.legend()
plt.colorbar(label="Selection Probability")
plt.tight_layout()
plt.show()


# === 改良版 CARS：統計 Selection Probability ===
def cars_with_prob(X, y, n_features=30, n_iterations=100):
    n_samples, n_wavelengths = X.shape
    selected_indices = list(range(n_wavelengths))  
    weights = np.ones(n_wavelengths)

    # 記錄每個波長被選中的次數
    selection_counts = np.zeros(n_wavelengths, dtype=int)

    for _ in range(n_iterations):
        n_components = min(7, len(selected_indices), n_samples)
        if n_components < 1 or len(selected_indices) < n_features:
            break

        pls = PLSRegression(n_components=n_components)
        pls.fit(X.iloc[:, selected_indices].values, y)
        coef = np.abs(pls.coef_.ravel())

        weights[selected_indices] = coef / np.sum(coef)

        n_select = max(n_features, int(len(selected_indices) * 0.98))
        selected_indices = np.random.choice(
            selected_indices, 
            size=n_select, 
            replace=False, 
            p=weights[selected_indices] / np.sum(weights[selected_indices])
        )

        # 更新選中次數
        selection_counts[selected_indices] += 1

    # 計算選擇機率
    selection_probability = selection_counts / n_iterations

    # 最後選定的特徵波長
    final_indices = np.argsort(selection_probability)[-n_features:]

    return final_indices, selection_probability


# === 執行 CARS 並繪製熱力圖 ===
selected_wavelengths, selection_probability = cars_with_prob(X_train, Y_train, 
                                                            n_features=PARAMS['CARS_N_FEATURES'], 
                                                            n_iterations=PARAMS['CARS_N_ITERATIONS'])

wavelengths = nir_final_df['Wavelength'].values
mean_spectrum = scaler_x.inverse_transform(X.mean().values.reshape(1, -1))[0]

# 平滑處理選擇機率
prob_smooth = gaussian_filter1d(selection_probability, sigma=8)

# 自訂顏色映射
cmap = mcolors.LinearSegmentedColormap.from_list(
    "blue_green_yellow_red", ["blue", "green", "yellow", "red"]
)

plt.figure(figsize=(12, 6))

# 畫熱力背景
plt.imshow(
    [prob_smooth], 
    extent=[wavelengths.min(), wavelengths.max(), mean_spectrum.min(), mean_spectrum.max()],
    aspect='auto',
    cmap=cmap,
    alpha=0.5
)