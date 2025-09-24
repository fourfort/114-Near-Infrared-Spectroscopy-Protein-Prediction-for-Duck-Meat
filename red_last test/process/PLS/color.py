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
