% 讀取 CSV 檔案
data = readmatrix('nir_data.csv');  

% 分離數據
wavelength = data(:,1);   % 第一欄是波長
spectra = data(:,2:end);  % 後面每一欄是不同樣品的光譜

% 讀取表頭名稱 (樣品名稱)
headers = readcell('nir_data.csv');  
sampleNames = headers(1,2:end);  % 跳過第一欄的 "Wavelength"

% 繪圖
figure;
plot(wavelength, spectra, 'LineWidth', 1.5);  % 一次畫多條曲線
xlabel('Wavelength (nm)', 'FontSize', 14);
ylabel('Absorbance (Log(1/R))', 'FontSize', 14);   % Y軸標籤
title('Multi-sample spectral curves', 'FontSize', 16);

% 設定坐標軸範圍與刻度
xlim([400 1100]);
ylim([1.5 5.5]);
xticks(400:50:1100);
yticks(1.5:0.1:5.5);

% 美化
set(gca, 'FontSize', 12, 'Box', 'on');  % 保留XY軸框線，但不顯示格線

% 添加圖例
legend(sampleNames, 'Location', 'bestoutside');  % 圖例放在外面避免擋住曲線
