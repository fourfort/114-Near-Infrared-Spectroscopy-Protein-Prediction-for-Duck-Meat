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
xlabel('Wavelength (nm)', 'FontSize', 16);   % 增大 X 軸標題字體
ylabel('Absorbance (Log(1/R))', 'FontSize', 16);   % 增大 Y 軸標題字體
title('Multi-sample spectral curves', 'FontSize', 16);

% 設定坐標軸範圍與刻度
xlim([400 1100]);
ylim([1.5 5.5]);
xticks(400:100:1100);  % X 軸每 100 nm 一個刻度
yticks(1.5:0.5:5.5);   % Y 軸每 0.5 一個刻度

% 美化
set(gca, 'FontSize', 12);
box off;  % 移除上、右邊的邊框

% 添加圖例
legend(sampleNames, 'Location', 'bestoutside', 'FontSize', 12);  % 圖例放在外面