% 主腳本：繪製校正集和驗證集的散佈圖（DT, PLS, SVM, XGB）
clear; clc;

% 定義檔案和模型名稱
files = {'DT_Scatter_Data.xlsx', 'PLS_Scatter_Data.xlsx', ...
         'SVM_Scatter_Data.xlsx', 'XGB_Scatter_Data.xlsx'};
model_names = {'DT', 'PLS', 'SVM', 'XGB'};
colors = {[1 0 0], [0 0 1], [0 0.5 0], [0.5 0 0.5]}; % 紅、藍、綠、紫

% 定義評估指標
metrics = struct(...
    'DT', struct('RMSEC', 0.0193, 'RMSEV', 0.4548, 'R2C', 0.9995, 'R2V', 0.5943, 'SEC', 0.0194, 'SEV', 0.4646), ...
    'PLS', struct('RMSEC', 0.1711, 'RMSEV', 0.1855, 'R2C', 0.9580, 'R2V', 0.9325, 'SEC', 0.1720, 'SEV', 0.1895), ...
    'SVM', struct('RMSEC', 0.0225, 'RMSEV', 0.0557, 'R2C', 0.9993, 'R2V', 0.9939, 'SEC', 0.0226, 'SEV', 0.0569), ...
    'XGB', struct('RMSEC', 0.0160, 'RMSEV', 0.1278, 'R2C', 0.9996, 'R2V', 0.9680, 'SEC', 0.0161, 'SEV', 0.1305));

% 定義圖表參數
x_label = 'Actual Protein Content (%)';
y_label = 'Predicted Protein Content (%)';
title_font_size = 16;
label_font_size = 16;
axes_font_size = 12;
legend_font_size = 12;
text_font_size = 12;
point_size = 80;
line_width = 1.5;
marker_alpha = 0.7;

% 設置初始窗口大小（單位：像素）
window_size = [800, 800]; % 正方形窗口

% 讀取所有數據
data_all = cell(length(files), 1);
for i = 1:length(files)
    data_all{i} = readtable(files{i});
end

% 繪製校正集圖表
figure(1);
set(gcf, 'Position', [100, 100, window_size(1), window_size(2)]);
hold on;

% 繪製校正集散佈點
for i = 1:length(files)
    data = data_all{i};
    is_calibration = strcmp(data.Dataset, 'Calibration');
    actual = data.Actual(is_calibration);
    predicted = data.Predicted(is_calibration);
    scatter(actual, predicted, point_size, colors{i}, 'o', 'filled', ...
        'DisplayName', model_names{i}, 'MarkerFaceAlpha', marker_alpha);
end

% 繪製理想相關線
min_val = 20;
max_val = 26;
plot([min_val max_val], [min_val max_val], 'k--', 'LineWidth', line_width, ...
    'DisplayName', 'Ideal Line');

% 添加校正集評估指標
text_str = sprintf('Model | RMSEC | R²C | SEC\n');
for i = 1:length(model_names)
    text_str = [text_str, sprintf('%s | %.2f | %.2f | %.2f\n', ...
        model_names{i}, metrics.(model_names{i}).RMSEC, ...
        metrics.(model_names{i}).R2C, metrics.(model_names{i}).SEC)];
end
text(25.5, 25.8, text_str, 'FontSize', text_font_size, 'FontWeight', 'normal', ...
    'HorizontalAlignment', 'right');

% 設置圖表屬性
title('Calibration Set Scatter Plot', 'FontSize', title_font_size, 'FontWeight', 'bold');
xlabel(x_label, 'FontSize', label_font_size);
ylabel(y_label, 'FontSize', label_font_size);
legend('show', 'Location', 'best', 'FontSize', legend_font_size);
box off;
set(gca, 'FontSize', axes_font_size);

% 設置軸範圍和刻度
xlim([20 26]);
ylim([20 26]);
xticks(20:1:26);
yticks(20:1:26);
axis equal;

% 啟用交互式縮放和平移
zoom on;
pan on;

hold off;

% 繪製驗證集圖表
figure(2);
set(gcf, 'Position', [150, 150, window_size(1), window_size(2)]);
hold on;

% 繪製驗證集散佈點
for i = 1:length(files)
    data = data_all{i};
    is_validation = strcmp(data.Dataset, 'Validation');
    actual = data.Actual(is_validation);
    predicted = data.Predicted(is_validation);
    scatter(actual, predicted, point_size, colors{i}, 'o', 'filled', ...
        'DisplayName', model_names{i}, 'MarkerFaceAlpha', marker_alpha);
end

% 繪製理想相關線
plot([min_val max_val], [min_val max_val], 'k--', 'LineWidth', line_width, ...
    'DisplayName', 'Ideal Line');

% 添加驗證集評估指標
text_str = sprintf('Model | RMSEV | R²V | SEV\n');
for i = 1:length(model_names)
    text_str = [text_str, sprintf('%s | %.2f | %.2f | %.2f\n', ...
        model_names{i}, metrics.(model_names{i}).RMSEV, ...
        metrics.(model_names{i}).R2V, metrics.(model_names{i}).SEV)];
end
text(25.5, 25.8, text_str, 'FontSize', text_font_size, 'FontWeight', 'normal', ...
    'HorizontalAlignment', 'right');

% 設置圖表屬性
title('Validation Set Scatter Plot', 'FontSize', title_font_size, 'FontWeight', 'bold');
xlabel(x_label, 'FontSize', label_font_size);
ylabel(y_label, 'FontSize', label_font_size);
legend('show', 'Location', 'best', 'FontSize', legend_font_size);
box off;
set(gca, 'FontSize', axes_font_size);

% 設置軸範圍和刻度
xlim([20 26]);
ylim([20 26]);
xticks(20:1:26);
yticks(20:1:26);
axis equal;

% 啟用交互式縮放和平移
zoom on;
pan on;

hold off;