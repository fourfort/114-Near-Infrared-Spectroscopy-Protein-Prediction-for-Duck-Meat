% 主腳本：繪製四個模型的散佈圖（DT, PLS, SVM, XGB）並顯示評估指標
clear; clc;

% 定義檔案和標題
files = {'DT_Scatter_Data.xlsx', 'PLS_Scatter_Data.xlsx', ...
         'SVM_Scatter_Data.xlsx', 'XGB_Scatter_Data.xlsx'};
titles = {'Decision Tree Scatter Plot', 'PLS Scatter Plot', ...
          'SVM Scatter Plot', 'XGBoost Scatter Plot'};

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

% 設置初始窗口大小（可選，單位：像素）
window_size = []; % 例如 [800, 800] 或 []（預設）

% 循環繪製四個模型的散佈圖
for i = 1:length(files)
    % 讀取 Excel 檔案
    data = readtable(files{i});
    
    % 分離校正集和驗證集
    is_calibration = strcmp(data.Dataset, 'Calibration');
    is_validation = strcmp(data.Dataset, 'Validation');
    actual = data.Actual;
    predicted = data.Predicted;
    
    % 獲取當前模型的評估指標
    model_name = erase(titles{i}, ' Scatter Plot'); % 提取模型名稱（DT, PLS, SVM, XGB）
    switch model_name
        case 'Decision Tree'
            metric = metrics.DT;
        case 'PLS'
            metric = metrics.PLS;
        case 'SVM'
            metric = metrics.SVM;
        case 'XGBoost'
            metric = metrics.XGB;
    end
    
    % 創建新圖表
    figure(i);
    if ~isempty(window_size)
        set(gcf, 'Position', [100 + 50*(i-1), 100 + 50*(i-1), window_size(1), window_size(2)]);
    end
    
    hold on;
    
    % 繪製校正集（紅色實心圓圈）
    scatter(actual(is_calibration), predicted(is_calibration), point_size, 'r', 'o', 'filled', ...
        'DisplayName', 'Calibration Set', 'MarkerFaceAlpha', marker_alpha);
    
    % 繪製驗證集（藍色實心圓圈）
    scatter(actual(is_validation), predicted(is_validation), point_size, 'b', 'o', 'filled', ...
        'DisplayName', 'Validation Set', 'MarkerFaceAlpha', marker_alpha);
    
    % 繪製理想相關線 (y=x)
    min_val = 20;
    max_val = 26;
    plot([min_val max_val], [min_val max_val], 'k--', 'LineWidth', line_width, ...
        'DisplayName', 'Ideal Line');
    
% 添加評估指標文字（只顯示 R²C, RMSEC, R²V, RMSEV）
text(25.5, 25.8, ...
    sprintf('R²C = %.4f, RMSEC = %.4f\nR²V = %.4f, RMSEV = %.4f', ...
    metric.R2C, metric.RMSEC, metric.R2V, metric.RMSEV), ...
    'FontSize', text_font_size, 'FontWeight', 'normal', 'HorizontalAlignment', 'right');

    
    % 設置圖表屬性
    title(titles{i}, 'FontSize', title_font_size, 'FontWeight', 'bold');
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
    
    % 設置均等比例
    axis equal;
    
    % 啟用交互式縮放和平移
    zoom on;
    pan on;
    
    hold off;
end