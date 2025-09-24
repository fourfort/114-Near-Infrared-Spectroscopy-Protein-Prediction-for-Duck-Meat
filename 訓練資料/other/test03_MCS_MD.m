% 主腳本：分別繪製馬氏距離和 MCS 殞差散佈圖
clear; clc;

% 定義兩個圖表的參數
files = {'outlier_detection_md.xlsx', 'outlier_detection_mcs.xlsx'};
titles = {'Mahalanobis Distance Residual Scatter Plot', 'MCS Sampling Residual Scatter Plot'};
thresholds = [4.77, 0.19];
offsets = [0.2, 0.02]; % 標籤偏移量（馬氏距離 0.2，MCS 0.02）

% 用戶可自訂初始窗口大小（寬度、高度，單位：像素）
window_size = []; % 例如 [1000, 600] 或 []（預設）

% 定義字體大小
title_font_size = 14; % 標題字體大小
label_font_size = 12; % 軸標籤和圖例字體大小
text_font_size = 12;  % 異常值標籤字體大小
axes_font_size = 12;  % 軸刻度字體大小

% 定義軸標籤
x_label = 'Sample Number';
y_labels = {'Mahalanobis Distance', 'Average Residual'};

% 讀取兩個 Excel 檔案
data_md = readtable(files{1});
data_mcs = readtable(files{2});

% 處理 Is_Outlier 格式
is_outlier_md = data_md.Is_Outlier;
if isstring(is_outlier_md) || iscellstr(is_outlier_md)
    is_outlier_md = strcmp(is_outlier_md, 'TRUE');
elseif ~islogical(is_outlier_md)
    error('Is_Outlier column in %s must be logical, string, or cell array of strings.', files{1});
end

is_outlier_mcs = data_mcs.Is_Outlier;
if isstring(is_outlier_mcs) || iscellstr(is_outlier_mcs)
    is_outlier_mcs = strcmp(is_outlier_mcs, 'TRUE');
elseif ~islogical(is_outlier_mcs)
    error('Is_Outlier column in %s must be logical, string, or cell array of strings.', files{2});
end

% 為異常值生成自訂標號（1 到 99）
label_md = zeros(size(data_md.Sample_Index));
label_mcs = zeros(size(data_mcs.Sample_Index));
current_label = 1;

% 優先分配共享標號
for i = 1:length(data_md.Sample_Index)
    if is_outlier_md(i) && is_outlier_mcs(i)
        label_md(i) = current_label;
        label_mcs(i) = current_label;
        current_label = current_label + 1;
    end
end

% 為剩餘的馬氏距離異常值分配標號
for i = 1:length(data_md.Sample_Index)
    if is_outlier_md(i) && label_md(i) == 0
        label_md(i) = current_label;
        current_label = current_label + 1;
    end
end

% 為剩餘的 MCS 異常值分配標號
for i = 1:length(data_mcs.Sample_Index)
    if is_outlier_mcs(i) && label_mcs(i) == 0
        label_mcs(i) = current_label;
        current_label = current_label + 1;
    end
end

% 分別繪製兩個圖表
figure(1);
if ~isempty(window_size)
    set(gcf, 'Position', [100, 100, window_size(1), window_size(2)]);
end
plot_residual_scatter(files{1}, titles{1}, thresholds(1), offsets(1), ...
    data_md.Sample_Index, data_md.Value, is_outlier_md, label_md, ...
    title_font_size, label_font_size, text_font_size, axes_font_size, ...
    x_label, y_labels{1});

figure(2);
if ~isempty(window_size)
    set(gcf, 'Position', [150, 150, window_size(1), window_size(2)]);
end
plot_residual_scatter(files{2}, titles{2}, thresholds(2), offsets(2), ...
    data_mcs.Sample_Index, data_mcs.Value, is_outlier_mcs, label_mcs, ...
    title_font_size, label_font_size, text_font_size, axes_font_size, ...
    x_label, y_labels{2});

% 通用繪圖函數
function plot_residual_scatter(filename, plot_title, threshold, offset, sample_index, value, is_outlier, label, ...
    title_font_size, label_font_size, text_font_size, axes_font_size, x_label, y_label)
    % 繪製散佈圖（空心圓圈）
    hold on;
    scatter(sample_index(~is_outlier), value(~is_outlier), 80, 'b', 'o', ...
        'DisplayName', 'Normal Points', 'LineWidth', 1.5);
    scatter(sample_index(is_outlier), value(is_outlier), 80, 'r', 'o', ...
        'DisplayName', 'Outliers', 'LineWidth', 1.5);
    
    % 為異常值添加標籤（帶隨機偏移，無背景框）
    text_handles = gobjects(sum(is_outlier), 1);
    idx = 1;
    for j = 1:length(sample_index)
        if is_outlier(j) && label(j) > 0
            x_offset = offset * (rand - 0.5);
            y_offset = offset * (rand - 0.5);
            text_handles(idx) = text(sample_index(j) + x_offset, value(j) + y_offset, ...
                num2str(label(j)), 'FontSize', text_font_size, 'Color', 'k', 'FontWeight', 'bold');
            idx = idx + 1;
        end
    end
    
    % 繪製閾值線
    x_limits = [min(sample_index) - 1, max(sample_index) + 1];
    plot(x_limits, [threshold threshold], 'k--', 'LineWidth', 1.5, ...
        'DisplayName', sprintf('Threshold (%.2f)', threshold));
    
    % 設置圖表屬性
    title(plot_title, 'FontSize', title_font_size, 'FontWeight', 'bold');
    xlabel(x_label, 'FontSize', label_font_size);
    ylabel(y_label, 'FontSize', label_font_size);
    legend('show', 'Location', 'best', 'FontSize', label_font_size);
    box off;
    set(gca, 'FontSize', axes_font_size);
    
    % 動態調整軸範圍
    axis tight;
    if strcmp(y_label, 'Average Residual')
        ylim([0, 0.5]); % MCS 圖使用較緊密的 Y 軸範圍
    else
        ylim([0, max(value) * 1.1]); % 馬氏距離圖適應數據範圍
    end
    xlim([min(sample_index) - 1, max(sample_index) + 1]);
    
    % 啟用交互式縮放和平移
    zoom on;
    pan on;
    
    % 啟用交互式數據標籤
    dcm = datacursormode(gcf);
    set(dcm, 'Enable', 'on', 'DisplayStyle', 'datatip', 'SnapToDataVertex', 'on');
    set(dcm, 'UpdateFcn', @(obj, event) customDatatip(obj, event, sample_index, value, label, is_outlier));
    
    hold off;
end

% 自定義數據提示框回調函數
function txt = customDatatip(~, event, sample_index, value, label, is_outlier)
    pos = get(event, 'Position');
    idx = find(sample_index == pos(1) & value == pos(2) & is_outlier, 1);
    if ~isempty(idx)
        txt = {['Sample Number: ', num2str(sample_index(idx))], ...
               ['Value: ', num2str(value(idx), '%.4f')], ...
               ['Label: ', num2str(label(idx))]};
    else
        txt = {['Sample Number: ', num2str(pos(1))], ...
               ['Value: ', num2str(pos(2), '%.4f')]};
    end
end