import pandas as pd
import matplotlib.pyplot as plt
import os

# 設置 Times New Roman 字體
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

def plot_combined_scatter(scatter_files, custom_labels, output_path=None, component='fat', 
                         sheet_name='Scatter_Data'):
    """
    將多個模型的散點圖數據繪製在同一張圖表上，自動檢測欄位名稱。

    參數:
        scatter_files (list): Excel 文件路徑列表
        custom_labels (list): 對應模型的顯示標籤
        output_path (str, optional): 輸出圖檔路徑
        component (str): 成分名稱，預設為 'Protein'
        sheet_name (str): Excel 工作表名稱，預設為 'Scatter_Data'
    """
    # 檢查 scatter_files 和 custom_labels 長度是否一致
    if len(scatter_files) != len(custom_labels):
        raise ValueError("❌ scatter_files 和 custom_labels 的長度必須一致")

    # 定義顏色和標記符號
    colors = ['blue', 'green', 'red', 'purple']
    markers = ['o', '^', 's', 'x']
    if len(scatter_files) > len(colors):
        print(f"⚠️ 模型數量 ({len(scatter_files)}) 超過顏色數量 ({len(colors)})，可能導致視覺混淆")

    # 初始化繪圖
    plt.figure(figsize=(8, 8))

    # 儲存所有數據點以計算理想線的範圍
    all_actual = []
    all_predicted = []

    # 讀取每個模型的散點圖數據並繪製
    for idx, (file_path, label) in enumerate(zip(scatter_files, custom_labels)):
        if not os.path.exists(file_path):
            print(f"❌ 檔案不存在: {file_path}")
            continue
        try:
            # 檢查工作表是否存在
            xl = pd.ExcelFile(file_path)
            if sheet_name not in xl.sheet_names:
                print(f"❌ {file_path} 缺少工作表 '{sheet_name}'，可用工作表: {xl.sheet_names}")
                continue

            # 讀取 Excel 文件，跳過第一行（標題行）
            df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=1)
            
            # 檢查欄位名稱
            available_cols = df.columns.tolist()
            print(f"✅ {file_path} 的欄位: {available_cols}")

            # 假設前兩列是實際值和預測值（根據你的範例）
            if len(available_cols) < 2:
                print(f"❌ {file_path} 欄位數量不足，至少需要兩列")
                continue

            actual_col, pred_col = available_cols[0], available_cols[1]
            
            # 檢查數據是否為數值型
            if not (pd.api.types.is_numeric_dtype(df[actual_col]) and pd.api.types.is_numeric_dtype(df[pred_col])):
                print(f"❌ {file_path} 的 '{actual_col}' 或 '{pred_col}' 包含非數值數據")
                continue

            # 繪製散點
            plt.scatter(df[actual_col], df[pred_col], 
                        alpha=0.7, color=colors[idx % len(colors)], 
                        marker=markers[idx % len(markers)], 
                        label=label)
            all_actual.extend(df[actual_col])
            all_predicted.extend(df[pred_col])
        except ValueError as e:
            print(f"❌ 讀取 {file_path} 的工作表 '{sheet_name}' 失敗: {e}")
        except Exception as e:
            print(f"❌ 讀取 {file_path} 時發生其他錯誤: {e}")

    # 添加理想線 (y = x)
    if all_actual and all_predicted:
        min_val = min(min(all_actual), min(all_predicted))
        max_val = max(max(all_actual), max(all_predicted))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal (y=x)')
    else:
        print("⚠️ 無有效數據可繪製理想線")

    # 設置圖表屬性
    plt.xlabel(f'Actual {component} Content (%)')
    plt.ylabel(f'Predicted {component} Content (%)')
    plt.title(f'Combined Regression: Actual vs Predicted {component} for All Models')
    try:
        plt.legend()
    except Exception as e:
        print(f"⚠️ 生成圖例時發生錯誤: {e}")

    plt.grid(True)
    plt.tight_layout()

    # 儲存圖檔
    if output_path is None:
        output_path = os.path.join(os.path.dirname(scatter_files[0]), f'combined_scatter_{component}.png')
    try:
        plt.savefig(output_path)
        print(f"✅ 合併散點圖已儲存至 {output_path}")
    except Exception as e:
        print(f"❌ 儲存圖檔 {output_path} 時發生錯誤: {e}")

    # 顯示圖表（可選）
    try:
        plt.show()
    except KeyboardInterrupt:
        print("⚠️ 圖表顯示被鍵盤中斷")
    except Exception as e:
        print(f"❌ 顯示圖表時發生錯誤: {e}")

if __name__ == "__main__":
    # 設置 Excel 文件路徑和對應標籤
    scatter_files = [
        r'D:\meta\Scatter\Protein\PLS_Scatter_Data.xlsx',
        r'D:\meta\Scatter\Protein\SVM_Scatter_Data.xlsx',
        # r'D:\meta\Scatter\Fat\SVM_Scatter_Data.xlsx',
        # r'D:\meta\RF_Scatter_Data.xlsx'
        r'D:\meta\Scatter\Protein\CNN(PCA+Noise)_Scatter_Data.xlsx',
    ]
    custom_labels = ['PLS', 'SVM', 'CNN(PCA+Noise)']

    # 設置輸出路徑
    # output_file = r'D:\meta\combined_scatter_Protein.png'

    # 設置工作表名稱
    sheet_name = 'Scatter_Data'

    # 執行繪圖
    try:
        plot_combined_scatter(
            scatter_files=scatter_files,
            custom_labels=custom_labels,
            # output_path=output_file,
            component='Protein',
            sheet_name=sheet_name
        )
    except ValueError as e:
        print(e)