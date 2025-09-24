import pandas as pd
import matplotlib.pyplot as plt

# 設定中文字體（視需求可改為其他字體）
plt.rcParams['font.family'] = 'Microsoft JhengHei'

# 假設 Excel 檔案位於當前目錄
file_path = r'D:\Users\wedke\document\extra document\大三上\113專題研究之製作\論文人生\紅外線\紅外線過程資料\各ML技術之各成分數據呈現\last01.xlsx'
models = ['PLS', 'DT', 'RF', 'SVM']
metrics = ['RMSEC', 'RMSEP', 'R²C', 'R²P', 'SEC', 'SEP']

# 建立儲存平均值的字典
avg_data = {
    "Name": [],
    "RMSEC": [],
    "RMSEP": [],
    "R²C": [],
    "R²P": [],
    "SEC": [],
    "SEP": [],
}

# 模型名稱轉換
name_map = {"PLS": "PLS", "DT": "Decision Tree", "RF": "Random Forest", "SVM": "SVM"}

# 讀取各模型資料並計算平均
for model in models:
    df = pd.read_excel(file_path, sheet_name=model)
    df_metrics = df[metrics]
    mean_vals = df_metrics.mean()

    display_name = name_map.get(model, model)

    avg_data["Name"].append(display_name)
    avg_data["RMSEC"].append(round(mean_vals["RMSEC"], 4))
    avg_data["RMSEP"].append(round(mean_vals["RMSEP"], 4))
    avg_data["R²C"].append(round(mean_vals["R²C"], 4))
    avg_data["R²P"].append(round(mean_vals["R²P"], 4))
    avg_data["SEC"].append(round(mean_vals["SEC"], 4))
    avg_data["SEP"].append(round(mean_vals["SEP"], 4))

# 轉為 DataFrame
df_avg = pd.DataFrame(avg_data)

# 畫出論文格式表格
def plot_fps_table(dataframe):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')

    # 設定欄位名稱（將 R²C 和 R²P 顯示為 R²_C 和 R²_P）
    col_labels = ['Model', 'RMSEC', 'RMSEP', '$R^2_C$', '$R^2_P$', 'SEC', 'SEP']

    table = ax.table(
        cellText=dataframe.values,
        colLabels=col_labels,
        cellLoc='center',
        loc='center'
    )

    table.auto_set_font_size(False)

    for (row, col), cell in table.get_celld().items():
        if row == 0:  # 表頭
            cell.set_text_props(weight='bold', fontsize=20)
        elif col == 0:  # 模型名稱欄
            cell.set_text_props(weight='bold', fontsize=14)
        else:  # 數值欄
            cell.set_text_props(weight='bold', fontsize=20)

        # 將 R²C 和 R²P 數值欄（第 4 和第 5 欄）設為紅色
        if col in [3, 4] and row != 0:
            cell.get_text().set_color('red')

    table.scale(1.2, 1.4)
    plt.tight_layout()
    plt.show()

plot_fps_table(df_avg)
# plt.savefig("table_output.png", dpi=300)