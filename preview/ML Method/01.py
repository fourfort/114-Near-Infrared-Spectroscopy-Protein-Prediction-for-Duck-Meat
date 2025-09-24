import pandas as pd
import matplotlib.pyplot as plt

# 假設你已有的平均數據 DataFrame（例如 df_avg）
df_avg = pd.DataFrame({
    "Name": ["PLS", "Decision Tree", "Random Forest", "SVM"],
    "RMSE": [1.234, 1.456, 1.123, 1.345],
    "MSE":  [1.52, 2.1, 1.43, 1.88],
    "R²":   [0.85, 0.82, 0.88, 0.87],
    "SEC":  [1.12, 1.43, 1.21, 1.33],
    "SEP":  [1.25, 1.54, 1.28, 1.40]
})

# 畫出論文風格表格
def plot_paper_style_table(df):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('off')

    table = ax.table(
        cellText=df.round(3).values,
        colLabels=["Model", "RMSE", "MSE", "$R^2$", "SEC", "SEP"],
        cellLoc='center',
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.4)

    # 加粗表頭
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')

    plt.title("Table X. Performance Metrics of Regression Models", fontsize=14, weight='bold', pad=20)
    plt.tight_layout()
    plt.show()

plot_paper_style_table(df_avg)
