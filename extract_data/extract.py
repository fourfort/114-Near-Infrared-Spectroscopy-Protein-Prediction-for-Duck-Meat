import pyreadr
import pandas as pd

# 輸入檔案（你自己的 .rda/.RData 檔案路徑）
in_file_path = r"D:\Users\wedke\document\extra document\大三上\113專題研究之製作\論文人生\紅外線\red_coding main\extract_data\meatspec.rda"

# 讀取 R 資料
datas = pyreadr.read_r(in_file_path)  # 回傳為 OrderedDict

# 查看裡面有哪些資料集（key）
print(datas.keys())  # 通常會顯示一個或多個變數名稱，例如：dict_keys(['mydata'])

# 提取你需要的 dataframe（例如 'mydata' 是 key）
df = datas["meatspec"]  # 替換為實際 key 名稱

# 顯示基本資訊
print(df.columns)
print(df.index)
print(df.head())

# 轉存為 CSV 檔
df.to_csv("轉出的檔案.csv", index=False)  # 若不需索引
