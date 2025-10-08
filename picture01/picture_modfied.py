from PIL import Image

# 可選擇的解析度 (寬, 高)
sizes = [
    (300, 360),
    (500, 600),
    (800, 960),
    (1100, 1320),
    (1300, 1560)
]

print("請選擇要輸出的圖片尺寸（高寬比為 1.2:1）:")
for i, (w, h) in enumerate(sizes, start=1):
    print(f"{i}. {w} x {h}")

choice = int(input("\n輸入選擇編號 (1-5): "))
if choice < 1 or choice > len(sizes):
    print("❌ 無效的選擇。請重新執行。")
    exit()

target_size = sizes[choice - 1]

# 輸入圖片路徑
img_path = input("\n請輸入圖片檔案名稱或路徑 (例如 photo.jpg): ")

# 開啟圖片
img = Image.open(img_path)
print(f"原始尺寸：{img.width}x{img.height}")

# 進行比例裁剪（保持中心構圖）
target_ratio = target_size[1] / target_size[0]
current_ratio = img.height / img.width

if current_ratio > target_ratio:
    # 太高 -> 裁掉上下
    new_h = int(img.width * target_ratio)
    offset = (img.height - new_h) // 2
    img = img.crop((0, offset, img.width, offset + new_h))
elif current_ratio < target_ratio:
    # 太寬 -> 裁掉左右
    new_w = int(img.height / target_ratio)
    offset = (img.width - new_w) // 2
    img = img.crop((offset, 0, offset + new_w, img.height))

# 調整為選定尺寸
img = img.resize(target_size)

# 輸出結果
output_name = f"output_{target_size[0]}x{target_size[1]}.jpg"
img.save(output_name)
print(f"✅ 已輸出：{output_name}")
print(f"最終尺寸：{img.width}x{img.height}（高寬比約 {target_ratio:.2f}:1）")
