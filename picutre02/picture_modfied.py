from PIL import Image

# === 1. 開啟影像 ===
img = Image.open(r"D:\Users\wedke\document\extra document\大三上\113專題研究之製作\論文人生\紅外線\red_coding main\picutre02\test02.jpg")

# === 2. 設定基本參數 ===
target_ratio = 2.0 / 1.5  # 高寬比 (1.33:1)
min_w, min_h = 450, 600   # 最小像素
dpi_setting = (300, 300)  # DPI 設為 300，可改 500

# === 3. 調整比例 (裁切為 1.33:1) ===
w, h = img.size
current_ratio = h / w

if current_ratio > target_ratio:
    # 太高 → 裁上下
    new_h = int(w * target_ratio)
    offset = (h - new_h) // 2
    img = img.crop((0, offset, w, offset + new_h))
elif current_ratio < target_ratio:
    # 太寬 → 裁左右
    new_w = int(h / target_ratio)
    offset = (w - new_w) // 2
    img = img.crop((offset, 0, offset + new_w, h))

# === 4. 調整大小 (至少 450×600 像素) ===
# 以 450×600 為基準放大或縮小
target_size = (450, 600)
img = img.resize(target_size, Image.LANCZOS)

# === 5. 若要灰階或彩色，可切換模式 ===
# 灰階：img = img.convert("L")
# 彩色：img = img.convert("RGB")
img = img.convert("RGB")

# === 6. 儲存成 JPEG，設定 DPI ===
img.save("output_resized.jpg", "JPEG", dpi=dpi_setting, quality=95)

print(f"✅ 已輸出影像尺寸: {img.width}×{img.height}, 高寬比約 {img.height/img.width:.2f}:1, DPI={dpi_setting[0]}")
