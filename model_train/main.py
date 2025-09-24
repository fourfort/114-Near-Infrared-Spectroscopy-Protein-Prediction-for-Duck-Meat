import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.spatial.distance import mahalanobis
from ptflops import get_model_complexity_info
import matplotlib.pyplot as plt
import matplotlib
import yaml
from model_build import build_model_from_yaml

# 設置圖表字體為 Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'

# =========================================================
# 1. 計算模型參數量和 FLOPs
# ---------------------------------------------------------
def get_model_parameters_and_flops(model, input_shape):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_m = total_params / 1e6
    flops, _ = get_model_complexity_info(model, input_shape, as_strings=False, print_per_layer_stat=False)
    flops_m = flops / 1e6
    return total_params, params_m, flops_m

# =========================================================
# 2. Monte Carlo + Mahalanobis 異常值剔除（可選）
# ---------------------------------------------------------
def remove_outliers_mc_md(X, Y, num_iter=10, p=0.8, threshold=8.0, min_samples=10):
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    retained_idx = np.arange(len(X), dtype=np.int64)
    initial_samples = len(X)
    for iteration in range(num_iter):
        if len(retained_idx) < min_samples:
            print(f"警告：剩餘樣本數過少 ({len(retained_idx)})，停止異常值剔除")
            break
        subset_size = max(min_samples, int(p * len(retained_idx)))
        subset_idx = np.random.choice(retained_idx, size=subset_size, replace=False).astype(np.int64)
        X_sub = X[subset_idx]
        cov = np.cov(X_sub, rowvar=False)
        cov += np.eye(cov.shape[0]) * 1e-6
        try:
            cov_inv = np.linalg.pinv(cov)
        except np.linalg.LinAlgError:
            print(f"警告：第 {iteration+1} 次迭代協方差矩陣不可逆，跳過")
            continue
        mean = X_sub.mean(axis=0)
        retained_idx = np.array([i for i in retained_idx if mahalanobis(X[i], mean, cov_inv) < threshold], 
                               dtype=np.int64)
        print(f"迭代 {iteration+1}：剩餘 {len(retained_idx)} 個樣本")
    print(f"異常值剔除後：{len(retained_idx)}/{initial_samples} 個樣本剩餘")
    if len(retained_idx) < min_samples:
        print("錯誤：異常值剔除後樣本數不足，考慮增加閾值或減少迭代次數")
        return X, Y
    return X[retained_idx], Y[retained_idx]

# =========================================================
# 3. 資料擴增
# ---------------------------------------------------------
def augment_data(X, Y, augment_times=2):
    X_aug = []
    Y_aug = []
    for _ in range(augment_times):
        for x, y in zip(X, Y):
            alpha = np.random.rand()
            x_other = X[np.random.randint(0, len(X))]
            x_mix = alpha * x + (1 - alpha) * x_other
            roll_factor = np.random.randint(-12, 13)
            x_roll = np.roll(x_mix, roll_factor)
            slope = np.random.uniform(-0.005, 0.005)  # 進一步減弱斜率
            line = np.linspace(0, slope, len(x))
            x_slope = 0.5 * x_roll + 0.5 * line
            noise_snr = np.random.randint(150, 170)  # 提高 SNR
            noise = np.random.normal(0, 1/(10**(noise_snr/10)), len(x))
            x_noised = x_slope + noise
            X_aug.append(x_noised.astype(np.float32))
            Y_aug.append(y)
    return np.vstack([X, X_aug]), np.hstack([Y, Y_aug])

# =========================================================
# 4. 讀取數據與前處理
# ---------------------------------------------------------
nir_df = pd.read_csv(r'D:\Users\wedke\document\extra document\大三上\113專題研究之製作\論文人生\紅外線\red_coding main\preview\nir_data.csv')
trad_df = pd.read_excel(r'D:\Users\wedke\document\extra document\大三上\113專題研究之製作\論文人生\紅外線\red_coding main\preview\trad_data.xlsx')
nir_sample_col = [c for c in nir_df.columns if 'sample' in c.lower()][0]
trad_sample_col = [c for c in trad_df.columns if 'sample' in c.lower()][0]
nir_df.columns = ['wl'] + [f'1-{i}' for i in range(1, len(nir_df.columns))]
nir_df = nir_df.drop(0).reset_index(drop=True)
nir_df['wl'] = nir_df['wl'].astype(float)
nir_filtered = nir_df[(nir_df['wl'] >= 850) & (nir_df['wl'] <= 1100)]
nir_X = nir_filtered.set_index('wl').T
nir_X = nir_X.apply(pd.to_numeric, errors='coerce').dropna(axis=1)
X_snv = ((nir_X - nir_X.mean(axis=1).values[:,None]) /
         nir_X.std(axis=1).values[:,None]).values.astype(np.float32)
trad_df = trad_df.dropna(subset=['Protein'])
Y = trad_df['Protein'].values.astype(np.float32)
trad_df[trad_sample_col] = [f'1-{i+1}' for i in range(len(trad_df))]
sample_ids = trad_df[trad_sample_col].astype(str).tolist()
nir_X.index = nir_X.index.astype(str)
X = X_snv[np.isin(nir_X.index, sample_ids)]
Y = Y[np.isin(sample_ids, nir_X.index)]
scaler_y = StandardScaler()
Y_scaled = scaler_y.fit_transform(Y.reshape(-1,1)).ravel()

# 檢查數據分佈
print(f"X_snv mean: {X_snv.mean():.4f}, std: {X_snv.std():.4f}")
print(f"Y mean: {Y.mean():.4f}, std: {Y.std():.4f}")
print(f"Y_scaled mean: {Y_scaled.mean():.4f}, std: {Y_scaled.std():.4f}")
plt.figure(figsize=(8, 5))
plt.hist(Y, bins=20)
plt.title('Protein Content Distribution')
plt.xlabel('Protein Content (%)')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()

# 禁用異常值剔除（可選：取消註釋以啟用）
# X, Y_scaled = remove_outliers_mc_md(X, Y_scaled, num_iter=10, p=0.8, threshold=8.0, min_samples=10)

# =========================================================
# 5. 自定義數據集
# ---------------------------------------------------------
class NIRDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

# =========================================================
# 6. K 折交叉驗證與訓練
# ---------------------------------------------------------
yaml_path = r'D:\Users\wedke\document\extra document\大三上\113專題研究之製作\論文人生\紅外線\red_coding main\model_train\model.yaml'
with open(yaml_path, 'r') as f:
    yaml_config = yaml.safe_load(f)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
kf = KFold(n_splits=5, shuffle=True, random_state=42)
EPOCHS = 1000
train_r2_scores, test_r2_scores = [], []

for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    print(f"\n=== 第 {fold+1} 折交叉驗證 ===")
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = Y_scaled[train_idx], Y_scaled[test_idx]

    # 分割後對訓練集進行擴增
    X_train, y_train = augment_data(X_train, y_train, augment_times=2)

    # 創建 DataLoader
    train_loader = DataLoader(NIRDataset(X_train, y_train), batch_size=8, shuffle=True, drop_last=True)
    test_loader = DataLoader(NIRDataset(X_test, y_test), batch_size=1, shuffle=False)

    # 構建模型
    model = build_model_from_yaml(yaml_config, seq_len=X.shape[1]).to(device)
    total_params, params_m, flops_m = get_model_parameters_and_flops(model, (X.shape[1],))
    if fold == 0:
        print(f"模型名稱：{yaml_config['model']['name']}")
        print(f"總參數量：{total_params:,} ({params_m:.2f} M)")
        print(f"FLOPs：{flops_m:.2f} MFLOPs")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-5, weight_decay=5e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    train_losses = []
    best_train_loss = float('inf')
    patience = 200
    counter = 0

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            # 檢查梯度範數
            grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
            train_losses.append(avg_loss)
        else:
            print(f'第 {epoch+1}/{EPOCHS} 次訓練 | 無批次處理（空數據載入器）')
            train_losses.append(float('inf'))
            continue
        scheduler.step(avg_loss)
        if (epoch+1) % 10 == 0:
            print(f'第 {epoch+1}/{EPOCHS} 次訓練 | 訓練損失 {avg_loss:.4f} | 學習率：{scheduler.get_last_lr()[0]:.6f} | 梯度範數：{grad_norm:.4f}')
        if avg_loss < best_train_loss:
            best_train_loss = avg_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"提前停止訓練於第 {epoch+1} 次")
                break

    # 繪製訓練損失曲線
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error Loss')
    plt.title(f'Training Loss Curve (Fold {fold+1})')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 評估
    model.eval()
    train_preds_scaled, train_trues_scaled = [], []
    with torch.no_grad():
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).cpu().numpy()
            if pred.ndim == 0:
                pred = np.array([pred])
            train_preds_scaled.append(pred)
            train_trues_scaled.append(yb.cpu().numpy())
    if train_preds_scaled:
        train_preds_scaled = np.concatenate([p.flatten() for p in train_preds_scaled])
        train_trues_scaled = np.concatenate([t.flatten() for t in train_trues_scaled])
        train_y_pred = scaler_y.inverse_transform(train_preds_scaled.reshape(-1,1)).ravel()
        train_y_true = scaler_y.inverse_transform(train_trues_scaled.reshape(-1,1)).ravel()
        train_mse = mean_squared_error(train_y_true, train_y_pred)
        train_rmse = np.sqrt(train_mse)
        train_r2 = r2_score(train_y_true, train_y_pred)
    else:
        print("錯誤：訓練集無預測結果（空數據集）")
        continue

    test_preds_scaled, test_trues_scaled = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).cpu().numpy()
            if pred.ndim == 0:
                pred = np.array([pred])
            test_preds_scaled.append(pred)
            test_trues_scaled.append(yb.cpu().numpy())
    if test_preds_scaled:
        test_preds_scaled = np.concatenate([p.flatten() for p in test_preds_scaled])
        test_trues_scaled = np.concatenate([t.flatten() for t in test_trues_scaled])
        test_y_pred = scaler_y.inverse_transform(test_preds_scaled.reshape(-1,1)).ravel()
        test_y_true = scaler_y.inverse_transform(test_trues_scaled.reshape(-1,1)).ravel()
        test_mse = mean_squared_error(test_y_true, test_y_pred)
        test_rmse = np.sqrt(test_mse)
        test_r2 = r2_score(test_y_true, test_y_pred)
    else:
        print("錯誤：測試集無預測結果（空數據集）")
        continue

    train_r2_scores.append(train_r2)
    test_r2_scores.append(test_r2)

    # 繪製指標表格
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis('off')
    table_data = [
        ['MSE', f'{train_mse:.4f}', f'{test_mse:.4f}'],
        ['RMSE', f'{train_rmse:.4f}', f'{test_rmse:.4f}'],
        ['R²', f'{train_r2:.4f}', f'{test_r2:.4f}']
    ]
    table = ax.table(cellText=table_data, colLabels=['Metric', 'Training Set', 'Test Set'], loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.title(f'Evaluation Metrics (Fold {fold+1})')
    plt.tight_layout()
    plt.show()

    # 終端機輸出指標表格
    metrics_df = pd.DataFrame(table_data, columns=['Metric', 'Training Set', 'Test Set'])
    print(f"\nFold {fold+1} Evaluation Metrics:")
    print(metrics_df.to_string(index=False))

    # 繪製散點圖
    print(f"Fold {fold+1} Test Set True Values Range: Min {test_y_true.min():.2f}, Max {test_y_true.max():.2f}")
    if test_y_true.min() >= 20:
        x_min = 20
    else:
        print(f"Warning: Some test set true values in Fold {fold+1} are below 20%")
        x_min = test_y_true.min()
    plt.figure(figsize=(6,6))
    plt.scatter(test_y_true, test_y_pred, alpha=0.7)
    plt.plot([x_min, test_y_true.max()], [x_min, test_y_true.max()], 'r--')
    plt.xlim(x_min, test_y_true.max())
    plt.ylim(x_min, test_y_true.max())
    plt.xlabel('Actual Protein Content (%)')
    plt.ylabel('Predicted Protein Content (%)')
    plt.title(f'SpectralCNNLSTM Predictions (Fold {fold+1})')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 輸出交叉驗證平均 R²
print(f"\n交叉驗證平均 R²：訓練集 {np.mean(train_r2_scores):.4f} ± {np.std(train_r2_scores):.4f}")
print(f"交叉驗證平均 R²：測試集 {np.mean(test_r2_scores):.4f} ± {np.std(test_r2_scores):.4f}")