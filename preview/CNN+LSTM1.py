# 🔁 已整合功能：
# 1. 訓練集專用的資料擴充（weighted sum + roll + slope + noise）
# 2. 訓練前異常值剔除（Monte Carlo Sampling + Mahalanobis Distance）

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt

# -----------------------------
# 前處理（保持原樣命名）
# -----------------------------
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

# -----------------------------
# ✨ Monte Carlo + Mahalanobis
# -----------------------------
def remove_outliers_mc_md(X, Y, num_iter=100, p=0.75, threshold=3.0):
    retained_idx = np.arange(len(X))
    for _ in range(num_iter):
        subset_idx = np.random.choice(retained_idx, size=int(p*len(retained_idx)), replace=False).astype(int)
        X_sub = X[subset_idx]
        cov = np.cov(X_sub, rowvar=False)
        cov_inv = np.linalg.pinv(cov)
        mean = X_sub.mean(axis=0)
        retained_idx = [i for i in retained_idx if mahalanobis(X[i], mean, cov_inv) < threshold]
    return X[retained_idx], Y[retained_idx]

X, Y_scaled = remove_outliers_mc_md(X, Y_scaled)

# -----------------------------
# Custom Dataset with Aug
# -----------------------------
class NIRDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = X
        self.y = y
        self.augment = augment

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        if self.augment:
            x = self.augment_fn(x)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

    def augment_fn(self, x):
        alpha = np.random.rand()
        roll_factor = np.random.randint(-12, 13)
        noise_snr = np.random.randint(80, 101)
        slope = np.random.uniform(-0.2, 0.2)

        x_other = self.X[np.random.randint(0, len(self.X))]
        x_mix = alpha * x + (1 - alpha) * x_other

        x_roll = np.roll(x_mix, roll_factor)
        line = np.linspace(0, slope, len(x))
        x_slope = 0.5 * x_roll + 0.5 * line

        noise = np.random.normal(0, 1/(10**(noise_snr/10)), len(x))
        x_noised = x_slope + noise

        return x_noised.astype(np.float32)

# -----------------------------
# 資料分割 + 資料載入
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, Y_scaled, test_size=0.2, random_state=42)

train_loader = DataLoader(NIRDataset(X_train, y_train, augment=True), batch_size=32, shuffle=True)
test_loader  = DataLoader(NIRDataset(X_test , y_test , augment=False), batch_size=32, shuffle=False)

# -----------------------------
# CNN+LSTM Model
# -----------------------------
class SpectralCNNLSTM(nn.Module):
    def __init__(self, seq_len, lstm_hidden_size=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8)
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(lstm_hidden_size, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x.squeeze()

# -----------------------------
# 訓練
# -----------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SpectralCNNLSTM(seq_len=X.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 500
train_losses = []

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    if (epoch+1) % 50 == 0:
        print(f'Epoch {epoch+1}/{EPOCHS} | Train Loss {avg_loss:.4f}')

plt.plot(train_losses)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss')
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# 測試 & 評估
# -----------------------------
model.eval()
preds_scaled, trues_scaled = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        preds_scaled.append(model(xb).cpu().numpy())
        trues_scaled.append(yb.numpy())
preds_scaled = np.concatenate(preds_scaled)
trues_scaled = np.concatenate(trues_scaled)

# 還原標準化
y_pred = scaler_y.inverse_transform(preds_scaled.reshape(-1,1)).ravel()
y_true = scaler_y.inverse_transform(trues_scaled.reshape(-1,1)).ravel()

print('MSE:', mean_squared_error(y_true, y_pred))
print('RMSE:', np.sqrt(mean_squared_error(y_true, y_pred)))
print('R2:', r2_score(y_true, y_pred))

plt.figure(figsize=(6,6))
plt.scatter(y_true, y_pred, alpha=0.7)
plt.plot([20, y_true.max()], [20, y_true.max()], 'r--')
plt.xlim(20, y_true.max())
plt.ylim(20, y_true.max())
plt.xlabel('Actual Protein')
plt.ylabel('Predicted Protein')
plt.title('CNN-LSTM Prediction')
plt.grid(True)
plt.tight_layout()
plt.show()
