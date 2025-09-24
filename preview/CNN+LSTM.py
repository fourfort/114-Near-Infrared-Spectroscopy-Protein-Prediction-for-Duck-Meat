import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# =========================================================
# 1. 讀檔＋前處理
# ---------------------------------------------------------
nir_df = pd.read_csv('nir_data.csv')
trad_df = pd.read_excel('trad_data.xlsx')

# (1) 找樣本欄位
nir_sample_col = [c for c in nir_df.columns if 'sample' in c.lower()][0]
trad_sample_col = [c for c in trad_df.columns if 'sample' in c.lower()][0]

# (2) 指定欄名 & 波長篩選
nir_df.columns = ['wl'] + [f'1-{i}' for i in range(1, len(nir_df.columns))]
nir_df = nir_df.drop(0).reset_index(drop=True)
nir_df['wl'] = nir_df['wl'].astype(float)
nir_filtered = nir_df[(nir_df['wl'] >= 850) & (nir_df['wl'] <= 1100)]

# (3) 轉置成 (samples, features)
nir_X = nir_filtered.set_index('wl').T
nir_X = nir_X.apply(pd.to_numeric, errors='coerce').dropna(axis=1)

# (4) SNV
X_snv = ((nir_X - nir_X.mean(axis=1).values[:,None]) /
         nir_X.std(axis=1).values[:,None]).values.astype(np.float32)

# (5) 傳統 Protein
trad_df = trad_df.dropna(subset=['Protein'])
Y = trad_df['Protein'].values.astype(np.float32)

# (6) 對齊樣本
trad_df[trad_sample_col] = [f'1-{i+1}' for i in range(len(trad_df))]
sample_ids = trad_df[trad_sample_col].astype(str).tolist()
nir_X.index = nir_X.index.astype(str)
X = X_snv[np.isin(nir_X.index, sample_ids)]
Y = Y[np.isin(sample_ids, nir_X.index)]

# (7) 標準化 y
scaler_y = StandardScaler()
Y_scaled = scaler_y.fit_transform(Y.reshape(-1,1)).ravel()

# =========================================================
# 2. Dataset / DataLoader
# ---------------------------------------------------------
class NIRDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self): return len(self.X)

    def __getitem__(self, idx): return self.X[idx], self.y[idx]

X_train, X_test, y_train, y_test = train_test_split(
    X, Y_scaled, test_size=0.2, random_state=42)

train_loader = DataLoader(NIRDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader  = DataLoader(NIRDataset(X_test , y_test ), batch_size=32, shuffle=False)

# =========================================================
# 3. 1-D CNN + LSTM Model
# ---------------------------------------------------------
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
            nn.Dropout(0.3),  # 新增 Dropout 防止過擬合
            nn.Linear(lstm_hidden_size, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, seq_len)
        x = self.cnn(x)     # (B, 64, 8)
        x = x.transpose(1, 2)  # (B, 8, 64) for LSTM
        x, (hn, cn) = self.lstm(x)  # (B, 8, lstm_hidden_size)
        x = x[:, -1, :]     # Take last time step (B, lstm_hidden_size)
        x = self.fc(x)      # (B, 1)
        return x.squeeze()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SpectralCNNLSTM(seq_len=X.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# =========================================================
# 4. Training Loop with Loss Tracking
# ---------------------------------------------------------
EPOCHS = 500
train_losses = []

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    num_batches = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        num_batches += 1
    avg_loss = epoch_loss / num_batches
    train_losses.append(avg_loss)
    if (epoch+1) % 50 == 0:
        print(f'Epoch {epoch+1}/{EPOCHS} | Train Loss {avg_loss:.4f}')

# 繪製訓練損失曲線
plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS+1), train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error Loss')
plt.title('Training Loss Over Epochs')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ------------------- 評估 -------------------
model.eval()
preds_scaled, trues_scaled = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        preds_scaled.append(model(xb).cpu().numpy())
        trues_scaled.append(yb.numpy())
preds_scaled = np.concatenate(preds_scaled)
trues_scaled = np.concatenate(trues_scaled)

# 還原
y_pred = scaler_y.inverse_transform(preds_scaled.reshape(-1,1)).ravel()
y_true = scaler_y.inverse_transform(trues_scaled.reshape(-1,1)).ravel()

# 計算指標
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print('MSE:', mean_squared_error(y_true, y_pred))
print('RMSE:', rmse)
print('R² :', r2_score(y_true, y_pred))

# ------------------- 圖 -------------------
plt.figure(figsize=(6,6))
plt.scatter(y_true, y_pred, alpha=.7)
plt.plot([20, y_true.max()], [20, y_true.max()], 'r--')
plt.xlim(20, y_true.max())
plt.ylim(20, y_true.max())
plt.xlabel('Actual Protein (%)')
plt.ylabel('Predicted Protein (%)')
plt.title('CNN-LSTM Prediction')
plt.grid(True)
plt.tight_layout()
plt.show()