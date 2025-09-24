import pandas as pd
import numpy as np
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
except ImportError as e:
    print(f"Error importing PyTorch: {e}")
    print("Please install PyTorch using: pip install torch torchvision torchaudio")
    print("For GPU support, use: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    exit(1)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
try:
    from scipy.signal import savgol_filter
except ImportError as e:
    print(f"Error importing savgol_filter: {e}")
    print("Please ensure SciPy is installed. Run 'pip install scipy'.")
    exit(1)

# Step 1: 自定義數據集
class NIRDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Step 2: 定義 CNN 模型
class NIR_CNN(nn.Module):
    def __init__(self, input_size):
        super(NIR_CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)
        conv_output_size = input_size // 2  # 經過一次池化
        conv_output_size = conv_output_size // 2  # 經過第二次池化
        self.fc1 = nn.Linear(32 * conv_output_size, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # [batch, channels=1, seq_len]
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Step 3: 載入 NIR 數據
nir_data = pd.read_csv('1.csv', index_col=0).T
nir_data.index = nir_data.index.str.replace('sample no: ', '', regex=False).str.strip()
print(f"NIR data shape after loading: {nir_data.shape}")

# Step 4: 清理非數值數據
nir_data = nir_data.apply(pd.to_numeric, errors='coerce')
nir_data = nir_data.dropna(axis=1, how='all')
print(f"NIR data shape after cleaning: {nir_data.shape}")

# Step 5: 過濾波長範圍 (850-1100 nm)
wavelengths = nir_data.columns.astype(float)
nir_data = nir_data.loc[:, (wavelengths >= 850) & (wavelengths <= 1100)]
print(f"NIR data shape after wavelength filtering (850-1100 nm): {nir_data.shape}")

# Step 6: 載入化學分析數據
chem_data = pd.read_excel('Samples_91880802_12-06-2025_12-02-58.xlsx', sheet_name='Samples')
chem_data['Sample Number'] = chem_data['Sample Number'].astype(str).str.strip()
chem_data = chem_data.set_index('Sample Number')
print(f"Chem data shape: {chem_data.shape}")
print("Missing values in Protein:", chem_data['Protein'].isna().sum())

# Step 7: 對齊 NIR 數據與化學數據
common_samples = nir_data.index.intersection(chem_data.index)
print(f"Common samples: {len(common_samples)}")
print("Common samples list:", common_samples.tolist())
X = nir_data.loc[common_samples]
y = chem_data.loc[common_samples, 'Protein']
print(f"X shape: {X.shape}, y shape: {y.shape}")

# Step 8: 異常值處理
y = y.dropna()
X = X.loc[y.index]
outliers = y[y > 30].index
if len(outliers) > 0:
    print(f"Removing {len(outliers)} outliers with Protein > 30%")
    X = X.drop(outliers)
    y = y.drop(outliers)
print(f"X shape after outlier removal: {X.shape}, y shape: {y.shape}")

# Step 9: 光譜預處理
# 9.1: Savitzky-Golay 平滑
try:
    X_smooth = savgol_filter(X, window_length=7, polyorder=2, axis=1)
except Exception as e:
    print(f"Error in savgol_filter: {e}")
    print("Skipping Savitzky-Golay smoothing. Using raw data instead.")
    X_smooth = X.values

# 9.2: 標準正態變換（SNV）
X_snv = (X_smooth - X_smooth.mean(axis=1)[:, np.newaxis]) / X_smooth.std(axis=1)[:, np.newaxis]

# Step 10: 目標變量標準化
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()
print(f"X_snv shape: {X_snv.shape}, y_scaled shape: {y_scaled.shape}")

# Step 11: 分割數據集
X_train, X_test, y_train, y_test = train_test_split(X_snv, y_scaled, test_size=0.2, random_state=42)
print(f"Training set: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"Test set: X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Step 12: 創建數據加載器
train_dataset = NIRDataset(X_train, y_train)
test_dataset = NIRDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Step 13: 初始化模型、損失函數和優化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = NIR_CNN(input_size=X_snv.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 14: 訓練模型
num_epochs = 30000
train_losses = []
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * X_batch.size(0)
    train_losses.append(epoch_loss / len(train_loader.dataset))
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_losses[-1]:.4f}')

# Step 15: 評估模型
model.eval()
y_pred_scaled = []
y_test_scaled = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        y_pred_scaled.append(outputs.cpu().numpy())
        y_test_scaled.append(y_batch.numpy())
y_pred_scaled = np.concatenate(y_pred_scaled).ravel()
y_test_scaled = np.concatenate(y_test_scaled).ravel()

# 還原標準化
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
y_test_original = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).ravel()

# Step 16: 計算指標
mse = mean_squared_error(y_test_original, y_pred)
r2 = r2_score(y_test_original, y_pred)
print(f'MSE: {mse:.4f}, R²: {r2:.4f}')

# Step 17: 繪製預測與實際值的散點圖
plt.scatter(y_test_original, y_pred, alpha=0.5)
plt.xlabel('Actual Protein Content (%)')
plt.ylabel('Predicted Protein Content (%)')
plt.title('CNN Prediction of Protein Content')
plt.plot([min(y_test_original), max(y_test_original)], [min(y_test_original), max(y_test_original)], 'r--')
plt.show()

# Step 18: 繪製訓練損失曲線
plt.plot(train_losses)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss Curve')
plt.show()