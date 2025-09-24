import torch
import torch.nn as nn
import torch.optim as optim
from draw_model import Model
from get_yaml import get_loader_from_yaml
from pathlib import Path

def train_model(yaml_path, num_epochs=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 載入模型
    model = Model("test.yaml").to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 載入資料集
    train_loader = get_loader_from_yaml(yaml_path, split='train')
    val_loader = get_loader_from_yaml(yaml_path, split='val')

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(num_epochs):
        # 訓練階段
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss / len(train_loader):.4f}")

        # 驗證階段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_loss /= len(val_loader)
        val_acc = 100 * correct / total
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")

        # 學習率調度
        scheduler.step()

        # 早停機制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_path = Path("models/model.pt")
            save_path.parent.mkdir(exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break

if __name__ == '__main__':
    yaml_path = r"D:\Users\wedke\document\extra document\大三上\113專題研究之製作\論文人生\紅外線\red_coding main\data\mydata.yaml"
    train_model(yaml_path)