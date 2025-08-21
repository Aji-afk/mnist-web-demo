import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --- 設定是否使用 GPU ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用裝置：{device}")

# --- 資料預處理與下載 ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# --- 定義 CNN 模型 ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)   # 28x28 → 26x26
        self.pool = nn.MaxPool2d(2, 2)                 # 26x26 → 13x13
        self.fc1 = nn.Linear(32 * 13 * 13, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # TODO: 實作第一層
        x = self.conv1(x)  # 第一層卷積
        x = torch.relu(x)  # 激活函數
        x = self.pool(x)  # 池化層
        x = x.view(x.size(0), -1)                # 展平成全連接層輸入
        # TODO: 全連接層
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)  # 第二個全連接層
        return x

model = SimpleCNN().to(device)

# --- 定義損失與優化器 ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- 開始訓練 ---
EPOCHS = 5
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # TODO: 條件：清除梯度
        optimizer.zero_grad()
        # TODO: 前向傳播
        outputs = model(images)
        # TODO: 計算損失
        loss = criterion(outputs, labels)
        # TODO: 反向傳播 + 更新參數
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss:.4f}")

# --- 評估模型 ---
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"測試準確率：{100 * correct / total:.2f}%")

# --- 儲存模型 ---
torch.save(model.state_dict(), "saved_model.pth")
print("模型儲存成功")
