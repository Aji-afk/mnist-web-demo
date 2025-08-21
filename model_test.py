import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# ----- Step 1: 定義簡單的CNN模型 (跟MNIST相容) -----
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
    
# 載入已訓練模型（假設你有 mnist_cnn.pth）
model = SimpleCNN()
model.load_state_dict(torch.load("saved_model.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# ----- Step 2: 建立畫布 -----
class PaintApp:
    def __init__(self, root):
        self.root = root
        self.root.title("手寫數字辨識 (MNIST)")

        self.canvas = tk.Canvas(self.root, width=200, height=200, bg="white")
        self.canvas.pack()

        self.image = Image.new("RGB", (200, 200), "white")
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)

        btn_frame = tk.Frame(self.root)
        btn_frame.pack()
        tk.Button(btn_frame, text="辨識", command=self.predict).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="清除", command=self.clear).pack(side=tk.LEFT)

        self.label = tk.Label(self.root, text="請寫一個數字")
        self.label.pack()

    def paint(self, event):
        x1, y1 = (event.x - 14), (event.y - 14)
        x2, y2 = (event.x + 14), (event.y + 14)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black")
        self.draw.ellipse([x1, y1, x2, y2], fill="black")

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 200, 200], fill="white")
        self.label.config(text="請寫一個數字")

    def predict(self):
        img = self.image.convert("L")
        img = ImageOps.invert(img)  # 黑底白字
        img = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(img)
            pred = output.argmax(dim=1, keepdim=True).item()
        self.label.config(text=f"辨識結果: {pred}")

if __name__ == "__main__":
    root = tk.Tk()
    app = PaintApp(root)
    root.mainloop()
