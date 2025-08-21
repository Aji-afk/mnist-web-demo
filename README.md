# MNIST 手寫數字辨識 Web Demo

這是一個簡單的深度學習展示專案，使用 PyTorch 訓練的 CNN 模型來辨識手寫數字 (0-9)。  
前端提供一個類似小畫家的手寫畫布，使用者可以直接畫數字並即時進行預測。  

## 功能特色
- 🎨 可在網頁上手寫數字
- 🖌️ 可調整筆刷粗細與顏色
- 🤖 即時數字辨識
- ⚡ 使用 FastAPI + PyTorch 部署

## 專案結構
mnist-web-demo/
│── README.md            # 專案說明
│── requirements.txt     # 需要的套件
│── mnist_cnn.py         # 訓練程式
│── model_test.py        # 主程式 (Tkinter 畫布 + 推論)
│── saved_model.pth      # 訓練好的模型

## 安裝與執行
1. **Clone 專案**
```js
git clone https://github.com/Aji-afk/mnist-web-demo.git
cd mnist-web-demo
```
2. **安裝套件**
```js
pip install -r requirements.txt
```
3. **啟動程式**
```js
python model_test.py
```