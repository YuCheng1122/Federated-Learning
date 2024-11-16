# Malware Detection Federated Learning

基於CNN的惡意軟體檢測聯邦學習系統。

## 環境需求
- Python 3.12
- PyTorch 1.9+
- OpenCV
- NumPy

## 目錄結構
```plain text
malware_fl/
├── configs/
│   └── config.py        # 配置文件
├── tools/
│   ├── __init__.py
│   └── image_processor.py
├── src/
│   ├── __init__.py
│   ├── client.py
│   └── server.py
├── logs/               # 用於存放日誌和模型
├── main.py            # 主訓練程序
├── test.py            # 簡單的測試程序
└── README.md          # 專案說明
```

## 安裝
```bash
pip install -r requirements.txt
```

## 運行方式
1. 安裝依賴：
```bash
pip install torch torchvision opencv-python numpy
```

2. 運行主腳本：
```bash
python main.py
```
