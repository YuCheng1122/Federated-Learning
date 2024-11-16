import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from configs.config import ModelConfig
import logging
import os

class MalwareDataset(Dataset):
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.images = []
        self.labels = []
        self.logger = logging.getLogger('MalwareDataset')
        
        # 載入良性樣本
        benign_dir = os.path.join(base_dir, 'benign')
        self._load_directory(benign_dir, 0)  # 0 表示良性
        
        # 載入惡意樣本
        malware_dir = os.path.join(base_dir, 'malware')
        self._load_directory(malware_dir, 1)  # 1 表示惡意
        
        self.logger.info(f"Loaded {len(self.images)} images in total")
        self.logger.info(f"Benign: {self.labels.count(0)}, Malware: {self.labels.count(1)}")
    
    def _load_directory(self, directory, label):
        if not os.path.exists(directory):
            self.logger.error(f"Directory not found: {directory}")
            return
            
        for filename in os.listdir(directory):
            if filename.endswith('.png'):
                self.images.append(os.path.join(directory, filename))
                self.labels.append(label)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        try:
            # 讀取圖像（使用RGB格式）
            image_path = self.images[idx]
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"Failed to load image: {image_path}")
                # 返回一個零矩陣作為替代
                image = np.zeros((ModelConfig.IMAGE_SIZE, ModelConfig.IMAGE_SIZE, 3))
            else:
                # 轉換BGR到RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # 調整大小
                image = cv2.resize(image, (ModelConfig.IMAGE_SIZE, ModelConfig.IMAGE_SIZE))
            
            # 標準化到[0,1]範圍
            image = image / 255.0
            
            # 轉換為PyTorch張量 [H, W, C] -> [C, H, W]
            image = torch.FloatTensor(image).permute(2, 0, 1)
            label = torch.LongTensor([self.labels[idx]])
            
            return image, label
            
        except Exception as e:
            self.logger.error(f"Error loading image {self.images[idx]}: {str(e)}")
            # 返回一個零矩陣作為替代
            image = torch.zeros((3, ModelConfig.IMAGE_SIZE, ModelConfig.IMAGE_SIZE))
            label = torch.LongTensor([self.labels[idx]])
            return image, label
        
    @staticmethod
    def process_single_image(image_path):
        """處理單個圖像"""
        try:
            # 讀取RGB圖像
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # BGR轉RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 調整大小到224x224
            image = cv2.resize(image, (ModelConfig.IMAGE_SIZE, ModelConfig.IMAGE_SIZE))
            
            # 標準化到[0,1]範圍
            image = image / 255.0
            
            # 轉換為PyTorch張量並調整維度順序
            # 從 [HEIGHT, WIDTH, CHANNELS] 轉換為 [CHANNELS, HEIGHT, WIDTH]
            image = torch.FloatTensor(image).permute(2, 0, 1)
            
            return image
            
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return None