import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix
import logging
import numpy as np
from configs.config import ModelConfig
import torch.nn.functional as F
# 設置日誌
logger = logging.getLogger('Client')

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 修改輸入通道為3（RGB）
        self.conv_layers = nn.Sequential(
            # First conv layer: 224x224x3 -> 112x112x32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Second conv layer: 112x112x32 -> 56x56x64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Third conv layer: 56x56x64 -> 28x28x128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Fourth conv layer: 28x28x128 -> 14x14x256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Fifth conv layer: 14x14x256 -> 7x7x512
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # 其餘部分保持不變
        self.fc_input_dim = 512 * 7 * 7
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.fc_input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, ModelConfig.NUM_CLASSES),
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, self.fc_input_dim)
        x = self.fc_layers(x)
        return F.log_softmax(x, dim=1)  

class FederatedClient:
    def __init__(self, client_id, dataset):
        self.client_id = client_id
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 分割訓練集和驗證集
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # 初始化模型
        self.model = CNN().to(self.device)
        
        # 使用AdamW優化器，添加權重衰減
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=ModelConfig.LEARNING_RATE,
            weight_decay=0.01  # L2正則化
        )
        
        # 使用學習率調度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5,
            patience=2,
            verbose=True
        )
        
        self.criterion = nn.NLLLoss()  # 改用 NLLLoss
        
        # 數據加載器
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=ModelConfig.BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=ModelConfig.BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        # 早停相關
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.patience = 10  # 增加 patience

        
        logger.info(f"Client {client_id} initialized with {len(self.train_dataset)} training samples and {len(self.val_dataset)} validation samples")
    
    def set_model_params(self, params):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in params:
                    param.data = params[name].clone().to(self.device)
    
    def get_model_params(self):
        return {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
        }
    
    def train(self):
        best_model_params = None
        best_val_loss = float('inf')
        
        for epoch in range(ModelConfig.LOCAL_EPOCHS):
            # 訓練階段
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device).squeeze()
                
                self.optimizer.zero_grad()
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # 添加L2正則化
                l2_lambda = 0.01
                l2_norm = sum(p.pow(2.0).sum() for p in self.model.parameters())
                loss = loss + l2_lambda * l2_norm
                
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            # 驗證階段
            val_metrics = self.evaluate_local(self.val_loader)
            val_loss = val_metrics['loss']
            
            # 更新學習率
            self.scheduler.step(val_loss)
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_params = self.get_model_params()
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # 早停檢查
            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
            
            # 輸出訓練信息
            logger.info(
                f'Client {self.client_id}, Epoch: {epoch + 1}, '
                f'Train Loss: {train_loss/len(self.train_loader):.4f}, '
                f'Train Acc: {100.*train_correct/train_total:.2f}%, '
                f'Val Loss: {val_loss:.4f}, '
                f'Val Acc: {val_metrics["accuracy"]:.2f}%'
            )
        
        # 使用最佳模型參數
        if best_model_params is not None:
            self.set_model_params(best_model_params)
        
        # 最終評估
        final_metrics = self.evaluate_local(self.val_loader)
        return self.get_model_params(), final_metrics['loss']
    
    def evaluate_local(self, dataloader=None):
        if dataloader is None:
            dataloader = self.val_loader
            
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        # 確保 pred_distribution 在正確的設備上
        pred_distribution = torch.zeros(ModelConfig.NUM_CLASSES).to(self.device)
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device).squeeze()
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # 收集預測和實際標籤（移到 CPU 再轉換為 numpy）
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                
                # 確保在同一設備上進行計算
                pred_distribution += torch.bincount(
                    predicted,  # predicted 已經在 GPU 上
                    minlength=ModelConfig.NUM_CLASSES
                )
        
        # 計算指標
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(dataloader)
        f1 = f1_score(all_targets, all_preds, average='weighted')
        conf_matrix = confusion_matrix(all_targets, all_preds)
        
        # 將 pred_distribution 移到 CPU 再轉換為列表
        pred_distribution = pred_distribution.cpu().tolist()
        
        # 詳細的日誌輸出
        logger.info(f"\nClient {self.client_id} Local Evaluation Results:")
        logger.info(f"Confusion Matrix:\n{conf_matrix}")
        logger.info(f"Prediction Distribution: {pred_distribution}")
        logger.info(f"Accuracy: {accuracy:.2f}%")
        logger.info(f"F1 Score: {f1:.4f}")
        logger.info(f"Average Loss: {avg_loss:.4f}")
        
        # 如果發現模型只預測單一類別，輸出警告
        if len(set(all_preds)) == 1:
            logger.warning(
                f"Warning: Model is predicting only one class ({all_preds[0]})!"
            )
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1_score': f1,
            'confusion_matrix': conf_matrix,
            'pred_distribution': pred_distribution,
            'correct': correct,
            'total': total
        }