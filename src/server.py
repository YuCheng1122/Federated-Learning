import torch
import logging
from src.client import CNN  
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support
from torch import nn

class FederatedServer:
    def __init__(self):
        self.logger = logging.getLogger('Server')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Server using device: {self.device}")
        
        # 初始化全局模型
        self.global_model = CNN().to(self.device)
        self.round = 0
        
        # 記錄訓練指標
        self.training_history = {
            'round_losses': [],
            'client_losses': {},
            'model_updates': []
        }

        self.best_acc = 0
        self.best_model_path = None
    
    def distribute_model(self):
        """分發全局模型參數"""
        self.logger.info(f"Distributing global model parameters (Round {self.round + 1})")
        return {
            name: param.data.clone()
            for name, param in self.global_model.named_parameters()
        }
    
    def save_model(self, path, eval_results=None):
        """保存模型和評估結果
        
        Args:
            path (str): 保存路徑
            eval_results (dict, optional): 評估結果
        """
        try:
            save_dict = {
                'round': self.round,
                'model_state_dict': self.global_model.state_dict(),
                'training_history': self.training_history,
            }
            
            # 如果有評估結果，也保存
            if eval_results is not None:
                save_dict['eval_results'] = eval_results
                # 記錄評估指標
                self.logger.info(f"Saving model with evaluation results:")
                self.logger.info(f"Accuracy: {eval_results['accuracy']:.2f}%")
                self.logger.info(f"F1 Score: {eval_results['f1_score']:.4f}")
                
                # 處理警告：添加 zero_division 參數
                if 'precision' in eval_results:
                    self.logger.info(f"Precision: {eval_results['precision']:.4f}")
                if 'recall' in eval_results:
                    self.logger.info(f"Recall: {eval_results['recall']:.4f}")
                if 'confusion_matrix' in eval_results:
                    self.logger.info(f"Confusion Matrix:\n{eval_results['confusion_matrix']}")
            
            torch.save(save_dict, path)
            self.logger.info(f"Model saved to {path}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, path):
        """載入全局模型"""
        try:
            checkpoint = torch.load(path)
            self.round = checkpoint['round']
            self.global_model.load_state_dict(checkpoint['model_state_dict'])
            self.training_history = checkpoint['training_history']
            self.logger.info(f"Model loaded from {path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
    
    def get_training_metrics(self):
        """獲取訓練指標"""
        return {
            'current_round': self.round,
            'average_loss_history': self.training_history['round_losses'],
            'client_loss_history': self.training_history['client_losses']
        }
    
    def evaluate(self, test_loader):
        """評估全局模型"""
        self.global_model.eval()
        total_loss = 0
        correct = 0
        total = 0
        criterion = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device).squeeze()
                
                outputs = self.global_model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(test_loader)
        
        self.logger.info(f"Global Model Evaluation - "
                        f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }
    
    def aggregate_weighted(self, client_updates):
        self.logger.info(f"開始聚合 {len(client_updates)} 個客戶端的更新")
        updates = [update[0] for update in client_updates]
        losses = [update[1] for update in client_updates]
        sample_sizes = [update[2] for update in client_updates]
        
        # 計算權重
        total_samples = sum(sample_sizes)
        weights = [size / total_samples for size in sample_sizes]
        weights_tensor = torch.tensor(weights).to(self.device)
        
        # 聚合參數
        aggregated_params = {}
        for name, param in self.global_model.named_parameters():
            # 獲取當前參數值
            current_param = param.data.clone()
            
            # 計算參數更新
            updates_tensor = torch.stack([
                update[name].to(self.device) for update in updates
            ])
            
            # 根據參數的維度調整權重的形狀
            if len(param.shape) == 4:  # 卷積層權重
                weights_view = weights_tensor.view(-1, 1, 1, 1, 1)
            elif len(param.shape) == 2:  # 全連接層權重
                weights_view = weights_tensor.view(-1, 1, 1)
            elif len(param.shape) == 1:  # 偏置項
                weights_view = weights_tensor.view(-1, 1)
            else:
                weights_view = weights_tensor.view(-1)
            
            # 應用權重並求和
            weighted_updates = torch.sum(updates_tensor * weights_view, dim=0)
            
            # 檢查更新幅度
            param_diff = torch.norm(weighted_updates - current_param).item()
            self.logger.info(f"參數 {name} 的實際更新幅度: {param_diff:.4f}")
            
            if param_diff < 1e-6:
                self.logger.warning(f"Warning: 參數 {name} 的更新幅度過小")
            
            aggregated_params[name] = weighted_updates
        
        # 更新全局模型
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                param.data.copy_(aggregated_params[name])
        
        return sum(loss * weight for loss, weight in zip(losses, weights))

    def evaluate_global(self, test_loader):
        """評估全局模型性能
        
        Args:
            test_loader: DataLoader for test dataset
        
        Returns:
            dict: 包含各種評估指標的字典
        """
        self.logger.info("Evaluating global model...")
        self.global_model.eval()  # 設置為評估模式
        
        total = 0
        correct = 0
        total_loss = 0
        all_preds = []
        all_targets = []
        criterion = nn.CrossEntropyLoss()
        
        try:
            with torch.no_grad():  # 不計算梯度
                for batch_idx, (images, labels) in enumerate(test_loader):
                    images = images.to(self.device)
                    labels = labels.to(self.device).squeeze()
                    
                    # 前向傳播
                    outputs = self.global_model(images)
                    loss = criterion(outputs, labels)
                    
                    # 記錄損失
                    total_loss += loss.item()
                    
                    # 計算準確率
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                    
                    # 保存預測結果用於計算其他指標
                    all_preds.extend(predicted.cpu().numpy())
                    all_targets.extend(labels.cpu().numpy())
            
            # 計算各種指標
            accuracy = 100. * correct / total
            avg_loss = total_loss / len(test_loader)
            
            # 計算F1分數和混淆矩陣
            f1 = f1_score(all_targets, all_preds, average='weighted')
            conf_matrix = confusion_matrix(all_targets, all_preds)
            
            # 計算每個類別的精確度和召回率
            precision, recall, _, _ = precision_recall_fscore_support(
                all_targets, 
                all_preds, 
                average='weighted',
                zero_division=0
            )
            
            # 記錄評估結果
            eval_results = {
                'loss': avg_loss,
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'confusion_matrix': conf_matrix,
                'correct': correct,
                'total': total
            }
            
            # 輸出詳細日誌
            self.logger.info(f"Global Model Evaluation Results:")
            self.logger.info(f"Loss: {avg_loss:.4f}")
            self.logger.info(f"Accuracy: {accuracy:.2f}%")
            self.logger.info(f"F1 Score: {f1:.4f}")
            self.logger.info(f"Precision: {precision:.4f}")
            self.logger.info(f"Recall: {recall:.4f}")
            self.logger.info(f"Confusion Matrix:\n{conf_matrix}")
            
            return eval_results
            
        except Exception as e:
            self.logger.error(f"Error during global evaluation: {str(e)}")
            raise