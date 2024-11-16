import os
import logging
import torch
from torch.utils.data import random_split
import numpy as np
from configs.config import ModelConfig
from tools.image_processor import MalwareDataset
from src.client import FederatedClient
from src.server import FederatedServer
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

def setup_logging():
    """設置日誌"""
    ModelConfig.make_dirs()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(ModelConfig.LOG_DIR, 'federated.log')),
            logging.StreamHandler()
        ]
    )

def select_clients(available_clients, fraction=0.8):
    """選擇部分客戶端參與訓練"""
    num_participants = max(2, int(len(available_clients) * fraction))
    return np.random.choice(available_clients, num_participants, replace=False)    

def distribute_data(dataset, num_clients):
    """更均衡的數據分配"""
    # 按標籤分組
    label_indices = {0: [], 1: []}
    for idx, (_, label) in enumerate(dataset):
        label_indices[label.item()].append(idx)
    
    # 為每個客戶端分配平衡的數據
    client_indices = [[] for _ in range(num_clients)]
    
    for label in label_indices:
        indices = label_indices[label]
        np.random.shuffle(indices)
        samples_per_client = len(indices) // num_clients
        
        for i in range(num_clients):
            start_idx = i * samples_per_client
            end_idx = start_idx + samples_per_client if i < num_clients-1 else len(indices)
            client_indices[i].extend(indices[start_idx:end_idx])
    
    return [torch.utils.data.Subset(dataset, indices) for indices in client_indices]

def main():
    # 設置日誌
    setup_logging()
    logger = logging.getLogger('Main')
    
    logger.info("=== 初始化聯邦學習系統 ===")
    
    # 設置隨機種子
    torch.manual_seed(ModelConfig.RANDOM_SEED)
    np.random.seed(ModelConfig.RANDOM_SEED)
    
    # 檢查CUDA可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用設備: {device}")
    
    # 載入數據集
    logger.info("載入數據集...")
    try:
        full_dataset = MalwareDataset(ModelConfig.BASE_DIR)
        logger.info(f"成功載入 {len(full_dataset)} 個樣本")
        
        # 先分割出預訓練數據
        pretrain_size = int(0.1 * len(full_dataset))
        remaining_size = len(full_dataset) - pretrain_size
        
        pretrain_dataset, remaining_dataset = random_split(
            full_dataset,
            [pretrain_size, remaining_size],
            generator=torch.Generator().manual_seed(ModelConfig.RANDOM_SEED)
        )
        
        # 再從剩餘數據中分割訓練集和測試集
        train_size = int(0.8 * remaining_size)
        test_size = remaining_size - train_size
        
        train_dataset, test_dataset = random_split(
            remaining_dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(ModelConfig.RANDOM_SEED)
        )
        
        logger.info(f"數據集分割 - 預訓練: {pretrain_size}, 訓練集: {train_size}, 測試集: {test_size} 樣本")
        
        # 創建數據加載器
        pretrain_loader = DataLoader(
            pretrain_dataset,
            batch_size=ModelConfig.BATCH_SIZE,
            shuffle=True,
            num_workers=2
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=ModelConfig.BATCH_SIZE,
            shuffle=False,
            num_workers=2
        )
        
    except Exception as e:
        logger.error(f"載入數據集失敗: {str(e)}")
        return
    
    # 創建並預訓練服務器
    logger.info("初始化服務器...")
    server = FederatedServer()
    
    # 預訓練全局模型
    logger.info("\n=== 開始預訓練全局模型 ===")
    server.global_model.train()
    optimizer = optim.Adam(server.global_model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    best_pretrain_acc = 0
    best_pretrain_state = None

    # 修改預訓練部分
    for epoch in range(5):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(pretrain_loader):
            data = data.to(device)
            # 確保 target 是一維的
            target = target.squeeze().to(device)  # 添加 squeeze()
            
            optimizer.zero_grad()
            output = server.global_model(data)
            
            # 添加調試信息
            logger.debug(f"Output shape: {output.shape}")
            logger.debug(f"Target shape: {target.shape}")
            
            try:
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                # 計算準確率
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                total_loss += loss.item()
                
            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}:")
                logger.error(f"Output shape: {output.shape}")
                logger.error(f"Target shape: {target.shape}")
                logger.error(f"Error message: {str(e)}")
                raise
        
        # 輸出預訓練進度
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(pretrain_loader)
        logger.info(f'預訓練 Epoch {epoch+1}: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        # 保存最佳模型
        if accuracy > best_pretrain_acc:
            best_pretrain_acc = accuracy
            best_pretrain_state = server.global_model.state_dict()
    
    # 使用最佳預訓練模型
    if best_pretrain_state is not None:
        server.global_model.load_state_dict(best_pretrain_state)
    
    # 評估預訓練模型
    logger.info("\n=== 預訓練模型評估 ===")
    eval_results = server.evaluate_global(test_loader)
    logger.info(f"預訓練模型準確率: {eval_results['accuracy']:.2f}%")
    logger.info(f"預訓練模型F1分數: {eval_results['f1_score']:.4f}")
    logger.info(f"預訓練模型混淆矩陣:\n{eval_results['confusion_matrix']}")
    
    # 創建並分配客戶端
    logger.info("\n=== 初始化聯邦學習客戶端 ===")
    client_datasets = distribute_data(train_dataset, ModelConfig.MIN_CLIENTS)
    clients = []
    
    for i, client_dataset in enumerate(client_datasets):
        client = FederatedClient(f"client_{i}", client_dataset)
        clients.append(client)
        logger.info(f"創建客戶端 {i} (樣本數: {len(client_dataset)})")
    
    # 開始聯邦學習訓練
    logger.info("\n=== 開始聯邦學習訓練 ===")
    for round in range(ModelConfig.ROUNDS):
        logger.info(f"\n輪次 {round + 1}/{ModelConfig.ROUNDS}")
        
        # 選擇本輪參與的客戶端
        active_clients = select_clients(clients)
        logger.info(f"本輪選擇的客戶端數量: {len(active_clients)}")
        
        # 分發全局模型
        global_params = server.distribute_model()
        
        # 客戶端訓練
        client_updates = []
        for client in active_clients:
            client.set_model_params(global_params)
            updates, loss = client.train()
            
            # 評估客戶端模型
            eval_results = client.evaluate_local()
            logger.info(f"\n=== 客戶端 {client.client_id} 本地評估 ===")
            logger.info(f"混淆矩陣:\n{eval_results['confusion_matrix']}")
            logger.info(f"預測分布: {eval_results['pred_distribution']}")
            logger.info(f"準確率: {eval_results['accuracy']:.2f}%")
            logger.info(f"F1分數: {eval_results['f1_score']:.4f}")
            
            client_updates.append((updates, loss, len(client.train_dataset)))
        
        # 服務器聚合
        avg_loss = server.aggregate_weighted(client_updates)
        logger.info(f"輪次 {round + 1} 完成. 平均損失: {avg_loss:.4f}")
        
        # 定期評估全局模型
        if (round + 1) % 5 == 0:
            eval_results = server.evaluate_global(test_loader)
            logger.info(f"\n全局模型評估 (輪次 {round + 1}):")
            logger.info(f"準確率: {eval_results['accuracy']:.2f}%")
            logger.info(f"F1分數: {eval_results['f1_score']:.4f}")
            logger.info(f"混淆矩陣:\n{eval_results['confusion_matrix']}")
            
            model_path = os.path.join(ModelConfig.LOG_DIR, f'model_round_{round+1}.pth')
            server.save_model(model_path, eval_results)
            logger.info(f"模型保存至: {model_path}")
    
    # 最終評估
    logger.info("\n=== 最終模型評估 ===")
    final_results = server.evaluate_global(test_loader)
    logger.info(f"最終準確率: {final_results['accuracy']:.2f}%")
    logger.info(f"最終F1分數: {final_results['f1_score']:.4f}")
    logger.info(f"最終混淆矩陣:\n{final_results['confusion_matrix']}")
    
    # 保存最終模型
    final_model_path = os.path.join(ModelConfig.LOG_DIR, 'model_final.pth')
    server.save_model(final_model_path, final_results)
    logger.info(f"\n訓練完成！最終模型保存至: {final_model_path}")

if __name__ == "__main__":
    main()