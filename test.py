import torch
from configs.config import ModelConfig
from tools.image_processor import MalwareDataset
from src.server import FederatedServer
import logging
import os

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def test_model(model_path='logs/model_final.pth'):
    logger = logging.getLogger('Test')
    
    # 檢查模型文件是否存在
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return
    
    # 載入測試數據
    test_dataset = MalwareDataset(ModelConfig.BASE_DIR)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=ModelConfig.BATCH_SIZE,
        shuffle=False
    )
    
    # 載入訓練好的模型並測試
    try:
        server = FederatedServer()
        server.load_model(model_path)
        
        # 評估模型
        results = server.evaluate(test_loader)
        
        # 輸出結果
        logger.info("=== Test Results ===")
        logger.info(f"Accuracy: {results['accuracy']:.2f}%")
        logger.info(f"Loss: {results['loss']:.4f}")
        logger.info(f"Correct/Total: {results['correct']}/{results['total']}")
        
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")

if __name__ == '__main__':
    setup_logging()
    test_model()