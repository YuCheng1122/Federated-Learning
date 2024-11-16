import os

class ModelConfig:
    # 路徑設置
    BASE_DIR = '/home/tommy/cySecAIMidterm/images'
    LOG_DIR = 'logs'
    
    # 模型參數
    INPUT_CHANNELS = 3    
    IMAGE_SIZE = 224      # 224x224
    NUM_CLASSES = 2       # 二分類
    
    # 訓練參數
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    LOCAL_EPOCHS = 5
    
    # 聯邦學習參數
    MIN_CLIENTS = 2
    MAX_CLIENTS = 3
    ROUNDS = 20
    
    # 數據處理參數
    TRAIN_SPLIT = 0.8
    RANDOM_SEED = 42

    AGGREGATION_CONFIG = {
        'data_weight': 0.7,  # 數據量權重比例
        'performance_weight': 0.3  # 性能權重比例
    }   
    
    @staticmethod
    def make_dirs():
        """創建必要的目錄"""
        os.makedirs(ModelConfig.LOG_DIR, exist_ok=True)