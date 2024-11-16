import os
import shutil
import numpy as np
from tools.image_processor import MalwareDataset
from sklearn.metrics import f1_score
import torch
from configs.config import ModelConfig
from src.client import CNN
import logging
import cv2

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('Preprocess')

def process_single_image(image_path, device):
    """處理單個圖像"""
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None
        
        # 調整大小到224x224
        image = cv2.resize(image, (ModelConfig.IMAGE_SIZE, ModelConfig.IMAGE_SIZE))
        
        # 標準化
        image = image / 255.0
        
        # 轉換為PyTorch張量並移到對應設備
        image = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)  # 添加batch和channel維度
        image = image.to(device)
        return image
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None

def evaluate_sample(sample_path, model, device):
    """評估單個樣本的可分類性"""
    try:
        # 讀取RGB圖像而不是灰度圖像
        image = cv2.imread(sample_path)
        if image is None:
            return None
            
        # BGR轉RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 調整大小
        image = cv2.resize(image, (ModelConfig.IMAGE_SIZE, ModelConfig.IMAGE_SIZE))
        
        # 標準化到[0,1]範圍
        image = image / 255.0
        
        # 轉換為PyTorch張量並調整維度順序
        image = torch.FloatTensor(image).permute(2, 0, 1)  # [H,W,C] -> [C,H,W]
        
        # 添加batch維度並移到正確的設備
        image = image.unsqueeze(0).to(device)  # [1,C,H,W]
        
        # 進行預測
        with torch.no_grad():
            outputs = model(image)
            prob = torch.softmax(outputs, dim=1)
            # 返回預測的確信度
            return prob.max().item()
            
    except Exception as e:
        print(f"Error processing {sample_path}: {str(e)}")
        return None

def select_best_samples(source_dir, dest_dir, num_samples=500):
    logger = setup_logging()
    logger.info("Starting sample selection process...")

    # 確保目標目錄存在
    os.makedirs(os.path.join(dest_dir, 'benign'), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, 'malware'), exist_ok=True)

    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # 載入預訓練模型
    model = CNN().to(device)
    checkpoint = torch.load('/home/tommy/cySecAIFLR/logs/model_final.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info("Model loaded successfully")

    # 處理良性樣本
    benign_scores = []
    benign_files = []
    benign_dir = os.path.join(source_dir, 'benign')
    logger.info(f"Processing benign samples from {benign_dir}")
    
    if not os.path.exists(benign_dir):
        logger.error(f"Directory not found: {benign_dir}")
        return
    
    total_benign = len([f for f in os.listdir(benign_dir) if f.endswith('.png')])
    logger.info(f"Found {total_benign} benign samples")
    
    for i, img_name in enumerate(os.listdir(benign_dir)):
        if img_name.endswith('.png'):
            img_path = os.path.join(benign_dir, img_name)
            score = evaluate_sample(img_path, model, device)
            if score is not None:
                benign_scores.append(score)
                benign_files.append(img_path)
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{total_benign} benign samples")
                
    logger.info(f"Successfully processed {len(benign_scores)} benign samples")
                
    # 處理惡意樣本
    malware_scores = []
    malware_files = []
    malware_dir = os.path.join(source_dir, 'malware')
    logger.info(f"Processing malware samples from {malware_dir}")
    
    if not os.path.exists(malware_dir):
        logger.error(f"Directory not found: {malware_dir}")
        return
    
    total_malware = len([f for f in os.listdir(malware_dir) if f.endswith('.png')])
    logger.info(f"Found {total_malware} malware samples")
    
    for i, img_name in enumerate(os.listdir(malware_dir)):
        if img_name.endswith('.png'):
            img_path = os.path.join(malware_dir, img_name)
            score = evaluate_sample(img_path, model, device)
            if score is not None:
                malware_scores.append(score)
                malware_files.append(img_path)
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{total_malware} malware samples")
                
    logger.info(f"Successfully processed {len(malware_scores)} malware samples")

    # 確保有足夠的樣本
    if len(benign_scores) < num_samples:
        logger.warning(f"Only found {len(benign_scores)} valid benign samples")
        num_samples = min(num_samples, len(benign_scores))
    
    if len(malware_scores) < num_samples:
        logger.warning(f"Only found {len(malware_scores)} valid malware samples")
        num_samples = min(num_samples, len(malware_scores))

    # 選擇最容易分類的樣本
    best_benign = np.argsort(benign_scores)[-num_samples:]
    best_malware = np.argsort(malware_scores)[-num_samples:]

    # 複製文件到新目錄
    logger.info("Copying selected files...")
    for idx in best_benign:
        src = benign_files[idx]
        dst = os.path.join(dest_dir, 'benign', os.path.basename(src))
        shutil.copy2(src, dst)
        
    for idx in best_malware:
        src = malware_files[idx]
        dst = os.path.join(dest_dir, 'malware', os.path.basename(src))
        shutil.copy2(src, dst)

    logger.info(f"Selected and copied {num_samples} samples from each category")
    logger.info(f"Average confidence for selected benign samples: {np.mean([benign_scores[i] for i in best_benign]):.4f}")
    logger.info(f"Average confidence for selected malware samples: {np.mean([malware_scores[i] for i in best_malware]):.4f}")
    logger.info(f"Files saved to {dest_dir}")

if __name__ == "__main__":
    source_dir = '/home/tommy/cySecAIMidterm/images'  # 原始數據目錄
    dest_dir = '/home/tommy/cySecAIFLR/dataset'       # 目標目錄
    select_best_samples(source_dir, dest_dir, num_samples=500)