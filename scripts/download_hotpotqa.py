import os
import json
import requests
from tqdm import tqdm
import zipfile
from pathlib import Path

def download_file(url, filename):
    """下载文件的函数，支持进度条显示"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    with open(filename, 'wb') as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()

def main():
    # 创建数据目录
    data_dir = Path("data/hotpotqa")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 下载训练集
    train_url = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json"
    train_file = data_dir / "train.json"
    print("下载训练集...")
    download_file(train_url, train_file)
    
    # 下载开发集
    dev_url = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
    dev_file = data_dir / "dev.json"
    print("下载开发集...")
    download_file(dev_url, dev_file)
    
    # 处理开发集，创建500样本的子集
    print("创建开发集子集...")
    with open(dev_file, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    
    # 取前500个样本
    dev_500 = dev_data[:500]
    
    # 保存为jsonl格式
    dev_500_file = data_dir / "dev_500.jsonl"
    with open(dev_500_file, 'w', encoding='utf-8') as f:
        for item in dev_500:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print("数据准备完成！")
    print(f"训练集保存在: {train_file}")
    print(f"开发集保存在: {dev_file}")
    print(f"开发集子集(500样本)保存在: {dev_500_file}")

if __name__ == "__main__":
    main() 