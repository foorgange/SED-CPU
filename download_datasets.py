import os
import sys
import argparse
import requests
import zipfile
import shutil
from tqdm import tqdm

def download_file(url, destination):
    """
    下载文件并显示进度条
    """
    if os.path.exists(destination):
        print(f"文件 {destination} 已存在，跳过下载")
        return
    
    print(f"下载 {url} 到 {destination}")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    with open(destination, 'wb') as f, tqdm(
            desc=destination,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            bar.update(len(data))
            f.write(data)

def extract_zip(zip_path, extract_to):
    """
    解压ZIP文件
    """
    print(f"解压 {zip_path} 到 {extract_to}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in tqdm(zip_ref.infolist(), desc="解压文件"):
            zip_ref.extract(member, extract_to)

def setup_dataset(dataset_name):
    """
    设置指定的数据集
    """
    # 这里应该替换为实际的下载链接
    # 由于weblyFG-dataset没有提供直接下载链接，这里只是示例
    download_urls = {
        "web-bird": "https://example.com/web-bird.zip",
        "web-aircraft": "https://example.com/web-aircraft.zip",
        "web-car": "https://example.com/web-car.zip"
    }
    
    if dataset_name not in download_urls:
        print(f"错误: 不支持的数据集 '{dataset_name}'")
        print(f"支持的数据集: {', '.join(download_urls.keys())}")
        return False
    
    # 创建Datasets目录（如果不存在）
    os.makedirs("Datasets", exist_ok=True)
    
    # 下载数据集
    zip_path = f"Datasets/{dataset_name}.zip"
    download_file(download_urls[dataset_name], zip_path)
    
    # 解压数据集
    extract_to = f"Datasets/{dataset_name}"
    os.makedirs(extract_to, exist_ok=True)
    extract_zip(zip_path, extract_to)
    
    # 检查目录结构
    if not os.path.exists(f"{extract_to}/train") or not os.path.exists(f"{extract_to}/val"):
        print(f"警告: 数据集 {dataset_name} 可能没有正确的目录结构")
        print("请确保数据集包含 'train' 和 'val' 子目录")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="下载和设置weblyFG数据集")
    parser.add_argument("--dataset", type=str, choices=["web-bird", "web-aircraft", "web-car", "all"],
                        default="all", help="要下载的数据集名称")
    args = parser.parse_args()
    
    if args.dataset == "all":
        datasets = ["web-bird", "web-aircraft", "web-car"]
    else:
        datasets = [args.dataset]
    
    for dataset in datasets:
        print(f"\n设置 {dataset} 数据集...")
        if setup_dataset(dataset):
            print(f"{dataset} 数据集设置完成!")
        else:
            print(f"{dataset} 数据集设置失败!")
    
    print("\n注意: 由于weblyFG-dataset没有提供直接下载链接，此脚本仅作为示例")
    print("请访问 https://github.com/NUST-Machine-Intelligence-Laboratory/weblyFG-dataset 获取数据集")
    print("并将下载的数据集解压到 Datasets 目录下的相应子目录中")

if __name__ == "__main__":
    main()