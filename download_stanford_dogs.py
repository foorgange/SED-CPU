import os
import requests
import tarfile
import shutil
from tqdm import tqdm
from scipy.io import loadmat

# 创建必要的目录
os.makedirs('Datasets/stanford-dogs/train', exist_ok=True)
os.makedirs('Datasets/stanford-dogs/val', exist_ok=True)
os.makedirs('downloads', exist_ok=True)

# 下载数据集文件
files_to_download = [
    ('http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar', 'downloads/images.tar'),
    ('http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar', 'downloads/annotation.tar'),
    ('http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar', 'downloads/lists.tar')
]

for url, path in files_to_download:
    if not os.path.exists(path):
        print(f'下载 {url} 到 {path}')
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(path, 'wb') as file, tqdm(desc=path, total=total_size, unit='B', unit_scale=True) as bar:
            for data in response.iter_content(chunk_size=1024):
                file.write(data)
                bar.update(len(data))
    else:
        print(f'{path} 已存在，跳过下载')

# 解压文件
for _, path in files_to_download:
    extract_dir = 'downloads'
    print(f'解压 {path} 到 {extract_dir}')
    with tarfile.open(path) as tar:
        tar.extractall(path=extract_dir)

# 处理数据集
print('处理数据集...')

# 加载训练集和测试集列表
train_data = loadmat('downloads/train_list.mat')
test_data = loadmat('downloads/test_list.mat')

# 获取文件列表
train_files = [f[0][0] for f in train_data['file_list']]
test_files = [f[0][0] for f in test_data['file_list']]

# 处理训练集
print('处理训练集...')
for file_path in tqdm(train_files):
    # 获取类别名称（文件夹名）
    class_name = os.path.dirname(file_path)
    # 创建类别目录
    class_dir = os.path.join('Datasets/stanford-dogs/train', class_name)
    os.makedirs(class_dir, exist_ok=True)
    
    # 复制图像
    src_path = os.path.join('downloads/Images', file_path)
    dst_path = os.path.join('Datasets/stanford-dogs/train', file_path)
    shutil.copy(src_path, dst_path)

# 处理测试集
print('处理测试集...')
for file_path in tqdm(test_files):
    # 获取类别名称（文件夹名）
    class_name = os.path.dirname(file_path)
    # 创建类别目录
    class_dir = os.path.join('Datasets/stanford-dogs/val', class_name)
    os.makedirs(class_dir, exist_ok=True)
    
    # 复制图像
    src_path = os.path.join('downloads/Images', file_path)
    dst_path = os.path.join('Datasets/stanford-dogs/val', file_path)
    shutil.copy(src_path, dst_path)

print('数据集准备完成！')
print('您现在可以使用以下命令训练模型：')
print('python main_web.py --dataset stanford-dogs --batch-size 32 --epochs 100')