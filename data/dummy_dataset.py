import os
import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class DummyWebDataset(Dataset):
    """
    创建一个虚拟的Web数据集，用于测试代码而不需要实际的数据集
    """
    def __init__(self, root, transform=None, num_classes=200, samples_per_class=10):
        self.transform = transform
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.samples = []
        self.targets = []
        
        # 创建虚拟样本和标签
        for class_idx in range(num_classes):
            for sample_idx in range(samples_per_class):
                # 创建一个随机的RGB图像 (3, 224, 224)
                img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
                self.samples.append((img, class_idx))
                self.targets.append(class_idx)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img, target = self.samples[idx]
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target, idx

def build_dummy_webfg_dataset(root, train_transform, test_transform, dataset_name="web-bird"):
    """
    构建虚拟的webfg数据集，用于测试代码
    """
    class_counts = {"web-aircraft": 100, "web-bird": 200, "web-car": 196}
    num_classes = class_counts.get(dataset_name, 200)
    
    train_data = DummyWebDataset(os.path.join(root, 'train'), 
                                transform=train_transform, 
                                num_classes=num_classes, 
                                samples_per_class=20)
    
    test_data = DummyWebDataset(os.path.join(root, 'val'), 
                               transform=test_transform, 
                               num_classes=num_classes, 
                               samples_per_class=5)
    
    return {'train': train_data, 'test': test_data, 
            'n_train_samples': len(train_data.samples), 
            'n_test_samples': len(test_data.samples)}