import os
import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from data.image_folder import IndexedImageFolder

def build_stanford_dogs_dataset(root, train_transform, test_transform):
    """
    构建Stanford Dogs数据集
    
    参数:
        root: 数据集根目录，应包含train和val子目录
        train_transform: 训练数据的转换
        test_transform: 测试数据的转换
    
    返回:
        包含训练和测试数据集的字典
    """
    train_data = IndexedImageFolder(os.path.join(root, 'train'), transform=train_transform)
    test_data = IndexedImageFolder(os.path.join(root, 'val'), transform=test_transform)
    return {'train': train_data, 'test': test_data, 'n_train_samples': len(train_data.samples), 'n_test_samples': len(test_data.samples)}


def build_dummy_stanford_dogs_dataset(root, train_transform, test_transform):
    """
    构建虚拟的Stanford Dogs数据集，用于测试代码
    
    参数:
        root: 数据集根目录
        train_transform: 训练数据的转换
        test_transform: 测试数据的转换
    
    返回:
        包含训练和测试数据集的字典
    """
    from data.dummy_dataset import DummyWebDataset
    
    # Stanford Dogs有120个类别
    num_classes = 120
    
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