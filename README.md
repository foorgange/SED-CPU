# SED-CPU: 支持CPU训练的自适应类别平衡噪声标签学习方法

## 改动说明

本仓库是对原始SED项目的Fork版本，在原有基础上进行了以下改动：

1. **添加了对Stanford Dogs数据集的支持**
   - 创建了`data/stanford_dogs.py`用于加载Stanford Dogs数据集
   - 修改了`main_web.py`中的`build_loader`函数以支持Stanford Dogs数据集
   - 修改了`main_web.py`中的`build_model`函数，使Stanford Dogs数据集也使用ResNet模型
   - 添加了`download_stanford_dogs.py`脚本用于下载和处理Stanford Dogs数据集

2. **适配了CPU训练环境**
   - 修改了相关代码逻辑，使项目可以在没有GPU的环境中运行
   - 优化了批处理大小和训练参数，适应CPU训练需求
   - 所有训练命令均提供了CPU版本，添加`--gpu -1`参数即可使用

3. **更新了依赖管理**
   - 修改了`requirements.txt`，移除了conda相关依赖，适配venv虚拟环境
   - 添加了镜像源安装指南，提高依赖安装成功率

4. **添加了详细的项目复现学习记录**
   - 创建了`SED项目复现学习实录.md`文件，记录了完整的学习和复现过程
   - 提供了详细的环境配置、数据集准备和训练命令说明

> 特别感谢原项目作者的优秀工作。本改动旨在扩展项目功能并提高其可访问性，所有原始功能和实现均保持不变。

---

# 原项目：SED: Foster Adaptivity and Balance in Learning with Noisy Labels
**Abstract:** Label noise is ubiquitous in real-world scenarios, posing a practical challenge to supervised models due to its effect in hurting the generalization performance of deep neural networks.
Existing methods primarily employ the sample selection paradigm and usually rely on dataset-dependent prior knowledge (e.g., a pre-defined threshold) to cope with label noise, inevitably degrading the adaptivity. Moreover, existing methods tend to neglect the class balance in selecting samples, leading to biased model performance.
To this end, we propose a simple yet effective approach named SED to deal with label noise in a Self-adaptivE and class-balanceD manner. 
Specifically, we first design a novel sample selection strategy to empower self-adaptivity and class balance when identifying clean and noisy data.
A mean-teacher model is then employed to correct labels of noisy samples.
Subsequently, we propose a self-adaptive and class-balanced sample re-weighting mechanism to assign different weights to detected noisy samples.
Finally, we additionally employ consistency regularization on selected clean samples to improve model generalization performance.
Extensive experimental results on synthetic and real-world datasets demonstrate the effectiveness and superiority of our proposed method.

# Pipeline

![framework](Figure.png)

# Installation

原始安装命令：
```
pip install -r requirements.txt
```

如果安装过程中遇到问题，可以尝试使用镜像源安装：
```
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 环境要求

- Python 3.8 或 3.9
- 虚拟环境设置（推荐）：
  ```
  python -m venv svenv
  svenv\Scripts\activate  # Windows
  source svenv/bin/activate  # Linux/Mac
  ```

### 注意事项

- 如果使用CPU训练，请确保在训练命令中添加 `--gpu -1` 参数
- 对于Windows用户，安装pyvips可能需要额外步骤：
  1. 安装Python包：`pip install pyvips`
  2. 下载libvips C库：https://github.com/libvips/build-win64-mxe/releases
  3. 将bin目录添加到系统环境变量PATH中

# Datasets
We conduct noise robustness experiments on two synthetically corrupted datasets (i.e., CIFAR100N and CIFAR80N) and three real-world datasets (i.e., Web-Aircraft, Web-Car and Web-Bird.
Specifically, we create the closed-set noisy dataset CIFAR100N and the open-set noisy dataset CIFAR80N based on CIFAR100.
To make the open-set noisy dataset CIFAR80N, we regard the last 20 categories in CIFAR100 as out-of-distribution. 
We adopt two classic noise structures: symmetric and asymmetric, with a noise ratio $n \in (0,1)$.

You can download the CIFAR10 and CIFAR100 on [this](https://www.cs.toronto.edu/~kriz/cifar.html).

You can download the Clothing1M from [here](https://github.com/NUST-Machine-Intelligence-Laboratory/weblyFG-dataset).

# Training

An example shell script to run SED on CIFAR-100N :

```python
CUDA_VISIBLE_DEVICES=0 python main.py --warmup-epoch 20 --epoch 100 --batch-size 128 --lr 0.05 --warmup-lr 0.05  --noise-type symmetric --closeset-ratio 0.2 --lr-decay cosine:20,5e-4,100  --opt sgd --dataset cifar100nc
```

如果没有GPU，可以使用CPU版本的命令：

```python
python main.py --warmup-epoch 5 --epoch 20 --batch-size 16 --lr 0.01 --warmup-lr 0.01 --noise-type symmetric --closeset-ratio 0.2 --lr-decay cosine:5,5e-4,20 --opt sgd --dataset cifar100nc --gpu -1
```

An example shell script to run SED on CIFAR-80N :

```python
CUDA_VISIBLE_DEVICES=0 python main.py --warmup-epoch 20 --epoch 100 --batch-size 128 --lr 0.05 --warmup-lr 0.05  --noise-type symmetric --closeset-ratio 0.2 --lr-decay cosine:20,5e-4,100  --opt sgd --dataset cifar80no
```

Here is an example shell script to run SED on Web-Aircraft :

```python
CUDA_VISIBLE_DEVICES=0 python main_web.py --warmup-epoch 10 --epoch 110 --batch-size 32 --lr 0.005 --warmup-lr 0.005  --lr-decay cosine:10,5e-4,110 --weight-decay 5e-4 --seed 123 --opt sgd --dataset web-bird --SSL True --gpu 0 --pretrain True
```

## Stanford Dogs Dataset

我们添加了对Stanford Dogs数据集的支持。Stanford Dogs数据集包含120个狗品种的图像，共约20,000张图像。

### 下载和准备数据集

使用提供的脚本下载和准备Stanford Dogs数据集：

```python
python download_stanford_dogs.py
```

这个脚本会自动下载数据集文件，并将其组织成适合训练的目录结构：

```
Datasets/
└── stanford-dogs/
    ├── train/
    │   ├── n02085620-Chihuahua/
    │   ├── n02085782-Japanese_spaniel/
    │   └── ...
    └── val/
        ├── n02085620-Chihuahua/
        ├── n02085782-Japanese_spaniel/
        └── ...
```

### 训练命令

使用以下命令在Stanford Dogs数据集上训练模型：

```python
CUDA_VISIBLE_DEVICES=0 python main_web.py --warmup-epoch 10 --epoch 100 --batch-size 32 --lr 0.005 --warmup-lr 0.005 --lr-decay cosine:10,5e-4,100 --weight-decay 5e-4 --opt sgd --dataset stanford-dogs --SSL True --pretrain True
```

CPU版本的训练命令：

```python
python main_web.py --warmup-epoch 5 --epoch 20 --batch-size 16 --lr 0.005 --warmup-lr 0.005 --lr-decay cosine:5,5e-4,20 --weight-decay 5e-4 --opt sgd --dataset stanford-dogs --SSL True --pretrain True --gpu -1
```

# Results on Cifar100N and Cifar80N:

![framework](Table1.png)


# Results on Web-Aircraft, Web-Bird, and Web-Car:

![framework](Table2.png)


# Effects of different components in test accuracy (%) on CIFAR100N (noise rate and noise type are 0.5 and symmetric, respectively)

![framework](Table3.png)

