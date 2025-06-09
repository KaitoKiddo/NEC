import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# 检查是否有可用的GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 自定义数据集类
class MedicalImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 加载灰度图像
        image = Image.open(self.image_paths[idx]).convert('L')
        
        # 将单通道灰度图转为三通道图像（因为预训练模型通常需要三通道输入）
        image = Image.merge('RGB', (image, image, image))
        
        if self.transform:
            image = self.transform(image)
            
        return image, self.labels[idx]

# 数据预处理和增强
def get_transforms():
    # 训练集的变换（包含数据增强）
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整大小以适应预训练模型
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomRotation(10),  # 随机旋转
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet标准化参数
    ])
    
    # 验证集的变换（不包含数据增强）
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

# 加载预训练模型
def load_model(num_classes):
    # 加载预训练的ResNet50模型
    model = models.resnet50(pretrained=True)
    
    # 冻结所有层
    for param in model.parameters():
        param.requires_grad = False
    
    # 替换最后的全连接层以适应我们的分类任务
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )
    
    return model

# 训练模型
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 统计
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())
        
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # 验证阶段
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # 前向传播
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = running_corrects.double() / len(val_loader.dataset)
        
        history['val_loss'].append(epoch_loss)
        history['val_acc'].append(epoch_acc.item())
        
        print(f'Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # 保存最佳模型
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'保存最佳模型，准确率: {best_acc:.4f}')
    
    return model, history

# 绘制训练历史
def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='train')
    plt.plot(history['val_loss'], label='val')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='train')
    plt.plot(history['val_acc'], label='val')
    plt.title('Accuracy')
    plt.legend()
    
    plt.savefig('training_history.png')
    plt.show()

# 主函数
def main():
    # 设置默认数据集路径
    default_data_dir = "/media/sophgo/4888-7CDF"
    data_dir = input(f"请输入数据集根目录路径 (默认: {default_data_dir}): ") or default_data_dir
    
    if not os.path.exists(data_dir):
        print(f"错误: 目录 {data_dir} 不存在")
        return
    
    # 检查是否存在必要的子文件夹
    nec_dir = os.path.join(data_dir, 'NEC-tif')
    normal_dir = os.path.join(data_dir, 'normal-tif')
    
    if not (os.path.exists(nec_dir) and os.path.exists(normal_dir)):
        print(f"错误: 在 {data_dir} 中未找到必要的子文件夹 (NEC-tif 和/或 normal-tif)")
        return
    
    # 导入数据准备模块中的函数
    try:
        from data_preparation import scan_dataset
        print("成功导入数据准备模块")
    except ImportError:
        print("错误: 无法导入data_preparation模块，请确保该文件在当前目录中")
        return
    
    # 扫描数据集
    print("\n扫描数据集...")
    try:
        image_paths, labels, class_names = scan_dataset(data_dir)
        
        if not image_paths:
            print("错误: 未找到任何图像文件或无法读取Excel标签文件")
            return
            
        print(f"成功加载数据集: 共 {len(image_paths)} 张图像")
        print(f"类别: {class_names}")
        print(f"标签分布: {labels.count(1)} 张有NEC特征，{labels.count(0)} 张正常或无NEC特征")
    except Exception as e:
        print(f"加载数据集时出错: {e}")
        return
    
    # 分割数据集
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"训练集: {len(train_paths)} 张图像")
    print(f"验证集: {len(val_paths)} 张图像")
    
    # 获取数据变换
    train_transform, val_transform = get_transforms()
    
    # 创建数据集
    try:
        train_dataset = MedicalImageDataset(train_paths, train_labels, transform=train_transform)
        val_dataset = MedicalImageDataset(val_paths, val_labels, transform=val_transform)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
        
        print("成功创建数据加载器")
    except Exception as e:
        print(f"创建数据集或数据加载器时出错: {e}")
        return
    
    # 加载模型
    num_classes = len(class_names)  # 应该是2: Normal和NEC
    model = load_model(num_classes)
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    
    # 只优化最后一层的参数
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    
    # 询问是否开始训练
    start_training = input("\n是否开始训练模型？(y/n): ").lower() == 'y'
    if start_training:
        print("\n开始训练模型...")
        model, history = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)
        plot_training_history(history)
        
        # 保存模型
        model_save_path = os.path.join(os.getcwd(), 'nec_classification_model.pth')
        torch.save(model.state_dict(), model_save_path)
        print(f"模型已保存到: {model_save_path}")
    else:
        print("\n训练已取消")
    
    print("\n完成!")
    print("您可以随时重新运行此脚本进行模型训练")

# 使用Hugging Face模型的替代方法
def use_huggingface_model():
    # 从Hugging Face加载预训练模型和特征提取器
    model_name = "microsoft/resnet-50"  # 可以替换为其他适合的模型
    
    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        model = AutoModelForImageClassification.from_pretrained(model_name)
        
        print(f"成功加载Hugging Face模型: {model_name}")
        
        # 修改分类头以适应我们的任务
        num_classes = 3  # 替换为您的实际类别数
        if hasattr(model, 'classifier'):
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
        elif hasattr(model, 'fc'):
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        else:
            print("无法识别模型的分类头，请手动修改模型结构")
        
        print("模型已准备好进行微调")
        return model, feature_extractor
    
    except Exception as e:
        print(f"加载Hugging Face模型时出错: {e}")
        print("回退到使用PyTorch内置的预训练模型")
        return None, None

if __name__ == "__main__":
    print("医学图像分类模型 - 迁移学习演示")
    print("="*50)
    
    # 选择使用PyTorch内置模型或Hugging Face模型
    use_hf = input("是否使用Hugging Face模型？(y/n): ").lower() == 'y'
    
    if use_hf:
        model, feature_extractor = use_huggingface_model()
        if model is not None:
            print("请根据您的数据集修改代码以使用Hugging Face模型进行训练")
        else:
            main()
    else:
        main()