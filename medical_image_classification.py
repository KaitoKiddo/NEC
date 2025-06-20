import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
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

# 从CSV文件加载数据
def load_data_from_csv(csv_path):
    if not os.path.exists(csv_path):
        print(f"错误: CSV 文件 {csv_path} 不存在")
        return None, None, None
    try:
        df = pd.read_csv(csv_path)
        image_paths = df['path'].tolist()
        labels = df['label_id'].tolist()
        # 确保所有路径都是绝对路径，如果它们已经是绝对路径，os.path.join不会改变它们
        # 如果csv中的路径是相对于某个基准目录的，需要在这里调整
        # 例如: image_paths = [os.path.join(BASE_DATA_DIR, p) for p in image_paths]
        # 目前假设CSV中的路径是可直接使用的绝对路径或相对于脚本运行位置的相对路径
        
        # 检查图像文件是否存在
        valid_image_paths = []
        valid_labels = []
        for img_path, label in zip(image_paths, labels):
            if os.path.exists(img_path):
                valid_image_paths.append(img_path)
                valid_labels.append(label)
            else:
                print(f"警告: 图像文件 {img_path} 未找到，将跳过此图像。")
        
        if not valid_image_paths:
            print(f"错误: 在 {csv_path} 中没有找到任何有效的图像文件路径。")
            return None, None, None

        # 获取类名并按照指定顺序排列 - ['Normal', 'NEC']
        class_names = ['Normal', 'NEC']
        return valid_image_paths, valid_labels, class_names
    except Exception as e:
        print(f"从 {csv_path} 加载数据时出错: {e}")
        return None, None, None

# 主函数
def main():
    dataset_info_dir = os.path.join(os.getcwd(), "dataset_info")
    train_csv_path = os.path.join(dataset_info_dir, "train.csv")
    val_csv_path = os.path.join(dataset_info_dir, "val.csv")
    test_csv_path = os.path.join(dataset_info_dir, "test.csv")

    print("\n从CSV文件加载训练数据...")
    train_paths, train_labels, class_names_train = load_data_from_csv(train_csv_path)
    if not train_paths:
        return
    print(f"成功加载训练数据: {len(train_paths)} 张图像")

    print("\n从CSV文件加载验证数据...")
    val_paths, val_labels, class_names_val = load_data_from_csv(val_csv_path)
    if not val_paths:
        return
    print(f"成功加载验证数据: {len(val_paths)} 张图像")

    print("\n从CSV文件加载测试数据...")
    test_paths, test_labels, class_names_test = load_data_from_csv(test_csv_path)
    if not test_paths:
        return
    print(f"成功加载测试数据: {len(test_paths)} 张图像")

    # 假设所有CSV文件中的类别名称一致，使用训练集的类别名称
    class_names = class_names_train
    if not class_names:
        print("错误：未能从CSV文件中获取类别名称。")
        return
    print(f"类别: {class_names}")
    print(f"训练集标签分布: {train_labels.count(1)} 张有NEC特征，{train_labels.count(0)} 张正常或无NEC特征")
    print(f"验证集标签分布: {val_labels.count(1)} 张有NEC特征，{val_labels.count(0)} 张正常或无NEC特征")
    print(f"测试集标签分布: {test_labels.count(1)} 张有NEC特征，{test_labels.count(0)} 张正常或无NEC特征")
    
    # 获取数据变换
    train_transform, val_transform = get_transforms()
    
    # 创建数据集
    try:
        train_dataset = MedicalImageDataset(train_paths, train_labels, transform=train_transform)
        val_dataset = MedicalImageDataset(val_paths, val_labels, transform=val_transform)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
        test_dataset = MedicalImageDataset(test_paths, test_labels, transform=val_transform) # 测试集使用验证集的变换
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
        
        print("成功创建数据加载器")
    except Exception as e:
        print(f"创建数据集或数据加载器时出错: {e}")
        return
    
    # 加载模型
    num_classes = len(class_names) 
    model = load_model(num_classes)
    model = model.to(device)
    
    # 定义损失函数和优化器
    # 计算类别权重 (更偏重少数类NEC)
    # 权重与类别样本数成反比
    # 假设 class_names = ['Normal', 'NEC'] 且 Normal对应标签0, NEC对应标签1
    count_normal = train_labels.count(0) # Normal 标签为 0
    count_nec = train_labels.count(1)    # NEC 标签为 1
    
    # 避免除以零，如果某个类别在训练集中不存在，则权重设为1.0
    # 您可以根据需要调整这个默认值，或者如果某个类别数量为0，则应该抛出错误或警告
    # 为了更侧重NEC，我们可以给NEC更高的权重，例如Normal的权重是NEC样本数，NEC的权重是Normal样本数
    # 或者更简单地，给NEC一个固定的较高倍数权重，例如 Normal:0.5, NEC:5 (如果NEC样本远少于Normal)
    # 这里我们使用之前讨论的与样本数成反比的策略，但可以调整比例因子来增强对NEC的关注
    # 例如，可以给少数类的权重再乘以一个大于1的因子
    nec_emphasis_factor = 5.0 # NEC强调因子，通常取值范围1.0-5.0，值越大对NEC类别的重视程度越高

    weight_normal_raw = (count_normal + count_nec) / (2 * count_normal) if count_normal > 0 else 1.0
    weight_nec_raw = (count_normal + count_nec) / (2 * count_nec) if count_nec > 0 else 1.0
    
    # 应用强调因子到NEC类别
    weight_nec_adjusted = weight_nec_raw * nec_emphasis_factor

    # 根据 class_names 的顺序来确定权重的顺序
    # 我们之前设定 class_names = ['Normal', 'NEC']
    # 所以 weights[0] 对应 Normal, weights[1] 对应 NEC
    class_weights = torch.tensor([weight_normal_raw, weight_nec_adjusted], dtype=torch.float).to(device)
    print(f"类别权重: Normal={weight_normal_raw:.2f} (标签0), NEC(调整后)={weight_nec_adjusted:.2f} (标签1)")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
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

        # 在测试集上评估模型
        print("\n在测试集上评估模型...")
        model.load_state_dict(torch.load('best_model.pth')) # 加载训练过程中保存的最佳模型
        model.eval()
        test_running_loss = 0.0
        test_running_corrects = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(test_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                test_running_loss += loss.item() * inputs.size(0)
                test_running_corrects += torch.sum(preds == labels.data)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        test_epoch_loss = test_running_loss / len(test_loader.dataset)
        test_epoch_acc = test_running_corrects.double() / len(test_loader.dataset)
        print(f'Test Loss: {test_epoch_loss:.4f} Acc: {test_epoch_acc:.4f}')
        
        # 可以进一步计算其他评估指标，如精确度、召回率、F1分数等
        from sklearn.metrics import classification_report
        print(classification_report(all_labels, all_preds, target_names=class_names))

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
        num_classes = 2  # 替换为您的实际类别数
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