import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import shutil

def scan_dataset(data_dir):
    """
    扫描数据集目录，返回图像路径和对应的标签
    
    参数:
        data_dir: 数据集根目录，包含NEC-tif和normal-tif两个子文件夹
        
    返回:
        image_paths: 所有图像的路径列表
        labels: 对应的标签列表（1表示有NEC特征，0表示正常或无NEC特征）
        class_names: 类别名称列表
    """
    image_paths = []
    labels = []
    class_names = ['Normal', 'NEC']
    
    # 处理NEC-tif文件夹（包含NEC病例的X光照片）
    nec_dir = os.path.join(data_dir, 'NEC-tif')
    nec_excel = os.path.join(nec_dir, 'nec.xlsx')
    
    if os.path.exists(nec_dir) and os.path.exists(nec_excel):
        # 读取Excel文件获取标签信息
        try:
            # 第一列为图片名，第二列为标签（是否具有NEC特征，是为1，否则为0）
            nec_df = pd.read_excel(nec_excel, header=0)  # header=0 表示第一行为表头
            print(f"成功读取NEC标签文件: {nec_excel}")
            print(f"NEC数据集包含 {len(nec_df)} 条记录")
            
            # 获取列名
            columns = nec_df.columns.tolist()
            img_col = columns[0]  # 第一列为图片名称列
            label_col = columns[1]  # 第二列为标签列

            for _, row in nec_df.iterrows():
                img_name = row[img_col]  # 使用列名访问数据
                is_nec_feature = row[label_col]  # 使用列名访问数据

                # 确保图片名称是字符串，并拼接.tif后缀
                img_full_name = str(img_name)
                img_path = os.path.join(nec_dir, img_full_name)
                
                if os.path.exists(img_path):
                    image_paths.append(img_path)
                    labels.append(int(is_nec_feature))  # 1表示有NEC特征，0表示无NEC特征
                else:
                    print(f"警告: 图片文件 {img_path} 不存在，已跳过")
        except Exception as e:
            print(f"读取NEC标签文件时出错: {e}")
    else:
        print(f"警告: NEC-tif目录或nec.xlsx文件不存在")
    
    # 处理normal-tif文件夹（包含正常的X光照片）
    normal_dir = os.path.join(data_dir, 'normal-tif')
    normal_excel = os.path.join(normal_dir, 'normal.xlsx')
    
    if os.path.exists(normal_dir) and os.path.exists(normal_excel):
        # 读取Excel文件获取标签信息
        try:
            # 假设normal.xlsx的结构与nec.xlsx类似，第一列为图片名，所有normal图片标签为0
            normal_df = pd.read_excel(normal_excel, header=0)
            print(f"成功读取Normal标签文件: {normal_excel}")
            print(f"Normal数据集包含 {len(normal_df)} 条记录")
            
            for _, row in normal_df.iterrows():
                img_name = row.iloc[0]  # 第一列为图片名称
                
                # 确保图片名称是字符串，并拼接.tif后缀
                img_full_name = str(img_name)
                img_path = os.path.join(normal_dir, img_full_name)
                
                if os.path.exists(img_path):
                    image_paths.append(img_path)
                    labels.append(0)  # 正常图像标签为0
                else:
                    print(f"警告: 图片文件 {img_path} 不存在，已跳过")
        except Exception as e:
            print(f"读取Normal标签文件时出错: {e}")
    else:
        print(f"警告: normal-tif目录或normal.xlsx文件不存在")
    
    return image_paths, labels, class_names

def split_dataset(image_paths, labels, test_size=0.2, val_size=0.1, random_state=42):
    """
    将数据集分割为训练集、验证集和测试集
    
    参数:
        image_paths: 图像路径列表
        labels: 标签列表
        test_size: 测试集比例
        val_size: 验证集比例（相对于训练集）
        random_state: 随机种子
        
    返回:
        train_paths, val_paths, test_paths: 分割后的图像路径
        train_labels, val_labels, test_labels: 分割后的标签
    """
    # 首先分割出测试集
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # 从剩余数据中分割出验证集
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, 
        test_size=val_size/(1-test_size),  # 调整验证集比例
        random_state=random_state, 
        stratify=train_val_labels
    )
    
    print(f"训练集: {len(train_paths)} 张图像")
    print(f"验证集: {len(val_paths)} 张图像")
    print(f"测试集: {len(test_paths)} 张图像")
    
    return train_paths, val_paths, test_paths, train_labels, val_labels, test_labels

def create_organized_dataset(image_paths, labels, class_names, output_dir):
    """
    创建组织良好的数据集目录结构
    
    参数:
        image_paths: 图像路径列表
        labels: 标签列表
        class_names: 类别名称列表
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 为每个类别创建子目录
    for class_name in class_names:
        os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)
    
    # 复制图像到对应的类别目录
    for path, label in zip(image_paths, labels):
        class_name = class_names[label]
        filename = os.path.basename(path)
        dst_path = os.path.join(output_dir, class_name, filename)
        
        try:
            shutil.copy2(path, dst_path)
        except Exception as e:
            print(f"复制 {path} 到 {dst_path} 失败: {e}")
    
    print(f"已创建组织良好的数据集在 {output_dir}")

def save_dataset_splits(train_paths, val_paths, test_paths, train_labels, val_labels, test_labels, 
                       class_names, output_dir):
    """
    保存数据集分割信息到CSV文件
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建训练集CSV
    train_df = pd.DataFrame({
        'path': train_paths,
        'label_id': train_labels,
        'label_name': [class_names[label] for label in train_labels]
    })
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    
    # 创建验证集CSV
    val_df = pd.DataFrame({
        'path': val_paths,
        'label_id': val_labels,
        'label_name': [class_names[label] for label in val_labels]
    })
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    
    # 创建测试集CSV
    test_df = pd.DataFrame({
        'path': test_paths,
        'label_id': test_labels,
        'label_name': [class_names[label] for label in test_labels]
    })
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    print(f"数据集分割信息已保存到 {output_dir}")

def main():
    print("NEC医学图像数据集准备工具")
    print("=" * 50)
    
    # 设置默认数据集路径，用户也可以输入其他路径
    default_data_dir = "/media/sophgo/4888-7CDF"
    data_dir = input(f"请输入数据集根目录路径 (默认: {default_data_dir}): ") or default_data_dir
    
    if not os.path.exists(data_dir):
        print(f"错误: 目录 {data_dir} 不存在")
        return
    
    # 检查是否存在必要的子文件夹
    nec_dir = os.path.join(data_dir, 'NEC-tif')
    normal_dir = os.path.join(data_dir, 'normal-tif')
    
    if not (os.path.exists(nec_dir) and os.path.exists(normal_dir)):
        print(f"错误: 在 {data_dir} 中未找到必要的子文件夹 (NEC-tif 和 normal-tif)")
        return
    
    # 扫描数据集
    print("\n扫描数据集...")
    image_paths, labels, class_names = scan_dataset(data_dir)
    
    if not image_paths:
        print("错误: 未找到任何图像文件或无法读取Excel标签文件")
        return

    print(f"Length of image_paths: {len(image_paths)}")
    print(f"Length of labels: {len(labels)}")

    if len(image_paths) != len(labels):
        print("错误：image_paths 和 labels 长度不一致！请检查 scan_dataset 函数的逻辑。")
        # 在这里可以进一步打印列表内容，帮助定位问题
        # for i in range(min(len(image_paths), len(labels))):
        #     print(f"{image_paths[i]} - {labels[i]}")
        return 

    # 分割数据集
    print("\n分割数据集...")
    train_paths, val_paths, test_paths, train_labels, val_labels, test_labels = split_dataset(
        image_paths, labels
    )
    
    # 保存分割信息
    output_dir = os.path.join(os.getcwd(), 'dataset_info')
    save_dataset_splits(
        train_paths, val_paths, test_paths,
        train_labels, val_labels, test_labels,
        class_names, output_dir
    )
    
    # 询问是否创建组织良好的数据集
    create_organized = input("\n是否创建组织良好的数据集目录结构？(y/n): ").lower() == 'y'
    if create_organized:
        organized_dir = os.path.join(os.path.dirname(data_dir), 'organized_dataset')
        create_organized_dataset(image_paths, labels, class_names, organized_dir)
    
    print("\n数据集准备完成!")
    print(f"处理了 {len(image_paths)} 张图像，其中 {labels.count(1)} 张有NEC特征，{labels.count(0)} 张正常或无NEC特征")
    print("现在您可以使用 medical_image_classification.py 脚本进行模型训练")

if __name__ == "__main__":
    main()