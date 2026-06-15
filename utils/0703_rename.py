import os
from pathlib import Path

def rename_images(base_path):
    # 定义两个子文件夹路径
    rainy_dir = Path(base_path) / "rainy"
    gt_dir = Path(base_path) / "gt"
    
    # 检查文件夹是否存在
    if not rainy_dir.exists() or not rainy_dir.is_dir():
        print(f"错误: rainy文件夹不存在: {rainy_dir}")
        return
    
    if not gt_dir.exists() or not gt_dir.is_dir():
        print(f"错误: gt文件夹不存在: {gt_dir}")
        return
    
    # 重命名rainy文件夹中的文件
    print(f"处理rainy文件夹: {rainy_dir}")
    rename_files_in_folder(rainy_dir, "rain")
    
    # 重命名gt文件夹中的文件
    print(f"\n处理gt文件夹: {gt_dir}")
    rename_files_in_folder(gt_dir, "norain")
    
    print("\n重命名完成！")

def rename_files_in_folder(folder_path, prefix):
    # 获取文件夹中所有.png文件
    files = sorted(f for f in folder_path.iterdir() if f.is_file() and f.suffix.lower() == ".png")
    
    if not files:
        print(f"  警告: 没有找到PNG文件")
        return
    
    # 遍历并重命名每个文件
    renamed_count = 0
    for file_path in files:
        # 从文件名中提取数字部分
        filename = file_path.stem
        if filename.isdigit():
            number = int(filename)
            # 创建新文件名
            new_name = f"{prefix}-{number}.png"
            new_path = file_path.parent / new_name
            
            # 重命名文件
            try:
                file_path.rename(new_path)
                print(f"  重命名: {file_path.name} -> {new_name}")
                renamed_count += 1
            except Exception as e:
                print(f"  错误重命名 {file_path}: {str(e)}")
    
    print(f"  成功重命名了 {renamed_count} 个文件")

if __name__ == "__main__":
    # 设置基础路径
    base_path = "/mnt/netdisk/liumh/workspace/PromptIR/data/Train/Derain"
    
    # 执行重命名操作
    rename_images(base_path)