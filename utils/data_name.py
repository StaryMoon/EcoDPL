import os

def write_all_paths(base_dir, output_file):
    """
    遍历 base_dir 下的所有文件，将相对路径写入 output_file，每行一个。
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for root, dirs, files in os.walk(base_dir):
            for name in files:
                full_path = os.path.join(root, name)
                # 计算相对于 base_dir 的相对路径
                rel_path = os.path.relpath(full_path, base_dir)
                f.write(rel_path + "\n")

if __name__ == "__main__":
    # 直接在这里指定要遍历的文件夹路径和输出文件名
    base_dir = "data/Train/Derain"      # 例："./my_folder"
    output_file = "data_dir/rainy/rainTrain.txt"  # 例："paths.txt"

    write_all_paths(base_dir, output_file)
    print(f"已将 “{base_dir}” 中所有文件的相对路径写入 “{output_file}”")
