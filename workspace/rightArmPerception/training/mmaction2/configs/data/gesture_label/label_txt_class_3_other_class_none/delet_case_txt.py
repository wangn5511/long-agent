# 有两个txt文件tx1和tx2，每个文件都有三列，第一列是文件名，第二列是标签，第三列是视频路径
# 现在要删除tx1中在tx2中没出现的文件名

import os

def remove_common_filenames(file1_path, file2_path, output_path=None):
    """
    从file1中删除那些在file2中没出现的文件名
    
    Args:
        file1_path: tx1文件路径
        file2_path: tx2文件路径
        output_path: 输出文件路径，如果为None则在file1文件名后添加'_filtered'后缀
    """
    # 如果未指定输出路径，创建默认输出路径
    if output_path is None:
        file1_dir, file1_name = os.path.split(file1_path)
        name_without_ext, ext = os.path.splitext(file1_name)
        output_path = os.path.join(file1_dir, f"{name_without_ext}_filtered{ext}")
    
    # 读取tx2中的文件名，存储在集合中以便快速查找
    tx2_filenames = set()
    try:
        with open(file2_path, 'r', encoding='utf-8') as f2:
            for line in f2:
                # 跳过空行
                if line.strip():
                    tx2_filenames.add(line.strip())
                    
        print(f"从 {file2_path} 中读取了 {len(tx2_filenames)} 个文件名")
    except Exception as e:
        print(f"读取 {file2_path} 时出错: {e}")
        return
    
    # 处理tx1文件，过滤掉在tx2中存在的文件名
    filtered_lines = []
    removed_count = 0
    total_count = 0
    
    try:
        with open(file1_path, 'r', encoding='utf-8') as f1:
            for line in f1:
                total_count += 1
                if line.strip() not in tx2_filenames:
                    removed_count += 1
                else:
                    filtered_lines.append(line)
        
        print(f"处理 {file1_path}: 总共 {total_count} 行，移除 {removed_count} 行")
    except Exception as e:
        print(f"读取 {file1_path} 时出错: {e}")
        return
    
    # 将过滤后的内容写入输出文件
    try:
        with open(output_path, 'w', encoding='utf-8') as f_out:
            f_out.writelines(filtered_lines)
        print(f"过滤后的结果已保存到: {output_path}")
    except Exception as e:
        print(f"写入 {output_path} 时出错: {e}")

if __name__ == "__main__":
    # 这里可以修改为您实际的文件路径
    tx1_path = "/home/fanchen/hmimodels-main/label_txt_class_3_other_class_none/class_11_第一批V1手势-更新-rewrite_balanced_test.txt"
    tx2_path = "/home/fanchen/hmimodels-main/label_txt_class_3_other_class_none/class_11_第一批V1手势-更新.txt"
    
    # 检查文件是否存在
    if not os.path.exists(tx1_path):
        print(f"错误: 文件 {tx1_path} 不存在")
    elif not os.path.exists(tx2_path):
        print(f"错误: 文件 {tx2_path} 不存在")
    else:
        # 执行过滤操作
        remove_common_filenames(tx1_path, tx2_path)