


def add_file_byby(file1_path, file2_path, file3_path):
    # 读取file1中以8结尾的行
    with open(file1_path, 'r') as file1:
        lines1 = file1.readlines()

    # 筛选出file1中以8结尾的行
    lines1_ending_with_8 = []
    for line in lines1:
        line_stripped = line.strip()
        if line_stripped and line_stripped.split()[-1] == '8':
            # 将最后的8变为0，别用replace，防止误替换
            parts = line_stripped.split()
            parts[-1] = '0'
            line_stripped = ' '.join(parts)
            lines1_ending_with_8.append(line_stripped + '\n')

    # 读取file2中的所有行
    with open(file2_path, 'r') as file2:
        lines2 = file2.readlines()

    # 将筛选出的file1行和file2的所有行合并并写入file3
    with open(file3_path, 'w') as file3:
        file3.writelines(lines1_ending_with_8 + lines2)

    print(f"已将file1中 {len(lines1_ending_with_8)} 行以8结尾的数据和file2中 {len(lines2)} 行数据合并到file3")


if __name__ == "__main__":
    file1_path = "/home/fanchen/hmimodels-main/label_txt_class_11/batch1-video_labels_test.txt"
    file2_path = "/home/fanchen/hmimodels-main/label_txt_class_2/batch1-video_labels_test_wave_out_2_none.txt"
    file3_path = "/home/fanchen/hmimodels-main/label_txt_class_2/batch1-video_labels_test_wave_out_2_none_add_byby.txt"

    add_file_byby(file1_path, file2_path, file3_path)




