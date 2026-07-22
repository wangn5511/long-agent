# 读取文件并过滤掉最后一列是2的行


def filter_wave_out_data(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # 过滤掉最后一列是2的行
    filtered_lines = [line for line in lines if line.strip().split()[-1] != '2']

    # 写回文件
    with open(output_file, 'w') as file:
        file.writelines(filtered_lines)

def other_class_2_none(input_file, output_file):
    # 读取文件并修改最后一列的值
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # 修改每一行的最后一列：如果最后一列不是1，则改为0
    modified_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) > 0:
            last_element = parts[-1]
            if last_element != '1':
                parts[-1] = '0'
            modified_line = ' '.join(parts) + '\n'
            modified_lines.append(modified_line)
        else:
            modified_lines.append(line)

    # 写回文件
    with open(output_file, 'w') as file:
        file.writelines(modified_lines)


if __name__ == "__main__":
    input_file = ['/home/fanchen/hmimodels-main/label_txt_class_3/batch1-video_labels_train.txt',
                  '/home/fanchen/hmimodels-main/label_txt_class_3/batch2-video_labels_train.txt',
                  '/home/fanchen/hmimodels-main/label_txt_class_3/batch1-video_labels_test.txt',
                  '/home/fanchen/hmimodels-main/label_txt_class_3/batch2-video_labels_test.txt',
                  "/home/fanchen/hmimodels-main/label_txt_class_3/第一批V1手势-更新-video_labels_rewrite_balanced_train_clip_None.txt",
                  "/home/fanchen/hmimodels-main/label_txt_class_3/第一批V1手势-更新-video_labels_rewrite_balanced_test_clip_None.txt",
                  "/home/fanchen/hmimodels-main/label_txt_class_3/第一批V2手势-更新-video_labels_rewrite_balanced_train_clip_None.txt",
                  "/home/fanchen/hmimodels-main/label_txt_class_3/第一批V2手势-更新-video_labels_rewrite_balanced_test_clip_None.txt"
                  ]
    output_file = [i.replace("class_3", "class_2").replace(".txt", "_wave_out_2_none.txt") for i in input_file]
    for i_input_file, i_output_file in zip(input_file, output_file):
        # filter_wave_out_data(i_input_file, i_output_file)
        other_class_2_none(i_input_file, i_output_file)



# with open('/home/fanchen/hmimodels-main/label_txt_class_3/batch1-video_labels_train.txt', 'r') as file:
#     lines = file.readlines()

# # 过滤掉最后一列是2的行
# filtered_lines = [line for line in lines if line.strip().split()[-1] != '2']

# # 写回文件
# with open('/home/fanchen/hmimodels-main/label_txt_class_2/batch1-video_labels_train.txt', 'w') as file:
#     file.writelines(filtered_lines)