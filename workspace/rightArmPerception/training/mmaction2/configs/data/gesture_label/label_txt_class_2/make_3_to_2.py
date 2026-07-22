def process_file(input_file, output_file):
    """
    将输入文件中最后一列元素为2的行，将其最后一列改为0，然后保存到输出文件
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    processed_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        parts = line.split()
        if len(parts) > 0:
            last_element = parts[-1]
            if last_element == '2':
                # 将最后一列从2改为0
                parts[-1] = '0'
            
            # 重建行并添加到结果列表
            processed_line = ' '.join(parts) + '\n'
            processed_lines.append(processed_line)
    
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(processed_lines)
    
    print(f"处理完成: 从 {input_file} 读取 {len(lines)} 行，写入 {output_file} {len(processed_lines)} 行")

if __name__ == '__main__':
    input_file = ['/home/fanchen/hmimodels-main/label_txt_class_3/第一批V1手势-更新-video_labels_rewrite_balanced_train_filtered_clip_None.txt',
                  '/home/fanchen/hmimodels-main/label_txt_class_3/第一批V1手势-更新-video_labels_rewrite_balanced_test_filtered_clip_None.txt',
                  '/home/fanchen/hmimodels-main/label_txt_class_3/第一批V2手势-更新-video_labels_rewrite_balanced_test_filtered_clip_None.txt',
                  '/home/fanchen/hmimodels-main/label_txt_class_3/第一批V2手势-更新-video_labels_rewrite_balanced_train_filtered_clip_None.txt']
    for i_input_file in input_file:
        output_file = i_input_file.replace('class_3', 'class_2').replace('.txt', '_from_3_to_2.txt')
        process_file(i_input_file, output_file)

