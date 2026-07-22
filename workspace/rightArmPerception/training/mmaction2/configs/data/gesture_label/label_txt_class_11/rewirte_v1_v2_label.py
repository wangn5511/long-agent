
txt_path = f"/home/fanchen/hmimodels-main/label_txt_class_11/gesture_v2_train.txt"
output_txt = f"{txt_path.split('.')[0]}_rewrite.txt"


with open(txt_path, mode='r', encoding='utf-8') as txtfile:
    lines = txtfile.readlines()
    with open(output_txt, mode='w', encoding='utf-8') as out_txtfile:
        for line in lines:
            line = line.strip().split(" ")   # file_name offset length label
            
            # offset + 1； length - 1，其他的维持不变
            line[1] = str(int(line[1]) + 1)

            #  - 1之后为负数则报错
            if int(line[2]) - 1 < 0:
                print(f"length 为 {line[2]}，- 1 后为负数，跳过 {line[0]}")
                continue
            line[2] = str(int(line[2]) - 1)
            
            # 写入新的行
            out_txtfile.write(f"{line[0]} {line[1]} {line[2]} {line[3]}\n")

            
            

            

            


