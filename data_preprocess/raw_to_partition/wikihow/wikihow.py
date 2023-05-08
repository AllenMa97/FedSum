import os
import argparse
import random

### 1. 把后缀名更改为story
def change_suffix(folder_path):
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.story'):
            continue  # 不需要修改后缀名的文件
        old_path = os.path.join(folder_path, file_name)
        new_path = os.path.join(folder_path, file_name.split('.')[0] + '.story')
        os.rename(old_path, new_path)

### 2. 所有文件内操作
def file_operation(folder_path,encodings):
    cnt = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 获取文件的绝对路径
            file_path = os.path.join(root, file)
            for encoding in encodings:
                try:
                    ### 2.1把所有的 @ summary改为 @ highlight
                    with open(file_path, "r", encoding=encoding) as f:
                        data = f.read()
                        data = data.replace("@summary", "@highlight")
                    with open(file_path, "w", encoding=encoding) as f:
                        f.write(data)

                    ### 2.2 把所有的 @article删除
                    with open(file_path, 'r',encoding=encoding) as file:
                        lines = file.readlines()
                    with open(file_path, 'w',encoding=encoding) as file:
                        for line in lines:
                            if "@article" not in line:
                                file.write(line)

                    ### 2.3 把所有含有@highlight的行及其后一行放到文章的最后
                    # 打开文件并读取内容
                    with open(file_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()

                    # 找到所有包含@highlight的行及其后一行，并将其存储到highlights列表中
                    highlights = []
                    i = 0
                    while i < len(lines):
                        if "@highlight" in lines[i]:
                            highlights.append(lines[i].strip())
                            highlights.append(lines[i + 1].strip())
                            highlights.append("\n")
                            # 删除该行及其后一行
                            del lines[i:i + 2]
                        else:
                            i += 1
                    # 将修改后的内容写入原文件
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.writelines(lines)
                    # 将highlights中的内容加入到文件末尾
                    with open(file_path, "a", encoding="utf-8") as f:
                        f.write("\n".join(highlights))

                    ### 2.4 删除所有开头空行并将连续多行空行变为一行空行
                    # 打开文件并读取内容
                    with open(file_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                    # 删除开头的空行
                    while lines and not lines[0].strip():
                        del lines[0]
                    # 将连续多行空行变为一行空行
                    i = 0
                    while i < len(lines):
                        if lines[i].strip() == "":
                            # 找到连续的多行空行
                            j = i + 1
                            while j < len(lines) and lines[j].strip() == "":
                                j += 1
                            # 将连续多行空行缩减为一行空行
                            del lines[i + 1:j]
                            lines[i] = "\n"
                        i += 1
                    # 将修改后的内容写回到文件中
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.writelines(lines)

                    break  # 如果成功打开文件，则跳出循环
                except Exception as e:
                    print(f"Error occurred when using {encoding}: {e}")
                    continue  # 如果出现异常，则继续尝试下一个编码格式

            else:
                print("All encoding formats failed, unable to read the file.")
                print(file)
                os.remove(file_path)  # 如果所有的编码格式都无法成功打开文件，则打印错误信息并删除文件
                cnt += 1
    print("打开失败的文件数目:"+str(cnt))

### 3. 划分train，test，valid的mapping文件
def partition_dataset(folder_path,map_path):
    ## 获取文件夹中的文件名列表
    file_names = os.listdir(folder_path)
    num_files = len(file_names)

    ## 根据比例计算各部分文件数量
    # 定义划分比例
    train_ratio = 0.6
    val_ratio = 0.2
    test_ratio = 0.2
    num_train = int(num_files * train_ratio)
    num_val = int(num_files * val_ratio)
    num_test = num_files - num_train - num_val

    ## 随机打乱文件名列表
    random.shuffle(file_names)

    ## 将文件名写入对应的 txt 文件中
    # partition_dir = os.path.join(folder_path,"mapping")
    partition_dir = map_path
    if not os.path.exists(partition_dir):
        os.mkdir(partition_dir)
    mapping_train = os.path.join(partition_dir,"mapping_train.txt")
    mapping_valid = os.path.join(partition_dir,"mapping_val.txt")
    mapping_test = os.path.join(partition_dir,"mapping_test.txt")
    with open(mapping_train, "w") as f_train, \
            open(mapping_valid, "w") as f_val, \
            open(mapping_test, "w") as f_test:
        for i, file_name in enumerate(file_names):
            if i < num_train:
                f_train.write(file_name.split('.')[0] + "\n")
            elif i < num_train + num_val:
                f_val.write(file_name.split('.')[0] + "\n")
            else:
                f_test.write(file_name.split('.')[0] + "\n")

def wikihow_raw_to_partition(param_dict):
    raw_path = param_dict["raw_path"]
    partition_path = param_dict["partition_path"]
    if not os.path.exists(partition_path):
        os.mkdir(partition_path)

    import shutil
    for filename in os.listdir(raw_path): # 如果文件符合某种条件，则将其从src移动到dst
        shutil.copy(os.path.join(raw_path,filename) ,os.path.join(partition_path,filename))

    map_path = param_dict["map_path"]
    encodings = ["utf-8","gbk", 'Windows-1252', 'TIS-620', 'Windows-1254', 'ISO-8859-1', 'ascii', 'EUC-KR']  # 定义编码格式列表
    change_suffix(partition_path)
    file_operation(partition_path, encodings)
    partition_dataset(partition_path, map_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-folder_path", default=r'C:\Users\LZP\Desktop\nlp数据集\wikihow\WikiHow-Dataset-master\articles', type=str)
    parser.add_argument("-map_path", default=r'C:\Users\LZP\Desktop\nlp数据集\wikihow\WikiHow-Dataset-master\articles', type=str)
    args = parser.parse_args()
    # 放所有文件所有可能的编码方式
    # encodings = ["utf-8", 'Windows-1252', 'TIS-620', 'Windows-1254', 'ISO-8859-1', 'ascii', 'EUC-KR']  # 定义编码格式列表
    # change_suffix(args.folder_path)
    # file_operation(args.folder_path,encodings)
    # partition_dataset(args.folder_path,args.map_path)
    wikihow_raw_to_partition(args.folder_path,args.map_path)