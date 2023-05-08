import os


def _partition_govreport_single_file(type,raw_data_dir_path,save_data_dir_path):
    # 读取输入文件
    src_file = os.path.join(raw_data_dir_path, type + ".source")  # 输入的源文件名
    tgt_file = os.path.join(raw_data_dir_path, type + ".target")  # 输入的源文件名
    with open(src_file, "r", encoding="utf-8") as src, open(tgt_file, "r", encoding="utf-8") as tgt:
        src_lines = src.readlines()
        tgt_lines = tgt.readlines()
    # src_lines = src_lines[:16]
    # tgt_lines = tgt_lines[:16]
    # 计算需要生成的输出文件数量
    num_lines = len(src_lines)

    # 检查并创建保存文件的文件夹
    file_type = type
    if not os.path.exists(save_data_dir_path):
        os.mkdir(save_data_dir_path)
    save_type_data_dir = os.path.join(save_data_dir_path,type)
    if not os.path.exists(save_type_data_dir):
        os.mkdir(save_type_data_dir)

    # 生成输出文件
    for i in range(num_lines):
        # 打开输出文件
        out_file_name = os.path.join(save_type_data_dir, file_type + f"{i + 1}.story")
        print(out_file_name)
        with open(out_file_name, "w", encoding="utf-8") as out_file:
            out_file.write(src_lines[i])
            # 目标文件行
            out_file.write("\n@highlight\n" + tgt_lines[i])


def govreport_raw_to_partition(param_dict):
    raw_data_dir_path = param_dict["raw_path"]
    save_data_dir_path = param_dict["partition_path"]
    type_li = ["train", "val", "test"]
    for type in type_li:
        _partition_govreport_single_file(type,raw_data_dir_path,save_data_dir_path)

if __name__ == '__main__':
    param_dict = {}
    param_dict["raw_path"] = r"E:\Lab\nlp_dataset\gov-report\raw_data"
    param_dict["partition_path"] = r"E:\Lab\nlp_dataset\gov-report\partition_data"
    govreport_raw_to_partition(param_dict)