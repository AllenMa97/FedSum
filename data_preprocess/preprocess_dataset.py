from data_preprocess.prepro.data_builder import tokenize
from data_preprocess.prepro.data_builder import format_to_lines, format_wikihow_to_lines
from data_preprocess.prepro.data_builder import format_to_bert
from data_preprocess.raw_to_partition.govreport.govreport import govreport_raw_to_partition
from data_preprocess.raw_to_partition.wikihow.wikihow import wikihow_raw_to_partition
from data_preprocess.raw_to_partition.PubMed.pubmed import pubmed_raw_to_partition
import json
import os


# 把之前命令行的设置args参数改为在每个函数调用前设置路径
def process_GovernmentReport():
    type_li = ["train", "val", "test"]
    dir_path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(dir_path, "param_dict.json"), "r") as f:
        param_dict = json.load(f)
    # print(param_dict)

    ## raw_data to one_article_per_file_data
    govreport_raw_to_partition(param_dict)

    ## one_article_per_file_data to tokenized_data
    for type in type_li:
        tokenize(param_dict, type)

    # tokenized_data to json_data
    format_to_lines(param_dict)

    # json_data to bert_data
    format_to_bert(param_dict)


def process_Wikihow():
    # type_li = ["train","val","test"]
    dir_path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(dir_path, "param_dict.json"), "r") as f:
        param_dict = json.load(f)
    # print(param_dict)

    ## raw_data to one_article_per_file_data
    wikihow_raw_to_partition(param_dict)

    ## one_article_per_file_data to tokenized_data
    # for type in type_li:
    tokenize(param_dict, None)

    # tokenized_data to json_data
    format_wikihow_to_lines(param_dict)

    # json_data to bert_data
    format_to_bert(param_dict)


def process_PubMed():
    type_li = ["train", "val", "test"]
    dir_path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(dir_path, "param_dict.json"), "r") as f:
        param_dict = json.load(f)
    # print(param_dict)

    ## raw_data to one_article_per_file_data
    for type in type_li:
        pubmed_raw_to_partition(param_dict,type)

    ## one_article_per_file_data to tokenized_data
    for type in type_li:
        tokenize(param_dict, type)

    # tokenized_data to json_data
    format_to_lines(param_dict)

    # json_data to bert_data
    format_to_bert(param_dict)


if __name__ == '__main__':
    # process_GovernmentReport()
    # process_Wikihow()
    pass
