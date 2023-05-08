import os
import bisect
import torch
import glob
import numpy as np
from tool.logger import *
from torch.utils.data import Dataset
from data_preprocess.preprocess_dataset \
    import process_GovernmentReport,process_Wikihow,process_PubMed

# import sys
# # 将指定文件夹添加到Python的搜索路径中
# sys.path.insert(0, r"E:\Lab\论文代码\open_source_version_code")

class BERTSUMDataset(Dataset):
    def _pad(self, data, pad_id, width=-1):
        if (width == -1):
            width = max(len(d) for d in data)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data

    def __init__(self, attribute_dict):
        pre_src = attribute_dict["src"]
        pre_tgt = attribute_dict["tgt"]
        pre_segs = attribute_dict["segs"]
        pre_clss = attribute_dict["clss"]
        pre_src_sent_labels = attribute_dict["src_sent_labels"]

        self.src = torch.tensor(self._pad(pre_src, 0))
        self.tgt = torch.tensor(self._pad(pre_tgt, 0))
        self.segs = torch.tensor(self._pad(pre_segs, 0))
        self.mask_src = ~(self.src == 0)  # mask_src = 1 - (src == 0)
        self.mask_tgt = ~(self.tgt == 0)  # mask_tgt = 1 - (tgt == 0)

        self.clss = torch.tensor(self._pad(pre_clss, -1))
        self.src_sent_labels = torch.tensor(self._pad(pre_src_sent_labels, 0))
        self.mask_cls = ~(self.clss == -1)  # mask_cls = 1 - (clss == -1)
        self.clss[self.clss == -1] = 0

        self.src_txt = attribute_dict["src_txt"]
        self.tgt_txt = attribute_dict["tgt_txt"]

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return {
            "src": self.src[idx],
            "tgt": self.tgt[idx],
            "src_sent_labels": self.src_sent_labels[idx],
            "segs": self.segs[idx],
            "clss": self.clss[idx],
            "mask_src": self.mask_src[idx],
            "mask_tgt": self.mask_tgt[idx],
            "mask_cls": self.mask_cls[idx],
            "src_txt": self.src_txt[idx],
            "tgt_txt": self.tgt_txt[idx],
        }


def get_CNNDM_dataset(data_path, corpus_type, only_one=True):
    def preprocess(data_batch):
        max_tgt_len = 140
        max_pos = 512

        src_list = []
        tgt_list = []
        src_sent_labels_list = []
        segs_list = []
        clss_list = []
        src_txt_list = []
        tgt_txt_list = []
        for i in range(len(data_batch["src"])):
            src = data_batch["src"][i]
            tgt = data_batch["tgt"][i][:max_tgt_len][:-1] + [2]
            src_sent_labels = data_batch["src_sent_labels"][i]
            segs = data_batch["segs"][i]
            clss = data_batch["clss"][i]
            src_txt = data_batch["src_txt"][i]
            tgt_txt = data_batch["tgt_txt"][i]

            end_id = [src[-1]]
            src = src[:-1][:max_pos - 1] + end_id
            segs = segs[:max_pos]
            max_sent_id = bisect.bisect_left(clss, max_pos)
            src_sent_labels = src_sent_labels[:max_sent_id]
            clss = clss[:max_sent_id]
            # src_txt = src_txt[:max_sent_id]

            src_list.append(src)
            tgt_list.append(tgt)
            src_sent_labels_list.append(src_sent_labels)
            segs_list.append(segs)
            clss_list.append(clss)
            src_txt_list.append(src_txt)
            tgt_txt_list.append(tgt_txt)

        return {
            'src': src_list,
            'tgt': tgt_list,
            'src_sent_labels': src_sent_labels_list,
            "segs": segs_list,
            "clss": clss_list,
            "src_txt": src_txt_list,
            "tgt_txt": tgt_txt_list,
        }

    pts = sorted(glob.glob(data_path + '/' + corpus_type + '.[0-9]*.pt'))

    if only_one:
        pts = pts[:1]

    full_dataset = []
    for idx, pt in enumerate(pts):
        pieces = torch.load(pt)
        logger.info('Loading %s dataset from %s, number of examples: %d' % (corpus_type, pt, len(pieces)))
        full_dataset += pieces

    full_dataset = full_dataset[:8]

    attribute_dict = preprocess(
        {
            'src': [dic["src"] for dic in full_dataset],
            'tgt': [dic["tgt"] for dic in full_dataset],
            'src_sent_labels': [dic["src_sent_labels"] for dic in full_dataset],
            "segs": [dic["segs"] for dic in full_dataset],
            "clss": [dic["clss"] for dic in full_dataset],
            "src_txt": [dic["src_txt"] for dic in full_dataset],
            "tgt_txt": [dic["tgt_txt"] for dic in full_dataset],
        })
    return BERTSUMDataset(attribute_dict=attribute_dict)



def get_WikiHow_dataset(data_path, corpus_type, only_one=True):
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    if len(glob.glob(data_path + '/' + corpus_type + '.[0-9]*.pt')) == 0:
        process_Wikihow()

    return get_CNNDM_dataset(data_path, corpus_type, only_one=True)


def get_GovernmentReport_dataset(data_path, corpus_type, only_one=True):
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    if "val" in corpus_type:
        corpus_type = "val"

    if len(glob.glob(data_path + '/' + corpus_type + '.[0-9]*.pt')) == 0:
        process_GovernmentReport()

    return get_CNNDM_dataset(data_path, corpus_type, only_one=True)


def get_PubMed_dataset(data_path, corpus_type, only_one=True):
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    if "val" in corpus_type:
        corpus_type = "val"

    if len(glob.glob(data_path + '/' + corpus_type + '.[0-9]*.pt')) == 0:
        process_PubMed()

    return get_CNNDM_dataset(data_path, corpus_type, only_one=True)


if __name__ == '__main__':
    print("Testing")
    # data_path = '../dataset/DRUG/'
    # mask_s1_flag = False
    # mask_s2_flag = False
    # mask_s1_s2_flag = False
    # qq, bb = get_DRUG_dataset(data_path, mask_s1_flag, mask_s2_flag, mask_s1_s2_flag)

    # training_dataset, testing_dataset = get_COMPAS_dataset("../dataset/COMPAS")

    # aa = get_CNNDM_dataset("../dataset/GovernmentReport", corpus_type="train")
    aa = get_GovernmentReport_dataset("../dataset/test_gov",corpus_type="train")
    print(aa)
    print(1)
