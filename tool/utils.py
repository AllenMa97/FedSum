#!/usr/bin/python
# -*- coding: utf-8 -*-
import re
import re
import os
import shutil
import copy
import datetime
import numpy as np
import torch
import scipy
import datetime
import sys
import time
import pyrouge
import argparse

from typing import List
from collections import OrderedDict
from tool.logger import *

sys.setrecursionlimit(10000)


def get_specific_time():
    now = time.localtime()
    year, month, day = str(now.tm_year), str(now.tm_mon), str(now.tm_mday)
    hour, minute, second = str(now.tm_hour), str(now.tm_min), str(now.tm_sec)
    return str(year + "_" + month + "_" + day + "_" + hour + "h" + minute + "m" + second + "s")


REMAP = {"-lrb-": "(", "-rrb-": ")", "-lcb-": "{", "-rcb-": "}",
         "-lsb-": "[", "-rsb-": "]", "``": '"', "''": '"'}


def clean(x):
    x = x.lower()
    return re.sub(
        r"-lrb-|-rrb-|-lcb-|-rcb-|-lsb-|-rsb-|``|''",
        lambda m: REMAP.get(m.group()), x)


def check_and_make_the_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


# compute the cos similarity between a and b. a, b are numpy arrays
def cos_sim(self, a, b):
    return 1 - scipy.spatial.distance.cosine(a, b)


def eval_label(match_true, pred, true, total, match):
    match_true, pred, true, match = match_true.float(), pred.float(), true.float(), match.float()
    try:
        print("match_true:", match_true.data, " ;pred:", pred.data, " ;true:", true.data, " ;match:", match.data,
              " ;total:", total)
        accu = match / total
        precision = match_true / pred
        recall = match_true / true
        F = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        accu, precision, recall, F = 0.0, 0.0, 0.0, 0.0
        logger.error("[Error] float division by zero")
    return accu, precision, recall, F


def normalization(x):
    """"
    归一化到区间{0,1]
    返回副本
    """
    _range = np.max(x) - np.min(x)
    return (x - np.min(x)) / _range


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
    return net


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def test_rouge(temp_dir, cand, ref):
    candidates = [line.strip() for line in open(cand, encoding='utf-8', errors='ignore')]
    references = [line.strip() for line in open(ref, encoding='utf-8', errors='ignore')]
    print(len(candidates))
    print(len(references))
    assert len(candidates) == len(references)

    cnt = len(candidates)
    tmp_dir = os.path.join(temp_dir, "rouge-tmp")
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
        os.mkdir(tmp_dir + "/candidate")
        os.mkdir(tmp_dir + "/reference")
    try:

        for i in range(cnt):
            if len(references[i]) < 1:
                continue
            with open(tmp_dir + "/candidate/cand.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(candidates[i])
            with open(tmp_dir + "/reference/ref.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(references[i])
        rouge_dir = "./tool/ROUGE-1.5.5"
        r = pyrouge.Rouge155(rouge_dir=rouge_dir)
        r.model_dir = tmp_dir + "/reference/"
        r.system_dir = tmp_dir + "/candidate/"
        r.model_filename_pattern = 'ref.#ID#.txt'
        r.system_filename_pattern = r'cand.(\d+).txt'
        rouge_results = r.convert_and_evaluate()
        print(rouge_results)
        results_dict = r.output_to_dict(rouge_results)
    finally:
        pass
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
    return results_dict


def rouge_results_to_str(results_dict):
    return ">> ROUGE-F(1/2/3/l): {:.2f}/{:.2f}/{:.2f}\nROUGE-R(1/2/3/l): {:.2f}/{:.2f}/{:.2f}\n".format(
        results_dict["rouge_1_f_score"] * 100,
        results_dict["rouge_2_f_score"] * 100,
        # results_dict["rouge_3_f_score"] * 100,
        results_dict["rouge_l_f_score"] * 100,
        results_dict["rouge_1_recall"] * 100,
        results_dict["rouge_2_recall"] * 100,
        # results_dict["rouge_3_f_score"] * 100,
        results_dict["rouge_l_recall"] * 100

        # ,results_dict["rouge_su*_f_score"] * 100
    )