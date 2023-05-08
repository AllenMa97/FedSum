import gc
import glob
import hashlib
import itertools
import json
import os
import random
import re
import subprocess
from collections import Counter
from os.path import join as pjoin
import platform
import torch
from multiprocess import Pool

from data_preprocess.others.logging import logger
from data_preprocess.others.tokenization import BertTokenizer
from pytorch_transformers import XLNetTokenizer

from data_preprocess.others.utils import clean
from data_preprocess.prepro.utils import _get_word_ngrams

import xml.etree.ElementTree as ET

nyt_remove_words = ["photo", "graph", "chart", "map", "table", "drawing"]


def recover_from_corenlp(s):
    s = re.sub(r' \'{\w}', '\'\g<1>', s)
    s = re.sub(r'\'\' {\w}', '\'\'\g<1>', s)


def load_json(p, lower):
    source = []
    tgt = []
    flag = False
    for sent in json.load(open(p, encoding='utf-8'))['sentences']:
        tokens = [t['word'] for t in sent['tokens']]
        # print("tokens")
        # print(tokens)
        # print(tokens[0])
        if (lower):
            tokens = [t.lower() for t in tokens]
        # print(flag)
        if (tokens[0] == '@highlight'):
            flag = True
            tgt.append([])
            continue
        # print(flag)
        if (flag):
            tgt[-1].extend(tokens)
        else:
            # print(tokens)
            source.append(tokens)

    source = [clean(' '.join(sent)).split() for sent in source]
    tgt = [clean(' '.join(sent)).split() for sent in tgt]
    return source, tgt


def load_xml(p):
    tree = ET.parse(p)
    root = tree.getroot()
    title, byline, abs, paras = [], [], [], []
    title_node = list(root.iter('hedline'))
    if (len(title_node) > 0):
        try:
            title = [p.text.lower().split() for p in list(title_node[0].iter('hl1'))][0]
        except:
            print(p)

    else:
        return None, None
    byline_node = list(root.iter('byline'))
    byline_node = [n for n in byline_node if n.attrib['class'] == 'normalized_byline']
    if (len(byline_node) > 0):
        byline = byline_node[0].text.lower().split()
    abs_node = list(root.iter('abstract'))
    if (len(abs_node) > 0):
        try:
            abs = [p.text.lower().split() for p in list(abs_node[0].iter('p'))][0]
        except:
            print(p)

    else:
        return None, None
    abs = ' '.join(abs).split(';')
    abs[-1] = abs[-1].replace('(m)', '')
    abs[-1] = abs[-1].replace('(s)', '')

    for ww in nyt_remove_words:
        abs[-1] = abs[-1].replace('(' + ww + ')', '')
    abs = [p.split() for p in abs]
    abs = [p for p in abs if len(p) > 2]

    for doc_node in root.iter('block'):
        att = doc_node.get('class')
        # if(att == 'abstract'):
        #     abs = [p.text for p in list(f.iter('p'))]
        if (att == 'full_text'):
            paras = [p.text.lower().split() for p in list(doc_node.iter('p'))]
            break
    if (len(paras) > 0):
        if (len(byline) > 0):
            paras = [title + ['[unused3]'] + byline + ['[unused4]']] + paras
        else:
            paras = [title + ['[unused3]']] + paras

        return paras, abs
    else:
        return None, None


def tokenize(param_dict, type):
    if not os.path.exists(param_dict["tokenized_path"]):
        os.mkdir(param_dict["tokenized_path"])
    if type!=None:
        stories_dir = os.path.join(os.path.abspath(param_dict["partition_path"]), type)
        tokenized_stories_dir = os.path.join(os.path.abspath(param_dict["tokenized_path"]), type)
        # tokenized_stories_dir = os.path.abspath(param_dict["tokenized_path"])
        if not os.path.exists(tokenized_stories_dir):
            os.mkdir(tokenized_stories_dir)
    else:
        stories_dir = os.path.abspath(param_dict["partition_path"])
        tokenized_stories_dir = os.path.abspath(param_dict["tokenized_path"])

    print("Preparing to tokenize %s to %s..." % (stories_dir, tokenized_stories_dir))
    stories = os.listdir(stories_dir)
    # make IO list file
    print("Making list of files to tokenize...")
    with open("mapping_for_corenlp.txt", "w") as f:
        for s in stories:
            if (not s.endswith('story')):
                continue
            f.write("%s\n" % (os.path.join(stories_dir, s)))
    os.environ['CLASSPATH'] = param_dict["stanford_class_path"]
    command = ['java', 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize,ssplit',
               '-ssplit.newlineIsSentenceBreak', 'always', '-filelist', 'mapping_for_corenlp.txt', '-outputFormat',
               'json', '-outputDirectory', tokenized_stories_dir]
    print("Tokenizing %i files in %s and saving in %s..." % (len(stories), stories_dir, tokenized_stories_dir))
    subprocess.call(command)
    print("Stanford CoreNLP Tokenizer has finished.")
    os.remove("mapping_for_corenlp.txt")

    # Check that the tokenized stories directory contains the same number of files as the original directory
    num_orig = len(os.listdir(stories_dir))
    num_tokenized = len(os.listdir(tokenized_stories_dir))
    if num_orig != num_tokenized:
        raise Exception(
            "The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
                tokenized_stories_dir, num_tokenized, stories_dir, num_orig))
    print("Successfully finished tokenizing %s to %s.\n" % (stories_dir, tokenized_stories_dir))


def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)


def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode('utf-8'))
    return h.hexdigest()


class BertData():
    def __init__(self, param_dict):
        self.param_dict = param_dict
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.tgt_bos = '[unused0]'
        self.tgt_eos = '[unused1]'
        self.tgt_sent_split = '[unused2]'
        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]

    def preprocess(self, src, tgt, sent_labels, use_bert_basic_tokenizer=False, is_test=False):

        if ((not is_test) and len(src) == 0):
            return None

        original_src_txt = [' '.join(s) for s in src]

        idxs = [i for i, s in enumerate(src) if (len(s) > self.param_dict["min_src_ntokens_per_sent"])]

        _sent_labels = [0] * len(src)
        for l in sent_labels:
            _sent_labels[l] = 1

        src = [src[i][:self.param_dict["max_src_ntokens_per_sent"]] for i in idxs]
        sent_labels = [_sent_labels[i] for i in idxs]
        src = src[:self.param_dict["max_src_nsents"]]
        sent_labels = sent_labels[:self.param_dict["max_src_nsents"]]

        if ((not is_test) and len(src) < self.param_dict["min_src_nsents"]):
            return None

        src_txt = [' '.join(sent) for sent in src]
        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)

        src_subtokens = self.tokenizer.tokenize(text)

        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        sent_labels = sent_labels[:len(cls_ids)]

        tgt_subtokens_str = '[unused0] ' + ' [unused2] '.join(
            [' '.join(self.tokenizer.tokenize(' '.join(tt), use_bert_basic_tokenizer=use_bert_basic_tokenizer)) for tt
             in tgt]) + ' [unused1]'
        tgt_subtoken = tgt_subtokens_str.split()[:self.param_dict["max_tgt_ntokens"]]
        if ((not is_test) and len(tgt_subtoken) < self.param_dict["min_tgt_ntokens"]):
            return None

        tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt_subtoken)

        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
        src_txt = [original_src_txt[i] for i in idxs]

        return src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt


def format_to_bert(param_dict):
    if not os.path.exists(param_dict["bert_path"]):
        os.mkdir(param_dict["bert_path"])
    if (param_dict["dataset"] != ''):
        datasets = [param_dict["dataset"]]
    else:
        datasets = ['train', 'val', 'test']
    for corpus_type in datasets:
        a_lst = []
        for json_f in glob.glob(pjoin(param_dict["json_path"], '*' + corpus_type + '.*.json')):
            os_name = platform.system()
            if os_name == "Windows":
                real_name = json_f.split('\\')[-1] # windows
            elif os_name == "Linux":
                real_name = json_f.split('/')[-1]  # linux
            a_lst.append(
                (corpus_type, json_f, param_dict, pjoin(param_dict["bert_path"], real_name.replace('json', 'bert.pt'))))
        pool = Pool(param_dict["n_cpus"])
        for d in pool.imap(_format_to_bert, a_lst):
            pass

        pool.close()
        pool.join()
    
    print("successfully transfer tokenized data to bert data")


def _format_to_bert(params):
    corpus_type, json_file, param_dict, save_file = params
    is_test = corpus_type == 'test'
    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        return

    bert = BertData(param_dict)

    logger.info('Processing %s' % json_file)
    jobs = json.load(open(json_file))
    datasets = []
    for d in jobs:
        source, tgt = d['src'], d['tgt']

        sent_labels = greedy_selection(source[:param_dict["max_src_nsents"]], tgt, 3)
        if (param_dict["lower"]):
            source = [' '.join(s).lower().split() for s in source]
            tgt = [' '.join(s).lower().split() for s in tgt]
        b_data = bert.preprocess(source, tgt, sent_labels,
                                 use_bert_basic_tokenizer=param_dict["use_bert_basic_tokenizer"],
                                 is_test=is_test)
        # b_data = bert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=param_dict["use_bert_basic_tokenizer"])

        if (b_data is None):
            continue
        src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data
        b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                       "src_sent_labels": sent_labels, "segs": segments_ids, 'clss': cls_ids,
                       'src_txt': src_txt, "tgt_txt": tgt_txt}
        datasets.append(b_data_dict)
    logger.info('Processed instances %d' % len(datasets))
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()


def format_to_lines(param_dict):
    type_li = ['train', 'val', 'test']
    train_files, val_files, test_files = [], [], []
    for type in type_li:
        for f in glob.glob(pjoin(param_dict["tokenized_path"],type, '*.json')):
            if type == "train":
                train_files.append(f)
            elif type =="val":
                val_files.append(f)
            elif type=="test":
                test_files.append(f)

    if not os.path.exists(param_dict["json_path"]):
        os.mkdir(param_dict["json_path"])
    corpora = {'train': train_files, 'val': val_files, 'test': test_files}
    # print(corpora)
    for corpus_type in ['train', 'val', 'test']:
        a_lst = [(f, param_dict) for f in corpora[corpus_type]]
        pool = Pool(param_dict["n_cpus"])
        dataset = []
        p_ct = 0
        for d in pool.imap_unordered(_format_to_lines, a_lst):
            dataset.append(d)
            if (len(dataset) > param_dict["shard_size"]):
                pt_file = "{:s}.{:d}.json".format(corpus_type, p_ct)
                pt_file = os.path.join(param_dict["json_path"], pt_file)
                with open(pt_file, 'w', encoding='utf-8') as save:
                    save.write(json.dumps(dataset))
                    p_ct += 1
                    dataset = []

        pool.close()
        pool.join()

        if (len(dataset) > 0):
            pt_file = "{:s}.{:d}.json".format(corpus_type, p_ct)
            # pt_file = "{:s}.{:s}.{:d}.json".format(param_dict["json_path"], corpus_type, p_ct)
            pt_file = os.path.join(param_dict["json_path"],pt_file)
            with open(pt_file, 'w', encoding='utf-8') as save:
                # save.write('\n'.join(dataset))
                save.write(json.dumps(dataset))
                p_ct += 1
                dataset = []
    print("successfully transfer tokenized data to json data")

def format_wikihow_to_lines(param_dict):
    corpus_mapping = {}
    for corpus_type in ['val', 'test', 'train']:
        temp = []
        for line in open(pjoin(param_dict["map_path"], 'mapping_' + corpus_type + '.txt'), encoding='utf-8'):

            os_name = platform.system()
            if os_name=="Windows":
                temp.append(line.strip().splitlines()[0])  # windows
            elif os_name=="Linux":
                temp.append(line.strip())  # linux
        corpus_mapping[corpus_type] = {key.strip(): 1 for key in temp}

    train_files, valid_files, test_files = [], [], []
    for f in glob.glob(pjoin(param_dict["tokenized_path"], '*.json')):

        os_name = platform.system()
        if os_name == "Windows":
            real_name = f.split('\\')[-1].split('.')[0]  # windows
        elif os_name == "Linux":
            real_name = f.split('/')[-1].split('.')[0] # linux

        if (real_name in corpus_mapping['val']):
            valid_files.append(f)
        elif (real_name in corpus_mapping['test']):
            test_files.append(f)
        elif (real_name in corpus_mapping['train']):
            train_files.append(f)

    if not os.path.exists(param_dict["json_path"]):
        os.mkdir(param_dict["json_path"])
    corpora = {'train': train_files, 'val': valid_files, 'test': test_files}
    '''
     'valid': ['E:\\Lab\\nlp_dataset\\wikihow\\data_process\\tokenized_data\\Howto1v1SomeoneinCallofDuty1.story.json',...]
    '''
    # print(corpora)
    for corpus_type in ['train', 'val', 'test']:
        a_lst = [(f, param_dict) for f in corpora[corpus_type]]
        pool = Pool(param_dict["n_cpus"])
        dataset = []
        p_ct = 0
        for d in pool.imap_unordered(_format_to_lines, a_lst):
            dataset.append(d)
            if (len(dataset) > param_dict["shard_size"]):
                pt_file = "{:s}.{:d}.json".format(corpus_type, p_ct)
                pt_file = os.path.join(param_dict["json_path"], pt_file)
                with open(pt_file, 'w', encoding='utf-8') as save:
                    save.write(json.dumps(dataset))
                    p_ct += 1
                    dataset = []

        pool.close()
        pool.join()

        if (len(dataset) > 0):
            pt_file = "{:s}.{:d}.json".format(corpus_type, p_ct)
            # pt_file = "{:s}.{:s}.{:d}.json".format(param_dict["json_path"], corpus_type, p_ct)
            pt_file = os.path.join(param_dict["json_path"], pt_file)
            with open(pt_file, 'w', encoding='utf-8') as save:
                # save.write('\n'.join(dataset))
                save.write(json.dumps(dataset))
                p_ct += 1
                dataset = []
    print("successfully transfer tokenized data to json data")

def _format_to_lines(params):
    f, param_dict = params
    source, tgt = load_json(f, param_dict["lower"])

    return {'src': source, 'tgt': tgt}

