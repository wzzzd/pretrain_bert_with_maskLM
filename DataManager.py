

import os
import random
import math
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from datasets import Dataset, load_dataset, load_metric
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding, BertTokenizer
# from Config import Config


def get_dataset(config):
    """
    获取数据集
    """
    # 获取数据
    # raw_datasets = load_dataset("wikitext", "wikitext-2-v1")
    # train_text = raw_datasets['train']['text']
    # test_text = raw_datasets['test']['text']
    train_text = open_file(config.path_datasets + 'train.txt')
    test_text = open_file(config.path_datasets + 'test.txt')
    
    train_datasets = pd.DataFrame({'src':train_text, 'labels':train_text})
    test_datasets = pd.DataFrame({'src':test_text, 'labels':test_text})
    
    raw_datasets_train = Dataset.from_pandas(train_datasets)
    raw_datasets_test = Dataset.from_pandas(test_datasets)
    
    # tokenizer.
    # train set
    tokenizer = AutoTokenizer.from_pretrained(config.initial_pretrain_tokenizer)        # 读取tokenizer分词模型
    tokenized_datasets = raw_datasets_train.map(lambda x: tokenize_function(x, tokenizer, config), batched=True)        # 对于样本中每条数据进行数据转换
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)                        # 对数据进行padding
    tokenized_datasets = tokenized_datasets.remove_columns(["src"])                     # 移除不需要的字段
    tokenized_datasets.set_format("torch")                                              # 格式转换
    
    # test set
    tokenized_datasets_test = raw_datasets_test.map(lambda x: tokenize_function(x, tokenizer, config), batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    tokenized_datasets_test = tokenized_datasets_test.remove_columns(["src"])
    tokenized_datasets_test.set_format("torch")
    
    # 转换成DataLoader类
    train_dataloader = DataLoader(tokenized_datasets, shuffle=True, batch_size=config.batch_size, collate_fn=data_collator)
    eval_dataloader = DataLoader(tokenized_datasets_test, batch_size=config.batch_size, collate_fn=data_collator)
    
    return train_dataloader, eval_dataloader


def tokenize_function(example, tokenizer, config):
    """
    数据转换
    """
    # 分词
    token = tokenizer(example["src"], truncation=True, max_length=config.sen_max_length, padding=config.padding)
    token.data['labels'] = token.data['input_ids']
    # 获取特殊字符ids
    token_mask = tokenizer.mask_token
    token_pad = tokenizer.pad_token
    token_cls = tokenizer.cls_token
    token_sep = tokenizer.sep_token
    ids_mask = tokenizer.convert_tokens_to_ids(token_mask)
    token_ex = [token_mask, token_pad, token_cls, token_sep]
    ids_ex = [tokenizer.convert_tokens_to_ids(x) for x in token_ex]
    # 获取vocab dict
    vocab = tokenizer.vocab
    # mask机制
    mask_token = [[op_mask(x, ids_mask, ids_ex, vocab) for i,x in enumerate(line)] for line in token.data['input_ids']]
    # mask_token = [[ids_mask if len(line) > 5 and random.random()<=0.15 and i not in [0, len(line)-1] else x for i,x in enumerate(line)] for line in token.data['input_ids']]
    token.data['input_ids'] = mask_token
    return token


def op_mask(token, ids_mask, ids_ex, vocab):
    """
    Bert的原始mask机制。
        （1）85%的概率，保留原词不变
        （2）15%的概率，使用以下方式替换
                80%的概率，使用字符'[MASK]'，替换当前token。
                10%的概率，使用词表随机抽取的token，替换当前token。
                10%的概率，保留原词不变。
    """
    # 若在额外字符里，则跳过
    if token in ids_ex:
        return token
    # 采样替换
    if random.random()<=0.15:
        x = random.random()
        if x <= 0.80:
            token = ids_mask
        if x> 0.80 and x <= 0.9:
            # 随机生成整数
            token = random.randint(0, len(vocab)-1)
    return token



def open_file(path):
    """读文件"""
    text = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip()
            text.append(line)
    return text


