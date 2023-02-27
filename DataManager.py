

import os
import copy
import random
import math
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from datasets import Dataset, load_dataset, load_metric
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding, BertTokenizer, DistilBertTokenizer

from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler



class DataManager(object):
    
    def __init__(self, config):
        self.config = config
        self.init_gpu_config()
    

    def init_gpu_config(self):
        """
        初始化GPU并行配置
        """
        print('loading GPU config ...')
        if self.config.mode == 'train' and torch.cuda.device_count() > 1:
            torch.distributed.init_process_group(backend='nccl', 
                                                 init_method=self.config.init_method,
                                                 rank=0, 
                                                 world_size=self.config.world_size)
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    
    
    def get_dataset(self, mode='train', sampler=True):
        """
        获取数据集
        """
        # 读取tokenizer分词模型
        tokenizer = AutoTokenizer.from_pretrained(self.config.initial_pretrain_tokenizer)     
        
        if mode=='train':
            train_dataloader = self.data_process('train.txt', tokenizer)
            return train_dataloader
        elif mode=='dev':
            eval_dataloader = self.data_process('dev.txt', tokenizer)
            return eval_dataloader
        else:
            test_dataloader = self.data_process('test.txt', tokenizer, sampler=sampler)
            return test_dataloader
        


    def data_process(self, file_name, tokenizer, sampler=True):
        """
        数据转换
        """
        # 获取数据
        text = self.open_file(self.config.path_datasets + file_name)#[:2000]
        dataset = pd.DataFrame({'src':text, 'labels':text})
        # dataframe to datasets
        raw_datasets = Dataset.from_pandas(dataset)
        # tokenizer.
        tokenized_datasets = raw_datasets.map(lambda x: self.tokenize_function(x, tokenizer), batched=True)        # 对于样本中每条数据进行数据转换
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)                        # 对数据进行padding
        tokenized_datasets = tokenized_datasets.remove_columns(["src"])                     # 移除不需要的字段
        tokenized_datasets.set_format("torch")                                              # 格式转换
        # 转换成DataLoader类
        # train_dataloader = DataLoader(tokenized_datasets, shuffle=True, batch_size=config.batch_size, collate_fn=data_collator)
        # eval_dataloader = DataLoader(tokenized_datasets_test, batch_size=config.batch_size, collate_fn=data_collator)
        # sampler = RandomSampler(tokenized_datasets) if not torch.cuda.device_count() > 1 else DistributedSampler(tokenized_datasets)
        # dataloader = DataLoader(tokenized_datasets, sampler=sampler, batch_size=self.config.batch_size, collate_fn=data_collator)
        if sampler:
            sampler = RandomSampler(tokenized_datasets) if not torch.cuda.device_count() > 1 else DistributedSampler(tokenized_datasets)
        else:
            sampler = None
        dataloader = DataLoader(tokenized_datasets, sampler=sampler, batch_size=self.config.batch_size)     #, collate_fn=data_collator , num_workers=2, drop_last=True
        return dataloader


    def tokenize_function(self, example, tokenizer):
        """
        数据转换
        """
        # 分词
        token = tokenizer(example["src"], truncation=True, max_length=self.config.sen_max_length, padding='max_length') #config.padding)
        label=copy.deepcopy(token.data['input_ids'])
        token.data['labels'] = label
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
        vocab_int2str = { v:k for k, v in vocab.items()}
        # mask机制
        if self.config.whole_words_mask:
            # whole words masking
            mask_token = [ self.op_mask_wwm(line, ids_mask, ids_ex, vocab_int2str) for line in token.data['input_ids']]
        else:
            mask_token = [[self.op_mask(x, ids_mask, ids_ex, vocab) for i,x in enumerate(line)] for line in token.data['input_ids']]
        # 验证mask后的字符长度
        mask_token_len = len(set([len(x) for x in mask_token]))
        assert mask_token_len==1, 'length of mask_token not equal.'
        flag_input_label = [1 if len(x)==len(y) else 0 for x,y in zip(mask_token, label)]
        assert sum(flag_input_label)==len(mask_token), 'the length between input and label not equal.'
        token.data['input_ids'] = mask_token
        return token



    # def op_mask_wwm(self, tokens, ids_mask, ids_ex, vocab_int2str):
    #     """
    #     基于全词mask
    #     """
    #     if len(tokens) <= 5:
    #         return tokens
    #     # string = [tokenizer.convert_ids_to_tokens(x) for x in tokens]
    #     line = tokens
    #     for i, token in enumerate(tokens):
    #         # 若在额外字符里，则跳过
    #         if token in ids_ex:
    #             line[i] = token
    #             continue
    #         # 采样替换
    #         if random.random()<=0.15:
    #             x = random.random()
    #             if x <= 0.80:
    #                 # 获取词string
    #                 token_str = vocab_int2str[token]
    #                 if '##' not in token_str:
    #                     # 若不含有子词标志
    #                     line[i] = ids_mask
    #                     # 后向寻找
    #                     curr_i = i + 1
    #                     flag = True
    #                     while flag:
    #                         # 判断当前词是否包含 ##
    #                         token_index = tokens[curr_i]
    #                         token_index_str = vocab_int2str[token_index]
    #                         if '##' not in token_index_str:
    #                             flag = False
    #                         else:
    #                             line[curr_i] = ids_mask
    #                         curr_i += 1
    #             if x> 0.80 and x <= 0.9:
    #                 # 随机生成整数
    #                 while True:
    #                     token = random.randint(0, len(vocab_int2str)-1)
    #                     # 不再特殊字符index里，则跳出
    #                     if token not in ids_ex:
    #                         break
    #     return line


    def op_mask(self, token, ids_mask, ids_ex, vocab):
        """
        Bert的原始mask机制。
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
                while True:
                    token = random.randint(0, len(vocab)-1)
                    # 不再特殊字符index里，则跳出
                    if token not in ids_ex:
                        break
                # token = random.randint(0, len(vocab)-1)
        return token
    
    
    def op_mask_wwm(self, tokens, ids_mask, ids_ex, vocab_int2str):
        """
        基于全词mask
        """
        if len(tokens) <= 5:
            return tokens
        # string = [tokenizer.convert_ids_to_tokens(x) for x in tokens]
        # line = tokens
        line = copy.deepcopy(tokens)
        for i, token in enumerate(tokens):
            # 若在额外字符里，则跳过
            if token in ids_ex:
                line[i] = token
                continue
            # 采样替换
            if random.random()<=0.10:
                x = random.random()
                if x <= 0.80:
                    # 获取词string
                    token_str = vocab_int2str[token]
                    # 若含有子词标志
                    if '##' in token_str:
                        line[i] = ids_mask
                        # 前向寻找
                        curr_i = i - 1
                        flag = True
                        while flag:
                            # 判断当前词是否包含 ##
                            token_index = tokens[curr_i]
                            token_index_str = vocab_int2str[token_index]
                            if '##' not in token_index_str:
                                flag = False
                            line[curr_i] = ids_mask
                            curr_i -= 1
                        # 后向寻找
                        curr_i = i + 1
                        flag = True
                        while flag:
                            # 判断当前词是否包含 ##
                            token_index = tokens[curr_i]
                            token_index_str = vocab_int2str[token_index]
                            if '##' not in token_index_str:
                                flag = False
                            else:
                                line[curr_i] = ids_mask
                            curr_i += 1
                    else:
                        # 若不含有子词标志
                        line[i] = ids_mask
                        # 后向寻找
                        curr_i = i + 1
                        flag = True
                        while flag:
                            # 判断当前词是否包含 ##
                            token_index = tokens[curr_i]
                            token_index_str = vocab_int2str[token_index]
                            if '##' not in token_index_str:
                                flag = False
                            else:
                                line[curr_i] = ids_mask
                            curr_i += 1
                if x> 0.80 and x <= 0.9:
                    # 随机生成整数
                    while True:
                        token = random.randint(0, len(vocab_int2str)-1)
                        # 不再特殊字符index里，则跳出
                        if token not in ids_ex:
                            break
        # # 查看mask效果：int转换成string
        # test = [vocab_int2str[x] for x in line]
        # test_gt = [vocab_int2str[x] for x in tokens]
        return line
    
    
    def open_file(self, path):
        """读文件"""
        text = []
        with open(path, 'r', encoding='utf8') as f:
            for line in f.readlines():#[:1000]:
                line = line.strip()
                text.append(line)
        return text
    
