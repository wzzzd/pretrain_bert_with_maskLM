
import os
import time
import random
import logging
import math
import numpy as np
import pandas as pd
import torch

from tqdm.auto import tqdm
from datasets import Dataset, load_dataset, load_metric
from torch.utils.data import DataLoader
from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler
from transformers import BertModel, BertConfig
from model.BertForMaskedLM import BertForMaskedLM
from Config import Config
from DataManager import get_dataset




def train(config):
    """
        预训练模型
    """
    print('training start')
    # 初始化配置
    os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices
    device = torch.device(config.device)

    # 多卡通讯配置
    if config.mode == 'train' and torch.cuda.device_count() > 1:
        torch.distributed.init_process_group(backend='nccl', 
                                             init_method=config.init_method, 
                                             rank=0, 
                                             world_size=config.world_size)
        torch.distributed.barrier()

    # 获取数据
    print('data loading')
    train_dl, eval_dl = get_dataset(config)

    # 初始化模型和优化器
    print('model loading')
    model = BertForMaskedLM.from_pretrained(config.initial_pretrain_model)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)

    # 定义优化器配置
    num_training_steps = config.num_epochs * len(train_dl)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=config.num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # 分布式训练
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, 
                                                    find_unused_parameters=True,
                                                    broadcast_buffers=True)
    print('start to train')
    model.train()
    progress_bar = tqdm(range(num_training_steps))
    loss_best = math.inf
    for epoch in range(config.num_epochs):
        for i, batch in enumerate(train_dl):
            batch.data = {k:v.to(device) for k,v in batch.data.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            
            if i % 500 == 0:
                print('epoch:{0}  iter:{1}/{2}  loss:{3}'.format(epoch, i, len(train_dl), loss.item()))
        # 模型保存
        current_loss = eval(eval_dl, model, epoch)
        # current_loss = loss.item()
        if current_loss < loss_best:
            loss_best = current_loss
            print('saving model')
            path = config.path_model_save + '/epoch_{}/'.format(epoch)
            if not os.path.exists(path):
                os.mkdir(path)
            model_save = model.module if torch.cuda.device_count() > 1 else model
            model_save.save_pretrained(path)


def eval(eval_dataloader, model, epoch):
    losses = []
    model.eval()
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        losses.append(loss)
    # 计算困惑度
    losses = torch.cat(losses)
    losses_avg = torch.mean(losses)
    perplexity = math.exp(losses_avg)
    print('eval {0}: loss:{1}  perplexity:{2}'.format(epoch, losses_avg.item(), perplexity))
    return losses_avg
    
    
def load_lm():
    print('model from pretrained')
    path = './checkpoint/epoch_0/'
    model = BertModel.from_pretrained(path)
    # torch.save(model.state_dict(), model_cp,_use_new_zipfile_serialization=False)
    return model


if __name__ == '__main__':
    
    config = Config()
    train(config)
    # load_lm()
