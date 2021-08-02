
import os
import random
import math
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from datasets import Dataset, load_dataset, load_metric
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding, BertTokenizer, AutoModelForMaskedLM    #, BertForMaskedLM
from accelerate import Accelerator
from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler
from model.BertForMaskedLM import BertForMaskedLM
from Config import Config
from DataManager import get_dataset





def train():
    """
        预训练模型
    """
    print('model from pretrained')
    
    # 初始化配置
    config = Config()
    os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices

    # 获取数据
    train_dataloader, eval_dataloader = get_dataset(config)

    # 初始化模型和优化器
    model = BertForMaskedLM.from_pretrained(config.initial_pretrain_model)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)

    # 开启Acclerator加速模块
    accelerator = Accelerator(split_batches=True)
    # accelerator.device='cuda:1'
    print(accelerator.state)
    train_dl, eval_dl, model, optimizer = accelerator.prepare(
        train_dataloader, 
        eval_dataloader, 
        model, 
        optimizer
    )

    # 定义优化器配置
    num_training_steps = config.num_epochs * len(train_dl)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=config.num_warmup_steps,
        num_training_steps=num_training_steps
    )

    print('start to train')
    model.train()
    progress_bar = tqdm(range(num_training_steps))
    for epoch in range(config.num_epochs):
        print('epoch: {}'.format(epoch))
        for i, batch in enumerate(train_dl):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            
            if i % 500 == 0:
                print('epoch:{0}  iter:{1}/{2}  loss:{3}'.format(epoch, i, len(train_dl), loss))
        print('evaluation')
        
        # 模型保存
        print('saving model')
        path = config.path_model_save + '/epoch_{}/'.format(epoch)
        if not os.path.exists(path):
            os.mkdir(path)
        accelerator.wait_for_everyone()                     # 将在不同GPU中的模型进行整合
        unwrapped_model = accelerator.unwrap_model(model)   # 获取模型
        accelerator.save(unwrapped_model.state_dict(), path+'pytorch_model.bin')
        # unwrapped_model.save_pretrained(config.path_model_save)
        
        # evaluate
        eval(eval_dl, model, epoch, accelerator, config.batch_size)



def eval(eval_dataloader, model, epoch, accelerator, batch_size=32):
    # metric= load_metric("glue", "mrpc")
    # loss_total = 0
    losses = []
    model.eval()
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        # loss_total += loss.item()
        losses.append(accelerator.gather(loss.repeat(batch_size)))
    # 计算困惑度
    losses = torch.cat(losses)
    losses_avg = torch.mean(losses)
    perplexity = math.exp(losses_avg)
    print('eval {0}: loss:{1}  perplexity:{2}'.format(epoch, losses_avg.item(), perplexity))



def load_lm():
    print('model from pretrained')
    path = './checkpoints/'
    model = BertForMaskedLM.from_pretrained(path)
    print(1)
    
    path = './checkpoint/epoch_0/pytorch_model.bin'
    accelerator = Accelerator(split_batches=True)
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.load_state_dict(torch.load(path))
    print(1)



if __name__ == '__main__':
    train()
    # load_lm()
