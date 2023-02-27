
import os
import time
import random
import logging
import math
import numpy as np
import pandas as pd
import torch
from torch.nn import CrossEntropyLoss

from tqdm.auto import tqdm
from datasets import Dataset, load_dataset, load_metric
from torch.utils.data import DataLoader
from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler
from transformers import BertModel, BertConfig
from model.BertForMaskedLM import BertForMaskedLM
from Config import Config
from transformers import DistilBertForMaskedLM
from transformers import AutoTokenizer, DataCollatorWithPadding, BertTokenizer, DistilBertTokenizer


class Trainer(object):
    
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.initial_pretrain_tokenizer) 

    def train(self, train_loader, valid_loader):
        """
            预训练模型
        """
        print('training start')
        # 初始化配置
        os.environ["CUDA_VISIBLE_DEVICES"] = self.config.cuda_visible_devices
        device = torch.device(self.config.device)

        # # 多卡通讯配置
        # if torch.cuda.device_count() > 1:
        #     torch.distributed.init_process_group(backend='nccl', 
        #                                          init_method=self.config.init_method, 
        #                                          rank=0, 
        #                                          world_size=self.config.world_size)
        #     torch.distributed.barrier()

        # 初始化模型和优化器
        print('model loading')
        model = BertForMaskedLM.from_pretrained(self.config.initial_pretrain_model)
        # model = DistilBertForMaskedLM.from_pretrained(self.config.initial_pretrain_model)
        print(">>>>>>>> Model Structure >>>>>>>>")
        for name,parameters in model.named_parameters():
            print(name,':',parameters.size())
        print(">>>>>>>> Model Structure >>>>>>>>\n")
        
        optimizer = AdamW(model.parameters(), lr=self.config.learning_rate)

        # 定义优化器配置
        num_training_steps = self.config.num_epochs * len(train_loader)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=self.config.num_warmup_steps,
            num_training_steps=num_training_steps
        )

        # 分布式训练
        model.to(device)
        if torch.cuda.device_count() > 1:
            model = torch.nn.parallel.DistributedDataParallel(model, 
                                                        find_unused_parameters=True)
                                                        #broadcast_buffers=True)
        print('start to train')
        model.train()
        # loss_func = CrossEntropyLoss()
        progress_bar = tqdm(range(num_training_steps))
        loss_best = math.inf
        for epoch in range(self.config.num_epochs):
            for i, batch in enumerate(train_loader):
                batch = {k:v.to(device) for k,v in batch.items()}
                outputs = model(**batch)
                # 计算loss
                loss = outputs.loss
                loss = loss.mean()                
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

                if i % 500 == 0:
                    print('epoch:{0}  iter:{1}/{2}  loss:{3}'.format(epoch, i, len(train_loader), loss.item()))
            # 模型保存
            self.eval(valid_loader, model, epoch, device)
            model_save = model.module if torch.cuda.device_count() > 1 else model
            path = self.config.path_model_save + 'epoch_{}/'.format(epoch)
            model_save.save_pretrained(path)


    def eval(self, eval_dataloader, model, epoch, device):
        losses = []
        model.eval()
        
        input = []
        label = []
        pred = []
        for batch in eval_dataloader:
            batch = {k:v.to(device) for k,v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            loss = loss.unsqueeze(0)
            losses.append(loss)
            
            # 还原成token string   
            tmp_src = batch['input_ids'].cpu().numpy()
            tmp_label = batch['labels'].cpu().numpy()
            tmp_pred = torch.max(outputs.logits, -1)[1].cpu().numpy()
            for i in range(len(tmp_label)):
                line_l = tmp_label[i]
                line_l_split = [ x for x in line_l if x not in [0]]
                line_s = tmp_src[i]
                line_s_split = line_s[:len(line_l_split)]
                line_p = tmp_pred[i]
                line_p_split = line_p[:len(line_l_split)]
                tmp_s = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(line_s_split))
                tmp_lab = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(line_l_split))
                tmp_p = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(line_p_split))
                input.append(tmp_s)
                label.append(tmp_lab)
                pred.append(tmp_p)
        # 计算困惑度
        losses = torch.cat(losses)
        losses_avg = torch.mean(losses)
        perplexity = math.exp(losses_avg)
        print('eval {0}: loss:{1}  perplexity:{2}'.format(epoch, losses_avg.item(), perplexity))
        for i in range(10):
            print('-'*30)
            print('input: {}'.format(input[i]))
            print('label: {}'.format(label[i]))
            print('pred : {}'.format(pred[i]))
        
        return losses_avg



if __name__ == '__main__':
    
    config = Config()
    train(config)
    # load_lm()
