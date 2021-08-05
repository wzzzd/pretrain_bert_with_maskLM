
import os
import random




class Config(object):
    
    def __init__(self):
        
        # 可以使用的GPU
        self.cuda_visible_devices = '1,2'                           # 可见的GPU
        self.device = 'cuda:0'                                      # master GPU
        self.mode = 'train'
        self.port = str(random.randint(10000,60000))                # 多卡训练进程间通讯端口
        self.init_method = 'tcp://localhost:' + self.port           # 多卡训练的通讯地址
        self.world_size = 1

        self.num_epochs = 100                                        # 迭代次数
        self.batch_size = 128                                       # 每个批次的大小
        self.learning_rate = 3e-4                                   # 学习率
        self.num_warmup_steps = 0.1                                 # warm up步数
        self.sen_max_length = 128                                   # 句子最长长度
        self.padding = True                                         # 是否对输入进行padding

        self.initial_pretrain_model = 'bert-base-uncased'           # 加载的预训练分词器checkpoint，默认为英文。若要选择中文，替换成 bert-base-chinese
        self.initial_pretrain_tokenizer = 'bert-base-uncased'       # 加载的预训练模型checkpoint，默认为英文。若要选择中文，替换成 bert-base-chinese
        self.path_model_save = './checkpoint/'                      # 模型保存路径
        self.path_datasets = './datasets/shopline-all/'             # 数据集
        self.path_log = './logs/'

