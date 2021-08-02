
import os




class Config(object):
    
    def __init__(self):
        
        # 可以使用的GPU
        self.cuda_visible_devices = '0,1'                           # GPU
        

        self.num_epochs = 3                                         # 迭代次数
        self.batch_size = 32                                        # 每个批次的大小
        self.learning_rate = 3e-5                                   # 学习率
        self.num_warmup_steps = 0                                   # warm up步数
        self.sen_max_length = 128                                   # 句子最长长度
        self.padding = True                                         # 是否对输入进行padding

        self.initial_pretrain_model = 'bert-base-uncased'           # 加载的预训练分词器checkpoint，默认为英文。若要选择中文，替换成 bert-base-chinese
        self.initial_pretrain_tokenizer = 'bert-base-uncased'       # 加载的预训练模型checkpoint，默认为英文。若要选择中文，替换成 bert-base-chinese
        self.path_model_save = './checkpoint/'
        self.path_datasets = './datasets/'
  

