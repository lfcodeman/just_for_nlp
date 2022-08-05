# 系统配置
class SystemConfig:
    def __init__(self):
        self.mode = ''
        self.dataset_fold = ''
        self.train_file = ''
        self.dev_file = ''
        self.use_bert = True

        # bert
        self.hg_bert_name = 'bert_base_chinese'
        # 模型保存路径
        self.model_cache_dir = './model'
        # 是否使用bilstm
        self.use_bilstm = True

        # 微调
        self.finetune = False
        self.vocabs_dir = ''

        # 日志
        self.log_dir = ''
        # 模型运行过程的文件路径
        self.checkpoints_dir = ''
        self.checkpoints_name = ''

        # label设置

        # model的配置

        # 模型训练的配置
        self.epoch = 10
        self.batch_size = 32
        self.dropout = 0.5
        self.learning_rate = 0.001

        # 优化器
        self.optimizer = 'Adam'
        



