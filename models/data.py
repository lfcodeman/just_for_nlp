import os


# 数据处理相关
class DataManager:
    def __init__(self, configs, logger):
        self.configs = configs
        self.train_file = ''

        pass

    def load_vocab(self):
        if not os.path.isfile(self.configs.token2id_file):
            return self.build_vocab(self.train_file)
        pass

    # 词典可以使用bert中的，也可以自己生成：
    # 如果要fine_tuning的话，则需要生成vocab加到bert的词典中
    # 如果不要fine_tuning的话，则直接使用bert的词表
    def build_vocab(self, train_path):
        if self.configs.fine_tuning:
            # 根据训练集做词表的生成：则从tain.csv文件中读取词和label，分别转为对应的id
            pass
        else:
            pass
        pass

    def padding(self, sample):

        pass

    def prepare(self, token, labels, is_padding=True):
        pass

    def prepare_bert_embedding(self, df):
        pass

    def get_training_set(self, train_val_ratio=0.9):
        pass

    def get_valid_set(self):
        pass

    def map_func(self, x, token2id):
        # token2id中是否有unkown,有的话，x应该如何标记，没有的话如何标记
        pass

    # 将单个句子转为字向量
    def prepare_single_sentence(self, sentence):
        pass
