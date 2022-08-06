import torch
import torch.nn as nn
from transformers import BertModel, BertConfig, BertTokenizer

class NerModel(nn.Module):
    def __init__(self, configs, vocab_size, num_classes):
        super(NerModel, self).__init__()
        self.use_bert = configs.use_bert
        self.fine_tuning = configs.fine_tuning
        if self.use_bert:
            self.bert_model = BertModel.from_pretrained('bert-base-chinese')
        self.use_bilstm = configs.use_bilstm
        self.embedding = nn.Embedding(vocab_size, configs.embedding_dim)
        self.hidden_dim = configs.hidden_dim
        self.dropout_rate = configs.dropout
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.bilstm = nn.LSTM(input_size=configs.input_size, hidden_dim=configs.hidden_dim)
        self.linear = nn.Linear(configs.hidden_dim, num_classes)
        self.transition_params = nn.Parameter(torch.randn(num_classes, num_classes))

    def forward(self, inputs, input_length, targets, training=None):
        if self.use_bert:
            if self.fine_tuning:
                embedding_inputs = self.bert_model(inputs[0], attention_mask=inputs[1])[0]
            else:
                embedding_inputs = inputs
        else:
            embedding_inputs = self.embedding(inputs)

        outputs = self.dropout(embedding_inputs, training)
        if self.use_bilstm:
            outputs = self.bilstm(outputs)
        logits = self.linear(outputs)

