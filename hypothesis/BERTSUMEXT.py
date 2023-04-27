import torch
import torch.nn as nn
from pytorch_transformers import BertModel, BertConfig
from torch.nn.init import xavier_uniform_

from hypothesis.TransformerEncoder import ExtTransformerEncoder


class Classifier(nn.Module):
    def __init__(self, hidden_size):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask_cls):
        h = self.linear1(x).squeeze(-1)
        sent_scores = self.sigmoid(h) * mask_cls.float()
        return sent_scores


class Bert(nn.Module):
    def __init__(self, large, temp_dir, finetune=False):
        super(Bert, self).__init__()
        if (large):
            self.model = BertModel.from_pretrained('bert-large-uncased', cache_dir=temp_dir)
        else:
            self.model = BertModel.from_pretrained('bert-base-uncased', cache_dir=temp_dir)

        self.finetune = finetune

    def forward(self, x, segs, mask):
        if (self.finetune):
            top_vec, _ = self.model(x, segs, attention_mask=mask)
        else:
            self.eval()
            with torch.no_grad():
                top_vec, _ = self.model(x, segs, attention_mask=mask)
        return top_vec


class ExtSummarizer(nn.Module):
    def __init__(self, classifier_type):
        super(ExtSummarizer, self).__init__()

        if 'baseline'.lower() in classifier_type.lower():  # not pre-trained BERT and Linear head
            bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=768, num_hidden_layers=2,
                                     num_attention_heads=8, intermediate_size=2048)
            self.bert.model = BertModel(bert_config)
            self.ext_layer = Classifier(self.bert.model.config.hidden_size)
        elif 'linear'.lower() in classifier_type.lower():  # pre-trained BERT and Linear head
            self.bert = Bert(False, "./temp/BERT", True)
            self.ext_layer = Classifier(self.bert.model.config.hidden_size)
        else:  # pre-trained BERT and Transformer head
            self.bert = Bert(False, "./temp/BERT", True)
            self.ext_layer = ExtTransformerEncoder(self.bert.model.config.hidden_size,  2048,  8, 0.2,  2)

        for p in self.ext_layer.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

        # self.to(device)

    def forward(self, src, segs, clss, mask_src, mask_cls):
        top_vec = self.bert(src, segs, mask_src)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        sent_scores = self.ext_layer(sents_vec, mask_cls).squeeze(-1)
        return sent_scores, mask_cls

