"""
定义一般模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# import sys
# sys.path.append("../")
import config
from utils import utils


# ======================================for baseline======================================

class BaseLine(nn.Module):

    def __init__(self, check_point=config.check_point, pool_way=config.pool_way):
        super(BaseLine, self).__init__()
        self.bert = utils.load_model(check_point=check_point)
        self.pool_way = pool_way

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            return_dict=True, output_hidden_states=True)

        return utils.out_based_pool_way(outputs, self.pool_way)
        # return outputs


# =====================================for simcse==========================================

class SimCSE(nn.Module):

    def __init__(self, check_point=config.check_point,
                 pool_way=config.pool_way,
                 attention_dropout=config.attention_dropout,
                 hidden_dropout=config.hidden_dropout):
        super(SimCSE, self).__init__()
        self.bert = utils.load_pretrained_model(check_point, attention_dropout, hidden_dropout)
        self.pool_way = pool_way

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            return_dict=True, output_hidden_states=True)

        return utils.out_based_pool_way(outputs, self.pool_way)


# ====================================for prompt======================================


class PromptBERT(nn.Module):
    def __init__(self, encoder, mask_id, denoise=True):
        super().__init__()
        self.encoder = encoder
        self.mask_id = mask_id
        self.denoise = denoise

    def forward(self, prompt, template):

        prompt_mask_embedding = self.calculate_mask_embedding(prompt)
        template_mask_embedding = self.calculate_mask_embedding(template)

        # 模板降噪方式，有没有其他的方法？
        sentence_embedding = prompt_mask_embedding - template_mask_embedding

        if self.denoise:
            return sentence_embedding
        else:
            return prompt_mask_embedding

    def calculate_mask_embedding(self, inputs):
        output = self.encoder(*inputs)
        input_ids = inputs[0]
        token_embeddings = output.last_hidden_state
        mask_index = (input_ids == self.mask_id).long()
        mask_embedding = utils.get_mask_embedding(token_embeddings, mask_index)
        return mask_embedding


# =======================================for mo prompt bert===========================

class RawBert(nn.Module):

    def __init__(self, bert, pool_way):
        super(RawBert, self).__init__()
        self.bert = bert
        self.pool_way = pool_way

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            return_dict=True, output_hidden_states=True)

        return utils.out_based_pool_way(outputs, self.pool_way)
