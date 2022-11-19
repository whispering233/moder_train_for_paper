"""
提供模型数据集类
"""
from abc import ABC

from torch.utils.data import Dataset

from utils import utils
import config


class CommonDataset(Dataset, ABC):
    def __init__(self, data: list, tokenizer, max_len=config.max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {
            'target': utils.tokenizer_fun(self.data[index][0], self.tokenizer, max_length=self.max_len),
            'source': utils.tokenizer_fun(self.data[index][1], self.tokenizer, max_length=self.max_len),
            'label': int(self.data[index][2])
        }


# ====================================for simcse====================================


class SimCSEDataset(Dataset):
    def __init__(self, data: list, tokenizer, max_len=config.max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    # simcse 是通过drop out 产生正例对的，两次传入文本是一致的
    # 都是无监督的方式，只是评测指标不一样
    def __getitem__(self, index):
        return utils.tokenizer_fun([self.data[index][0], self.data[index][0]], self.tokenizer)


# =======================================for prompt====================================


class PromptDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        line = self.data[index]

        # [sen, template, raw]
        # sen & template & raw: list [word, word......]
        sen_a = line[0]
        sen_a = [utils.tokenizer_fun(text, tokenizer=self.tokenizer, is_split_into_words=True) for text in sen_a]
        sen_b = line[1]
        sen_b = [utils.tokenizer_fun(text, tokenizer=self.tokenizer, is_split_into_words=True) for text in sen_b]

        label = line[2]

        return sen_a, sen_b, label

