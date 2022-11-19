"""
提供模型训练的一般功能
"""

import os

import numpy as np
from transformers import BertConfig, BertModel, BertTokenizer, AutoModel, AutoTokenizer, AlbertModel, AlbertConfig
from tqdm import tqdm
import time
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import pickle

import sys

sys.path.append("../")
from src import config

# ==============================for huggingface ===========================================
''''''


# load config
def load_model_config(check_point):
    """
    :param check_point: 除了Albert模型的模型加载使用 AlbertConfig，其他模型用的都是BertConfig
    :return:
    """
    if "albert" in check_point:
        return AlbertConfig.from_pretrained(check_point)
    else:
        return BertConfig.from_pretrained(check_point)


# load model
def load_pretrained_model(check_point, attention_dropout, hidden_dropout):
    """
    :param check_point:
    :param attention_dropout:  drop out
    :param hidden_dropout:
    :return:
    """

    model_config = load_model_config(check_point)
    model_config.attention_probs_dropout_prob = attention_dropout
    model_config.hidden_dropout_prob = hidden_dropout

    if "albert" in check_point:
        return AlbertModel.from_pretrained(check_point, config=model_config)
    else:
        return BertModel.from_pretrained(check_point, config=model_config)


# original model
def load_model(check_point):
    if "albert" in check_point:
        return AlbertModel.from_pretrained(check_point)
    else:
        return BertModel.from_pretrained(check_point)


# 加载词表
def load_tokenizer(check_point):
    """
    :param check_point: 所有模型都是用 BertTokenizer加载的
    :return:
    """
    return BertTokenizer.from_pretrained(check_point)


def tokenizer_fun(text, tokenizer,
                  max_length=config.max_len,
                  truncation=config.truncation,
                  padding=config.padding,
                  return_tensors=config.return_tensors, **kwargs):
    return tokenizer(text,
                     max_length=max_length,
                     truncation=truncation,
                     padding=padding,
                     return_tensors=return_tensors, **kwargs)


# 加载 prompt 专用此表
def load_tokenizer_prompt(check_point, special_tokens_dict=config.special_tokens_dict, mask_token=config.mask_token):
    tokenizer = load_tokenizer(check_point=check_point)
    tokenizer.add_special_tokens(special_tokens_dict=special_tokens_dict)
    mask_id = tokenizer.convert_tokens_to_ids(mask_token)
    return tokenizer, mask_id


# ==================================for read standard data===============================
''''''


# 读取标准格式的数据集
def read_standard_data(file_path, total_lines=config.total_lines_for_try, shuffle=config.read_data_shuffle):
    """
    直接在读取的时候就只读部分数据，或者返回部分数据
    :param file_path:
    :param total_lines:
    :param shuffle: 读取已经存储的文件的时候是否对读取结果随机
    :return:
    """

    res = []
    with open(file_path, 'r', encoding='utf-8') as f:

        lines = f.readlines()

        if shuffle:
            random.shuffle(lines)

        for i, line in enumerate(lines):

            # line: str -> list
            line = line.strip().split("_||_")
            res.append(line)

            if i == total_lines - 1:
                break

    f.close()

    return res


# 从标准文件路径结构中读取结果并返回
def read_from_dict(files: dict):
    train_file = files["train"]
    dev_file = files["dev"]
    test_file = files["test"]

    if (not os.path.exists(train_file)) or (not os.path.exists(dev_file)) or (not os.path.exists(test_file)):
        raise ValueError("文件{}不存在".format(train_file))

    train_data = read_standard_data(train_file)
    dev_data = read_standard_data(dev_file)
    test_data = read_standard_data(test_file)

    return train_data, dev_data, test_data


# 从标准数据结构的数据中解出句子对
def detach_sen_pair_supervised(data: list[list]):
    a = []
    b = []
    label = []

    for line in data:
        if len(line) == 3:
            a.append([line[0]])
            b.append([line[1]])
            label.append([line[2]])

    return a, b, label


# 组合 template_words 和 sen_words
def combine_template_and_words(sen_words: list, template_words: list, replace_token):
    pos = template_words.index(replace_token)

    res = template_words[: pos] + sen_words + template_words[pos + 1:]

    return res


# 填充 template
def fill_template(template_words: list, replace_token, num):
    pos = template_words.index(replace_token)

    res = template_words[:pos] + [replace_token] * num + template_words[pos + 1:]

    return res


# 将一个句子构造成 prompt 输入形式
def build_prompt_sen(sen: str, template: str, tokenizer, max_len: int, replace_token) -> list:
    # 分词，句子里词的数目
    # 中文就是一个个字，但是在英文中就是一个个单词，直接截断有问题
    sen_words = tokenizer.tokenize(sen)

    # template 本身的长度
    # 注意不是字符长度，而是输入到 Bert 里的词长度
    template_words = tokenizer.tokenize(template.strip())
    template_len = len(template_words) - 1

    # 截断
    sen_words = sen_words[:max_len - template_len]

    # 截断后的词数
    sen_len = len(sen_words)

    # 重新组合
    # 只是在组合的时候需要考虑，以免输入到Bert被截断了影响 template 本身的结构
    # 但是原始的句子直接输入即可，Bert 会自动进行截断
    prompt_word = combine_template_and_words(sen_words=sen_words, template_words=template_words, replace_token=replace_token)
    template_words = fill_template(template_words=template_words, replace_token=replace_token, num=sen_len)

    return [prompt_word, template_words]


# 构建 prompt 输入
# 两组句子， 一个或两个模板
# 但是要一一对应
def build_prompt_inputs(data: list[list], template, tokenizer, momentum=False, max_len=config.max_len, replace_token=config.replace_token):
    """
    :param momentum:
    :param data: [[sen_a, sen_b, label], ....] or [[sen]]
    :param template:
    :param tokenizer:
    :param max_len:
    :param replace_token:
    :return: [[[prompt_a, template_a], [prompt_b, template_b], [label]], ...] or [[[prompt, template], [prompt, template], [None]], ...]
    """
    res = []
    pair = []

    if len(template) == 2:
        template_a = template[0]
        template_b = template[1]
    else:
        template_a = template[0]
        template_b = template[0]

    for line in data:

        if len(line) == 3:
            sen_a = line[0]
            sen_b = line[1]
            label = line[2]
        else:
            sen_a = line[0]
            sen_b = line[0]
            label = -1

        if momentum:

            # unified format: list of word
            pair.append([*build_prompt_sen(sen=sen_a, template=template_a, tokenizer=tokenizer, max_len=max_len, replace_token=replace_token),
                         tokenizer.tokenize(sen_a)])
            pair.append([*build_prompt_sen(sen=sen_b, template=template_b, tokenizer=tokenizer, max_len=max_len, replace_token=replace_token),
                         tokenizer.tokenize(sen_b)])
            pair.append([label])
        else:
            pair.append(build_prompt_sen(sen=sen_a, template=template_a, tokenizer=tokenizer, max_len=max_len, replace_token=replace_token))
            pair.append(build_prompt_sen(sen=sen_b, template=template_b, tokenizer=tokenizer, max_len=max_len, replace_token=replace_token))
            pair.append([label])

        res.append(pair.copy())
        pair = []

    return res


# 构建 MoPrompt输入
def build_moprompt_inputs(data: list[list], template, tokenizer, max_len=config.max_len, replace_token=config.replace_token):
    """
    :param data: [[sen_a, sen_b, label], ...] or [[sen], ...]
    :param template:
    :param tokenizer:
    :param max_len:
    :param replace_token:
    :return: [[[prompt_a, template_a, raw_a], [prompt_b, template_b, raw_b], [label]], ...]
    """
    pass


# ==================================for model forward=================================
''''''


# 根据传入文件名，判断是有监督数据还是无监督数据
def judge_supervised(file_path):
    return ("lcqmc" in file_path) or ("afqmc" in file_path) or ("sts" in file_path)


# 根据 pool_way 处理数据
def out_based_pool_way(input, pool_way=config.pool_way):
    """
    :param input: huggingface bert out
    :param pool_way: 池化方式 ["first_last_avg", "last_avg", "cls", "pooler"]
    :return: [batch_size, sen_len, hidden_state] -> [batch_size, hidden_state]
    """

    if input is None:
        raise ValueError("输入数据为空")

    if pool_way == "first_last_avg":
        out = (input.hidden_states[-1] + input.hidden_states[1]).mean(dim=1)
    elif pool_way == "last_avg":
        out = input.last_hidden_state.mean(dim=-1)
    elif pool_way == "cls":
        out = input.last_hidden_state[:, 0, :]
    elif pool_way == "pooler":
        out = input.pooler_output
    else:
        raise ValueError("池化方式未知{}".format(pool_way))

    return out


# 处理 bert 输入
def preprocess_bert_input(inputs):
    """
    不能在这直接将其输入到模型中 返回结果
    :param inputs:
    :return: dict -> tuple
    """

    inputs = {x: inputs[x].squeeze(1) for x in inputs}

    input_ids = inputs['input_ids'].to(config.device)
    attention_mask = inputs['attention_mask'].to(config.device)
    token_type_ids = inputs['token_type_ids'].to(config.device)

    return input_ids, attention_mask, token_type_ids


# ------------------------------------------------------------for whiten--------------------------------------------------------------------
''''''


# 一个句子转换为一个向量
def sen_to_vec(sen, model, tokenizer):
    with torch.no_grad():
        inputs = tokenizer_fun(sen, tokenizer)
        inputs = preprocess_bert_input(inputs)

        out_puts = model(*inputs)
        out_puts = out_puts.cpu().detach().numpy()

        return out_puts


# 句子转换为句向量
def sens_to_vecs(sens: list, model, tokenizer):
    vecs = []

    for sen in sens:
        # [batch_size, hidden_state]
        vec = sen_to_vec(sen, model=model, tokenizer=tokenizer)

        # [hidden_state]
        vec = vec.squeeze()

        vecs.append(vec)

    assert len(sens) == len(vecs)

    vecs = np.array(vecs)

    return vecs


# 获取句子向量集合
def get_sens_vector(data: list[list], model, tokenizer):
    # 将传入句子解开
    sens = []

    for line in data:
        for sen in line[:2]:
            sens.append(sen)

    vecs = sens_to_vecs(sens, model=model, tokenizer=tokenizer)

    return vecs


# 计算白化核和偏差
def compute_kernel_bias(vecs):
    """计算kernel和bias
    最后的变换：y = (x + bias).dot(kernel)
    """
    vecs = np.concatenate(vecs, axis=0).squeeze()
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(1 / np.sqrt(s)))
    return W, -mu


# 读取白化核及偏差
def read_kernel_bias(path):
    with open(path, 'rb') as f:
        res = pickle.load(f)
    return res['kernel'], res['bias']


# 执行线性变换，并标准化
def transform_and_normalize(vecs, kernel, bias):
    if not ((kernel is None) or (bias is None)):
        vecs = (vecs + bias).dot(kernel)
    return vecs / (vecs ** 2).sum(axis=1, keepdims=True) ** 0.5


# 只是执行标准化
def normalize(vecs):
    return vecs / (vecs ** 2).sum(axis=1, keepdims=True) ** 0.5


# ----------------------------------------------------------------for prompt-----------------------------------------------
''''''


# 找到 mask_index 位置的embeding
def get_mask_embedding(token_embeddings, mask_index):
    input_mask_expanded = mask_index.unsqueeze(-1).expand(token_embeddings.size()).float()
    mask_embedding = torch.sum(token_embeddings * input_mask_expanded, 1)
    return mask_embedding


# ====================================for compute loss======================================
''''''


# SimCSE 损失函数
def simcse_compute_loss(y_pred, lambda_=config.lambda_):
    idxs = torch.arange(0, y_pred.shape[0], device=config.device)

    idxs_true = idxs + 1 - idxs % 2 * 2

    a = idxs.unsqueeze(0)
    b = idxs_true.unsqueeze(1)

    a = a.expand(a.shape[1], a.shape[1])
    b = b.expand(b.shape[0], b.shape[0])

    y_true = a.clone().eq_(b)
    y_true = y_true.to(dtype=torch.float32)

    similarity = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)
    similarity = similarity - torch.eye(y_pred.shape[0], device=config.device) * 1e12

    similarity = similarity / lambda_
    loss = F.cross_entropy(similarity, y_true)

    return torch.mean(loss)


# Prompt Bert 损失函数
def prompt_compute_loss(out_a, out_b, lambda_):
    cos_sim = F.cosine_similarity(out_a.unsqueeze(1), out_b.unsqueeze(0), dim=-1)
    # temp or norm?
    cos_sim = cos_sim / lambda_

    loss_fct = torch.nn.CrossEntropyLoss()
    labels = torch.arange(cos_sim.size(0)).long().to(config.device)

    loss = loss_fct(cos_sim, labels)
    loss = torch.mean(loss)

    return loss


# Mo Prompt Bert 损失函数
def mo_prompt_compute_loss(out_a, out_b, guide_a, guide_b, lambda_):
    cos_sim = F.cosine_similarity(out_a.unsqueeze(1), out_b.unsqueeze(0), dim=-1)
    guide_sim = F.cosine_similarity(guide_a.unsqueeze(1), guide_b.unsqueeze(0), dim=-1)

    guide_sim = guide_sim * lambda_
    cos_sim = torch.div(cos_sim, guide_sim)

    loss_fct = torch.nn.CrossEntropyLoss()
    labels = torch.arange(cos_sim.size(0)).long().to(config.device)

    loss = loss_fct(cos_sim, labels)

    return loss


# =================================for save================================
''''''


# 创建文件夹
def mkdir(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)


# 创建子文件夹
def make_sub_folder(father_path=config.res_save_path):
    # 路径结果只到小时？还是先创建一个，然后互相传递
    # 每次执行，创建一个好像会比较好
    path = os.path.join(father_path, time.strftime("%m-%d-%H-%M", time.localtime()))

    if not os.path.exists(path):
        os.mkdir(path)

    return path


# 保存 df
def save_df(df, path):
    path = os.path.join(path, "res.csv")
    df.to_csv(path, index=False, encoding="utf_8_sig")


# ----------------------------------------------------------for whiten-----------------------------------------------------


# 保存白化核及偏差
def save_whiten(path, kernel, bias):
    whiten = {
        'kernel': kernel,
        'bias': bias
    }

    with open(path, 'wb') as f:
        pickle.dump(whiten, f)

    return path
