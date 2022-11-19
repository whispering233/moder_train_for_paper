"""
提供测量指标方法
"""

import torch
import torch.nn.functional as F
import numpy as np
import scipy.stats
from tqdm import tqdm

import sys

sys.path.append("../")
from src.utils import utils as utils
from src import config

# ==============================for anisotropy======================================
''''''


# 获取向量编码
def encoder(batch, tokenizer, model):
    model.to(config.device)
    batch = utils.tokenizer_fun(batch, tokenizer)

    # cuda
    # batch: list dict or dict list
    for k in batch:
        batch[k] = batch[k].to(config.device) if batch[k] is not None else None

    # get raw embeding
    with torch.no_grad():
        outputs = model(**batch)

    return outputs


# 计算平均余弦相似度
def get_avg_cosine(embeds, n=100000):
    cos = torch.nn.CosineSimilarity(dim=-1)
    s = torch.tensor(embeds[:n]).clone().detach().to(config.device)

    res = []

    with torch.no_grad():
        for i in range(s.shape[0]):
            res.append(cos(s[i:i + 1], s).mean().item())

    return sum(res) / len(res)


def get_anisotropy(lines: list[list], tokenizer, model, batch_size=128):
    batch = []
    embeds = []

    for line in tqdm(lines, desc="获取句向量编码"):

        line = line[0]

        # batch.append(line.replace('\n', ''))
        batch.append(line.strip())

        if len(batch) >= batch_size:
            embeds.append(encoder(batch, tokenizer, model).cpu().detach().numpy())
            batch = []

    if len(batch) != 0:
        embeds.append(encoder(batch, tokenizer, model).cpu().detach().numpy())

    print("计算各向异性")

    embeds = np.concatenate(embeds, axis=0)
    # writer.add_embedding(embeds[:1000], global_step=2)
    cosine = get_avg_cosine(embeds)

    return cosine


def get_anisotropy_prompt(model, data):
    model.to(config.device)
    model.eval()

    embeds = []

    with torch.no_grad():
        for batch, data in tqdm(enumerate(data), total=(len(data.dataset) / config.batch_size), desc="get_cosine"):

            sen_a = data[0]
            sen_a = [utils.preprocess_bert_input(x) for x in sen_a]

            # 其实直接索引取值会好一点
            out_a = model(sen_a[0], sen_a[1])

            embeds.append(out_a)

    print("计算各向异性")
    embeds = np.concatenate(embeds, axis=0)
    cosine = get_avg_cosine(embeds)

    return cosine


# ==============================for corrcoef=====================================
''''''


# 计算Spearman相关系数
def compute_corrcoef(x, y):
    """Spearman相关系数
    """
    return scipy.stats.spearmanr(x, y).correlation


# 直接获取斯皮尔曼系数 for supervised data
def get_corrcoef(data, model):
    similarity_list = []
    label_list = []

    model.to(config.device)
    model.eval()

    with torch.no_grad():
        for batch, data in tqdm(enumerate(data), total=int((len(data.dataset) / data.batch_size)), desc="get_corrcoef"):
            target, source, label = data['target'], data['source'], data['label']

            target = model(*utils.preprocess_bert_input(target))

            source = model(*utils.preprocess_bert_input(source))

            similarity = F.cosine_similarity(target, source)
            similarity = similarity.cpu().detach().numpy()
            similarity_list.append(similarity)

            label = np.array(label, dtype=np.int).squeeze()
            label_list.append(label)

        similarity_list = np.concatenate(similarity_list, axis=0)
        label_list = np.concatenate(label_list, axis=0)

        corrcoef = compute_corrcoef(label_list, similarity_list)

    return corrcoef


def get_corrcoef_based_vecs(a, b, label):
    a = torch.tensor(a, dtype=torch.float32).to(config.device)
    b = torch.tensor(b, dtype=torch.float32).to(config.device)
    label = np.array(label, dtype=np.int).squeeze()

    similarity = F.cosine_similarity(a, b).cpu().detach().numpy()
    corrcoef = compute_corrcoef(similarity, label)

    return corrcoef


def get_corrcoef_prompt(model, data):
    model.to(config.device)
    model.eval()
    similarity_list = []
    label_list = []

    with torch.no_grad():
        for batch, data in tqdm(enumerate(data), total=(len(data.dataset) / config.batch_size), desc="get_corrcoef"):
            sen_a, sen_b, label = data

            sen_a = [utils.preprocess_bert_input(x) for x in sen_a]
            sen_b = [utils.preprocess_bert_input(x) for x in sen_b]

            out_a = model(sen_a[0], sen_a[1])
            out_b = model(sen_b[0], sen_b[1])

            similarity = F.cosine_similarity(out_a, out_b)
            similarity = similarity.cpu().numpy()
            similarity_list.append(similarity)

            label = np.array(label, dtype=np.int).squeeze()
            label_list.append(label)

        similarity_list = np.concatenate(similarity_list, axis=0)
        label_list = np.concatenate(label_list, axis=0)
        corrcoef = compute_corrcoef(label_list, similarity_list)
    return corrcoef
