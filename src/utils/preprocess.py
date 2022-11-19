"""
this file is to preprocess dataset into uniform format
"""
import random

import pandas as pd

from tqdm import tqdm
import os
import json

import sys

sys.path.append("../")
from src import config

# ====================for supervised data===================================
''''''

# -----------------------------------for process data---------------------------------------------------------------------
''''''


# 过滤一句话里的字符
def filter_sen(sen: str):
    """
    todo: filter_set 传入 需要过滤掉的字符集
    :param sen:
    :return:
    """
    return sen.replace("\n", "").replace("()", "").replace("（）", "").replace("《》", "").strip()


# 判断标签范围读取数据
def judge_label(label: int, min_label=-1, max_label=-1):
    """
    左闭右闭区间
    :param label:
    :param min_label:
    :param max_label:
    :return:
    """
    if (min_label < 0) or (max_label < 0):
        return True
    else:
        return (label <= max_label) & (label >= min_label)


# 将各数据集里的一行数据分割成统一的格式并返回
def partition_line_supervised(line, file_path: str) -> tuple:
    """
    通过 多重 if 实现
    :param line:
    :param file_path: 根据文件路径来判断 数据集类型
    :return: tuple
    """
    sen_a = None
    sen_b = None
    label = None
    if "sts" in file_path:
        res = line.strip().split("||")
        if len(res) == 4:
            sen_a = res[1]
            sen_b = res[2]
            label = res[3]
    elif ("lcqmc" in file_path) or ("afqmc" in file_path):
        res = line.strip().split("\t")
        if len(res) == 3:
            sen_a = res[0]
            sen_b = res[1]
            label = res[2]
    try:
        sen_a = sen_a.strip()
        sen_b = sen_b.strip()
        label = label.strip()
    except AttributeError:
        return None, None, None
    return sen_a, sen_b, label


# 分割无监督数据集
def cut_data_unsupervised(data: list, ratio: tuple = config.cut_ratio, sample=config.sample) -> tuple:
    # 先进行随机采样

    # 有时候只需要直接分割，再随机采样会报 ValueError
    if sample is not None:
        data = random.sample(data, sample)

    # 分割数据集
    left = int(ratio[0] * len(data))
    right = int((ratio[0] + ratio[1]) * len(data))
    train = data[: left]
    dev = data[left: right]
    test = data[right:]

    return train, dev, test


# --------------------------------------for read data----------------------------------------------------------
''''''


# 读取有监督数据集
def read_supervised_data(file_path, **kwargs):
    """
    :param file_path:
    :param kwargs: 调用了 judge_label(): 形参(min_label, max_label) 左闭右闭区间
    :return:
    """
    res = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            sen_a, sen_b, label = partition_line_supervised(line, file_path)
            if (sen_a is None) or (sen_b is None) or (label is None):
                continue
            try:
                if judge_label(int(float(label)), **kwargs):
                    res.append([sen_a, sen_b, label])
            except ValueError:
                print(label)
                continue
    f.close()
    return res


# 读取wiki_zh数据集
def read_wiki_zh(file_folder=config.raw_file_paths['wiki'], least_length=config.wiki_sen_least_len):
    res = []
    all_sens = 0
    all_text = 0
    sub_folder = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M']
    for sub in tqdm(sub_folder):
        sub = file_folder + "/A" + sub
        for i in range(100):
            i = str(i) if i > 9 else ("0" + str(i))
            file = sub + "/wiki_" + i
            if os.path.exists(file):
                with open(file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    all_text += len(lines)
                    for line in lines:
                        line = json.loads(line)
                        sens = line['text'].split('。')
                        sens = [filter_sen(sen) for sen in sens]
                        sens = [sen for sen in sens if len(sen) > least_length]
                        all_sens += len(sens)
                        res += sens
                f.close()
    print("wiki 数据集总篇数为{}".format(all_text))
    print("wiki 数据集总句数为{}".format(all_sens))
    return res


# 读取toutiao news 数据集
def read_toutiao_news(file_path=config.raw_file_paths['news'], max_sen=config.news_max_sen) -> tuple:
    """
    :param file_path:
    :param max_sen: 每个类别抽取同样多的句子
    :return:
    """

    news = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print("头条新闻的句子总数为: {}".format(len(lines)))
        for line in tqdm(lines):
            sens = line.strip().split("_!_")
            label_desc = sens[2]
            sentence = sens[3]
            if label_desc not in news.keys():
                # 没有的话就先占个位置
                news[label_desc] = []
            news[label_desc].append(sentence)
    f.close()

    # 如果后面要全部取出来，这里就没有必要全部存进去
    # 只要打印出数据集的信息就行了，或者 return 出去看之后怎么办
    # 应该直接返回3个，在这里就直接等类型分割数据集
    train_res = []
    dev_res = []
    test_res = []
    df_desc = pd.DataFrame(columns=['type', 'count'])
    for key, item in news.items():

        count = len(item)

        df_desc = pd.concat([df_desc, pd.DataFrame({
            'type': key,
            'count': count
        }, index=[df_desc.shape[0]])])

        if count >= max_sen:
            res = cut_data_unsupervised(random.sample(item, max_sen), sample=None)

            # 这里要把 res 解开
            train_res += res[0]
            dev_res += res[1]
            test_res += res[2]

    print(df_desc)

    # 最后对其随机一下
    random.shuffle(train_res)
    random.shuffle(dev_res)
    random.shuffle(test_res)

    return train_res, dev_res, test_res


# 将 raw 里的文件 全部转换成 标准格式
def transfer_raw_into_dataset(raw_paths: dict = config.raw_file_paths, file_paths: dict = config.file_paths):
    """
    只判断文件路径有没有没有什么用，
    有文件路径但是文件里面没有什么内容
    todo: label_dict 按标签范围转换 raw
    :param raw_paths:
    :param file_paths:
    :return:
    """

    bar = tqdm(total=len(raw_paths) * 3)

    for data_name, path_dict in raw_paths.items():
        if data_name == "wiki":
            train, dev, test = cut_data_unsupervised(read_wiki_zh())

            # 这里不能这样，要一一对应，重构一个dict就行
            res = {
                'train': train,
                'dev': dev,
                'test': test
            }

            for k, v in file_paths[data_name].items():
                write_list_unsupervised(res[k], v)
                bar.update(1)

        elif data_name == "news":
            train, dev, test = read_toutiao_news()

            res = {
                'train': train,
                'dev': dev,
                'test': test
            }

            for k, v in file_paths[data_name].items():
                write_list_unsupervised(res[k], v)
                bar.update(1)
        else:
            for k, v in raw_paths[data_name].items():
                write_list_supervised(read_supervised_data(v), file_paths[data_name][k])
                bar.update(1)


# ------------------------------------------for save data--------------------------------------------------------
''''''


# 将 list 写成统一的格式到 文件中
def write_list_supervised(data, save_path, recreated=config.supervised_data_recreated):
    if os.path.exists(save_path) and (not recreated):
        return

    with open(save_path, 'w', encoding='utf-8') as f:
        for index, line in tqdm(enumerate(data)):
            for sen in line[:len(line) - 1]:
                f.write(sen)
                f.write("_||_")
            f.write(line[-1])
            if index < len(data) - 1:
                f.write("\n")
    f.close()


# 写入无监督数据集
def write_list_unsupervised(data: list, save_path: str, recreated=config.unsupervised_data_recreated):
    if os.path.exists(save_path) and (not recreated):
        return

    with open(save_path, 'w', encoding='utf-8') as f:
        for i in tqdm(range(len(data))):
            f.write(data[i])
            if i < len(data) - 1:
                f.write("\n")
    f.close()
