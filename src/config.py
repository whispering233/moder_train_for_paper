import torch
import os

# ======================for preprocess==========================
''''''

unsupervised_data_recreated = False
supervised_data_recreated = False

sample = 100000
cut_ratio = (0.8, 0.1, 0.1)

# wiki
wiki_sen_least_len = 5

# news
news_max_sen = 10000

# ======================data path==============================
''''''

pwd = os.getcwd()
root = os.path.dirname(pwd)

file_paths = {
    'lcqmc': {
        'train': os.path.join(root, "data", "lcqmc", "lcqmc_train.txt"),
        'dev': os.path.join(root, "data", "lcqmc", "lcqmc_dev.txt"),
        "test": os.path.join(root, "data", "lcqmc", "lcqmc_test.txt")
    },
    'sts': {
        'train': os.path.join(root, "data", "sts-b", "sts_train.txt"),
        'dev': os.path.join(root, "data", "sts-b", "sts_dev.txt"),
        "test": os.path.join(root, "data", "sts-b", "sts_test.txt")
    },
    'wiki': {
        'train': os.path.join(root, "data", "wiki_zh", "wiki_train.txt"),
        'dev': os.path.join(root, "data", "wiki_zh", "wiki_dev.txt"),
        "test": os.path.join(root, "data", "wiki_zh", "wiki_test.txt")
    },
    'news': {
        'train': os.path.join(root, "data", "news", "news_train.txt"),
        'dev': os.path.join(root, "data", "news", "news_dev.txt"),
        "test": os.path.join(root, "data", "news", "news_test.txt")
    }
}

old_file_paths = [["../data/last/lcqmc_train.txt", "../data/last/lcqmc_dev.txt", "../data/last/lcqmc_test.txt"],
                  ["../data/last/cnsd-sts-train.txt", "../data/last/cnsd-sts-dev.txt", "../data/last/cnsd-sts-test.txt"],
                  ["../data/last/last_news_train.txt", "../data/last/last_news_dev.txt", "../data/last/last_news_test.txt"],
                  ["../data/last/last_wiki_train.txt", "../data/last/last_wiki_dev.txt", "../data/last/last_wiki_test.txt"]]

raw_file_paths = {
    'lcqmc': {
        'train': os.path.join(root, "data_raw", "lcqmc", "lcqmc_train.tsv"),
        'dev': os.path.join(root, "data_raw", "lcqmc", "lcqmc_dev.tsv"),
        "test": os.path.join(root, "data_raw", "lcqmc", "lcqmc_test.tsv")
    },
    'sts': {
        'train': os.path.join(root, "data_raw", "sts-b", "cnsd-sts-train.txt"),
        'dev': os.path.join(root, "data_raw", "sts-b", "cnsd-sts-dev.txt"),
        "test": os.path.join(root, "data_raw", "sts-b", "cnsd-sts-test.txt")
    },
    'wiki': "D:/Develop/Data/AI/wiki_zh_2019/wiki_zh",
    'news': os.path.join(root, "data_raw", "news", "toutiao_cat_data.txt")
}

# =======================for read data================================
''''''

# 读取文件时要不要提前随机
read_data_shuffle = True

# =======================for model===================================
''''''

# pool_ways = ["first_last_avg", "last_avg", "cls", "pooler"]
pool_ways = ["first_last_avg", "cls"]

# check_point = "hfl/chinese-roberta-wwm-ext"
check_point = "voidful/albert_chinese_tiny"

check_points = ["voidful/albert_chinese_tiny",
                "voidful/albert_chinese_base",
                "clue/roberta_chinese_clue_tiny",
                "clue/roberta_chinese_base",
                "hfl/chinese-roberta-wwm-ext",
                "bert-base-chinese",
                "hfl/chinese-bert-wwm-ext"]

# check_points_main = check_points
# check_points_main = ["clue/roberta_chinese_base", "bert-base-chinese"]
check_points_main = ["voidful/albert_chinese_tiny"]

# -------------------------------------for prompt-------------------------------------------------------
''''''

# token
replace_token = "[X]"
mask_token = "[MASK]"

special_tokens_dict = {
    'additional_special_tokens': ['[X]']
}

prompt_templates = ['{}，[MASK]。'.format(replace_token),
                    '{}[MASK]。'.format(replace_token),
                    '{}，描述了[MASK]。'.format(replace_token),
                    '{}描述了[MASK]。'.format(replace_token),
                    '“{}”，这句话描述了[MASK]。'.format(replace_token),
                    '这句话“{}”描述了[MASK]。'.format(replace_token),
                    '{}，意思是[MASK]。'.format(replace_token),
                    '{}意思是[MASK]。'.format(replace_token),
                    '“{}”，这句话的意思是[MASK]。'.format(replace_token),
                    '这句话“{}”的意思是[MASK]。'.format(replace_token),
                    '{}，主题是[MASK]。'.format(replace_token),
                    '{}主题是[MASK]。'.format(replace_token),
                    '“{}”，这句话的主题是[MASK]。'.format(replace_token),
                    '这句话“{}”的主题是[MASK]。'.format(replace_token)
                    ]

prompt_templates_main = ['{}，[MASK]。'.format(replace_token),
                         '{}[MASK]。'.format(replace_token),
                         '{}，主题是[MASK]。'.format(replace_token),
                         '{}主题是[MASK]。'.format(replace_token),
                         '“{}”，这句话的主题是[MASK]。'.format(replace_token),
                         '这句话“{}”的主题是[MASK]。'.format(replace_token)]

prompt_template = [['{}主题是[MASK]'.format(replace_token)]]

# ======================for train===================================
''''''

# cuda
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# epochs
epochs = 3

# data set
batch_size = 64
shuffle = True

# tokenizer
max_len = 64
truncation = True
padding = 'max_length'
return_tensors = 'pt'

# for try
total_lines_for_try = batch_size * 2
# total_lines_for_try = -1

# -----------------------------------hyper parameters-------------------------------------------

learning_rate = 1e-5

# bert structure
attention_dropout = 0.1
hidden_dropout = 0.1

# bert out
pool_way = "first_last_avg"

# simcse  temperature
lambda_ = 0.05

# mo prompt temperature
mo_lambda_ = 0.05

# prompt template

template_noise = False
template_noise_probability = 0.0
template_noise_dup_rate = 0.25

denoise = True

# momentum
momentum = 0.995

# =================================for save===============================
''''''

# data save


# obj_save 写入静态文件 重复保存最优模型
obi_save_path = os.path.join(root, "obj", "best_model.pth")

# res_save
res_save_path = os.path.join(root, "res")

if __name__ == '__main__':
    print(pwd)
    print(root)
    print(os.path.join(root, "data_raw", "lcqmc", "lcqmc_train.tsv"))
