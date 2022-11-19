"""
模型在没有经过对比学习训练时的基线性能
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

import config
from utils import utils
from utils import measurement

from models import BaseLine
from dataset import CommonDataset


# 方法体里的一切参数应该从形参里面得来
def baseline(file_paths=config.file_paths, check_points=config.check_points_main, pool_ways=config.pool_ways):
    df = pd.DataFrame(columns=["method", "check_point", "dataset", "sens", "pool_way", "templates", "epoch", "step",
                               "dev_anisotropy", "dev_corrcoef", "test_anisotropy", "test_corrcoef"])

    for data_name, paths in tqdm(file_paths.items(), desc="数据集评测进度"):

        test_file = paths['test']
        if not os.path.exists(test_file):
            print("文件{}不存在".format(test_file))

        data = utils.read_standard_data(test_file)
        supervised = utils.judge_supervised(test_file)

        for check_point in check_points:

            # tokenizer 只跟 check_point 有关，跟池化方式无关，可以先实例化一个
            tokenizer = utils.load_tokenizer(check_point=check_point)

            for pool_way in pool_ways:

                model = BaseLine(check_point=check_point, pool_way=pool_way)

                dataset = CommonDataset(data, tokenizer)
                dataloader = DataLoader(dataset, batch_size=config.batch_size)

                if supervised:
                    v = measurement.get_corrcoef(dataloader, model)
                    print(f"corrcoef: {v:>4f}")
                else:
                    v = measurement.get_anisotropy(data, tokenizer=tokenizer, model=model)
                    print(f"test_anisotropy: {v:>4f}")

                df.loc[df.shape[0]] = {
                    # D:\Develop\Project\Python\AI\for_paper\data\lcqmc\lcqmc_dev.txt
                    # columns=["method", "check_point", "dataset", "sens", "pool_way","epoch", "step", "dev_anisotropy", "dev_corrcoef", "test_anisotropy", "test_corrcoef"]
                    'method': "baseline",
                    'check_point': check_point,
                    'dataset': os.path.split(test_file)[1],
                    "sens": len(data),
                    "pool_way": pool_way,
                    "templates": np.nan,
                    "epoch": np.nan,
                    "step": np.nan,
                    "dev_anisotropy": np.nan,
                    "dev_corrcoef": np.nan,
                    'test_anisotropy': np.nan if supervised else v,
                    'test_corrcoef': v if supervised else np.nan,
                }

    return df
