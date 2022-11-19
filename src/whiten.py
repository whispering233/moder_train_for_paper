"""
数据后处理
"""

import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

import config
from utils import utils, measurement

from models import BaseLine


def whiten(model_save_folder, file_paths=config.file_paths, check_points=config.check_points_main, pool_ways=config.pool_ways):

    # 构建whiten结果参数保存文件夹
    parameters_save_path = os.path.join(model_save_folder, "parameters")
    utils.mkdir(parameters_save_path)

    df = pd.DataFrame(columns=["method", "check_point", "dataset", "sens", "pool_way", "epoch", "step",
                               "dev_anisotropy", "dev_corrcoef", "test_anisotropy", "test_corrcoef"])

    for data_name, files in tqdm(file_paths.items(), desc="whiten 进度"):

        try:
            train_data, dev_data, test_data = utils.read_from_dict(files)
        except ValueError:
            print("数据集: {}文件不全".format(data_name))
            continue

        supervised = utils.judge_supervised(files["train"])

        for check_point in check_points:

            tokenizer = utils.load_tokenizer(check_point)

            for pool_way in pool_ways:

                model = BaseLine(check_point, pool_way)
                model.to(config.device)

                # 获取句子表征向量集合
                # whitening su. 原文是所有数据集都用上了
                train_vecs = utils.get_sens_vector(data=train_data, model=model, tokenizer=tokenizer)
                dev_vecs = utils.get_sens_vector(data=dev_data, model=model, tokenizer=tokenizer)
                test_vecs = utils.get_sens_vector(data=test_data, model=model, tokenizer=tokenizer)

                # 计算白化核和偏差
                kernel, bias = utils.compute_kernel_bias(vecs=[train_vecs, dev_vecs, test_vecs])

                # 保存白化核及偏差
                file_name = "whiten-{}-{}-{}.pkl".format(check_point.split("/")[-1], os.path.split(files["train"])[1].split(".")[0], pool_way)
                save_path = os.path.join(parameters_save_path, file_name)
                save_path = utils.save_whiten(save_path, kernel=kernel, bias=bias)
                print("Save to {}".format(save_path))

                # 重新读取白化核及偏差
                kernel, bias = utils.read_kernel_bias(save_path)

                # 执行线性变换
                if supervised:

                    a_dev, b_dev, label_dev = utils.detach_sen_pair_supervised(dev_data)
                    a_test, b_test, label_test = utils.detach_sen_pair_supervised(test_data)

                    a_dev_vecs = utils.get_sens_vector(data=a_dev, model=model, tokenizer=tokenizer)
                    b_dev_vecs = utils.get_sens_vector(data=b_dev, model=model, tokenizer=tokenizer)
                    a_test_vecs = utils.get_sens_vector(data=a_test, model=model, tokenizer=tokenizer)
                    b_test_vecs = utils.get_sens_vector(data=b_test, model=model, tokenizer=tokenizer)

                    a_dev_vecs = utils.transform_and_normalize(a_dev_vecs, kernel=kernel, bias=bias)
                    b_dev_vecs = utils.transform_and_normalize(b_dev_vecs, kernel=kernel, bias=bias)
                    a_test_vecs = utils.transform_and_normalize(a_test_vecs, kernel=kernel, bias=bias)
                    b_test_vecs = utils.transform_and_normalize(b_test_vecs, kernel=kernel, bias=bias)

                    # 计算Spearman
                    dev_v = measurement.get_corrcoef_based_vecs(a_dev_vecs, b_dev_vecs, label=label_dev)
                    test_v = measurement.get_corrcoef_based_vecs(a_test_vecs, b_test_vecs, label=label_test)

                else:

                    dev_vecs = utils.transform_and_normalize(dev_vecs, kernel=kernel, bias=bias)
                    test_vecs = utils.transform_and_normalize(test_vecs, kernel=kernel, bias=bias)

                    dev_vecs = torch.tensor(dev_vecs, dtype=torch.float32).to(config.device)
                    test_vecs = torch.tensor(test_vecs, dtype=torch.float32).to(config.device)

                    # 计算 anisotropy
                    dev_v = measurement.get_avg_cosine(dev_vecs)
                    test_v = measurement.get_avg_cosine(test_vecs)

                df.loc[df.shape[0]] = {
                    "method": "whiten",
                    'check_point': check_point,
                    'dataset': os.path.split(files["train"])[1],
                    "sens": len(train_data),
                    "pool_way": pool_way,
                    "epoch": np.nan,
                    "step": np.nan,
                    "dev_anisotropy": np.nan if supervised else dev_v,
                    "dev_corrcoef": dev_v if supervised else np.nan,
                    'test_anisotropy': np.nan if supervised else test_v,
                    'test_corrcoef': test_v if supervised else np.nan,
                }

    return df
