"""
simcse 的一般过程
"""

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

import config
from utils import utils, measurement
from dataset import SimCSEDataset, CommonDataset
from models import SimCSE


def train(supervised, epochs, dataloader, dev_data, dev_dataloader, model, tokenizer, optimizer, save_path, lambda_):
    model.to(config.device)

    if supervised:
        target = 0
    else:
        target = 10000

    size = len(dataloader.dataset) * epochs
    total = (len(dataloader.dataset) / config.batch_size)

    step = 0
    optimal_step = 0

    for epoch in range(1, epochs + 1):

        print(f"Epoch {epoch}\n-------------------------------")
        model.train()

        for batch, data in tqdm(enumerate(dataloader), total=total, postfix={'epoch': epoch}):

            # 构建正例对
            input_ids = data['input_ids'].view(len(data['input_ids']) * 2, -1).to(config.device)
            attention_mask = data['attention_mask'].view(len(data['attention_mask']) * 2, -1).to(config.device)
            token_type_ids = data['token_type_ids'].view(len(data['token_type_ids']) * 2, -1).to(config.device)

            pred = model(input_ids, attention_mask, token_type_ids)
            loss = utils.simcse_compute_loss(y_pred=pred, lambda_=lambda_)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1

            # if batch % int(total / 20) == 0:
            if batch % 1 == 0:

                loss = loss.item()
                current = ((batch * int(len(input_ids) / 2)) + (epoch - 1) * len(dataloader.dataset))

                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

                model.eval()

                if supervised:
                    v = measurement.get_corrcoef(dev_dataloader, model)
                    print(f"corrcoef_dev: {v:>4f}")

                    if v > target:
                        target = v
                        optimal_step = step
                        torch.save(model.state_dict(), save_path)
                        print(f"Higher corrcoef: {target :>4f}, Saved PyTorch Model State to {save_path}")
                else:
                    v = measurement.get_anisotropy(dev_data, tokenizer=tokenizer, model=model)
                    print(f"anisotropy_dev: {v:>4f}")

                    if v < target:
                        target = v
                        optimal_step = step
                        torch.save(model.state_dict(), save_path)
                        print(f"Lower anisotropy: {target :>4f}, Saved PyTorch Model State to {save_path}")

                model.train()

    return target, optimal_step


def simcse(model_save_folder, epochs=config.epochs,
           file_paths=config.file_paths, check_points=config.check_points_main, pool_ways=config.pool_ways,
           attention_dropout=config.attention_dropout, hidden_dropout=config.hidden_dropout, lr=config.learning_rate, lambda_=config.lambda_):

    # 创建 simcse 训练模型保存文件夹
    model_save_path = os.path.join(model_save_folder, "simcse")
    utils.mkdir(model_save_path)

    df = pd.DataFrame(columns=["method", "check_point", "dataset", "sens", "pool_way", "templates", "epoch", "step",
                               "dev_anisotropy", "dev_corrcoef", "test_anisotropy", "test_corrcoef"])

    for data_name, files in file_paths.items():

        try:
            train_data, dev_data, test_data = utils.read_from_dict(files)
        except ValueError:
            print("数据集: {}文件不全".format(data_name))
            continue

        supervised = utils.judge_supervised(files["train"])

        for check_point in check_points:

            # 加载 tokenizer
            tokenizer = utils.load_tokenizer(check_point)

            for pool_way in pool_ways:
                # 加载模型 后面会通过反向传播进行更新，所以每次都要重新实例化
                # pool_way 不应该写入类内部，会不够灵活，但是写在外面的话，模型保存不好搞
                # 还是要写在里面，每次都是重新训练一个，如果写在外面，每次训练都相当于在之前的基础上训练的
                # 既然每种条件下都要重新实例化并训练，那还不如直接将 pool_way 写在里面
                # 写在里面可以控制池化方式一致
                model = SimCSE(check_point=check_point, pool_way=pool_way, attention_dropout=attention_dropout, hidden_dropout=hidden_dropout)

                # 加载优化器
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

                # load dataset
                train_dataset = SimCSEDataset(train_data, tokenizer)
                train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size)

                # 构建模型存储文件
                save_path = os.path.join(model_save_path, "{}-{}-{}.pth".format(check_point.split("/")[-1], os.path.split(files["train"])[1].split(".")[0], pool_way))

                dev_dataset = CommonDataset(dev_data, tokenizer)
                dev_dataloader = DataLoader(dev_dataset, shuffle=True, batch_size=config.batch_size)

                test_dataset = CommonDataset(test_data, tokenizer)
                test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=config.batch_size)

                print("Train_Start!")
                dev_v, step = train(supervised=supervised, epochs=epochs, dataloader=train_dataloader,
                                    dev_data=dev_data, dev_dataloader=dev_dataloader, model=model, tokenizer=tokenizer, optimizer=optimizer,
                                    save_path=save_path, lambda_=lambda_)
                print("Train_Done!")

                print("Test_Start!")
                # 从保存的文件里加载模型
                model.load_state_dict(torch.load(save_path))

                if supervised:
                    test_v = measurement.get_corrcoef(test_dataloader, model)
                    print(f"test_corrcoef: {test_v:>4f}")
                else:
                    test_v = measurement.get_anisotropy(test_data, tokenizer=tokenizer, model=model)
                    print(f"test_anisotropy: {test_v:>4f}")

                print("Test_Done!")

                df.loc[df.shape[0]] = {
                    "method": "simcse",
                    'check_point': check_point,
                    'dataset': os.path.split(files["train"])[1],
                    "sens": len(train_data),
                    "pool_way": pool_way,
                    "templates": np.nan,
                    "epoch": epochs,
                    "step": step,
                    "dev_anisotropy": np.nan if supervised else dev_v,
                    "dev_corrcoef": dev_v if supervised else np.nan,
                    'test_anisotropy': np.nan if supervised else test_v,
                    'test_corrcoef': test_v if supervised else np.nan,
                }

    return df
