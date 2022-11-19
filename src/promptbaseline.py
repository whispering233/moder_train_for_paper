"""
提供最原始版本的Prompt Bert 基线性能

1. 输入模板
2. 重设tokenizer
"""

import pandas as pd
import numpy as np
import torch.optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import os

import config
from utils import utils, measurement
from models import PromptBERT
from dataset import PromptDataset


def train(supervised, epochs, dataloader, dev_dataloader, model, optimizer, save_path, lambda_):

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

            sen_a, sen_b, label = data

            sen_a = [utils.preprocess_bert_input(x) for x in sen_a]
            sen_b = [utils.preprocess_bert_input(x) for x in sen_b]

            out_a = model(*sen_a)
            out_b = model(*sen_b)

            loss = utils.prompt_compute_loss(out_a=out_a, out_b=out_b, lambda_=lambda_)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1

            # if batch % int(total / 20) == 0:
            if batch % 1 == 0:

                loss = loss.item()
                current = ((batch * int(len(sen_a[0][0]) / 2)) + (epoch - 1) * len(dataloader.dataset))

                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

                model.eval()

                if supervised:
                    v = measurement.get_corrcoef_prompt(model=model, data=dev_dataloader)
                    print(f"corrcoef_dev: {v:>4f}")

                    if v > target:
                        target = v
                        optimal_step = step
                        torch.save(model.state_dict(), save_path)
                        print(f"Higher corrcoef: {target :>4f}, Saved PyTorch Model State to {save_path}")
                else:
                    v = measurement.get_anisotropy_prompt(model=model, data=dev_dataloader)
                    print(f"anisotropy_dev: {v:>4f}")

                    if v < target:
                        target = v
                        optimal_step = step
                        torch.save(model.state_dict(), save_path)
                        print(f"Lower anisotropy: {target :>4f}, Saved PyTorch Model State to {save_path}")

                model.train()

    return target, optimal_step


# 模板组合搜索耗费计算量过大
def prompt_baseline(model_save_folder, epochs=config.epochs,
                    file_paths=config.file_paths, check_points=config.check_points_main,
                    templates=config.prompt_template, denoise=config.denoise,
                    attention_dropout=config.attention_dropout, hidden_dropout=config.hidden_dropout, lr=config.learning_rate, lambda_=config.lambda_):
    # 创建 prompt_baseline 训练模型保存文件夹
    model_save_path = os.path.join(model_save_folder, "prompt_baseline")
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

            # 加载 prompt 专用词表
            tokenizer, mask_id, = utils.load_tokenizer_prompt(check_point=check_point)

            # 取 [MASK] 位置上的 embedding，没必要考虑pool_way
            # for pool_way in pool_ways:

            for template in templates:

                # 构造数据集
                # 输入还是同一个句子，不同的句子的话要考虑标签
                # label 只是在 dev, test 中用得到
                train_prompt = utils.build_prompt_inputs(data=train_data, template=template, tokenizer=tokenizer)
                dev_prompt = utils.build_prompt_inputs(data=dev_data, template=template, tokenizer=tokenizer)
                test_prompt = utils.build_prompt_inputs(data=test_data, template=template, tokenizer=tokenizer)

                # template 的数量不同，只会影响 model 和 dataset，其他流程是一样的
                # 若 template 是字符串，那长度必不为 2
                if len(template) == 2:
                    # 两个 template，不同模板构建正例对
                    encoder = utils.load_model(check_point)

                else:
                    # 一个模板，通过 drop out 构建正例对
                    encoder = utils.load_pretrained_model(check_point=check_point, attention_dropout=attention_dropout, hidden_dropout=hidden_dropout)

                model = PromptBERT(encoder=encoder, mask_id=mask_id, denoise=denoise)
                model.encoder.resize_token_embeddings(len(tokenizer))

                train_dataset = PromptDataset(data=train_prompt, tokenizer=tokenizer)
                train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=config.shuffle)

                dev_dataset = PromptDataset(data=dev_prompt, tokenizer=tokenizer)
                dev_dataloader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=config.shuffle)

                test_dataset = PromptDataset(data=test_prompt, tokenizer=tokenizer)
                test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=config.shuffle)

                # 优化器
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

                save_path = os.path.join(model_save_path, "{}-{}-{}.pth".format(check_point.split("/")[-1],
                                                                                os.path.split(files["train"])[1].split(".")[0],
                                                                                "_".join(template)))
                print("Train_Start!")
                dev_v, step = train(supervised=supervised, epochs=epochs, dataloader=train_dataloader, dev_dataloader=dev_dataloader,
                                    model=model, optimizer=optimizer, save_path=save_path, lambda_=lambda_)
                print("Train_Done!")

                print("Test_Start!")
                # 从保存的文件里加载模型
                model.load_state_dict(torch.load(save_path))

                if supervised:
                    test_v = measurement.get_corrcoef_prompt(model=model, data=test_dataloader)
                    print(f"test_corrcoef: {test_v:>4f}")
                else:
                    test_v = measurement.get_anisotropy_prompt(model=model, data=test_dataloader)
                    print(f"test_anisotropy: {test_v:>4f}")

                df.loc[df.shape[0]] = {
                    "method": "prompt_baseline",
                    'check_point': check_point,
                    'dataset': os.path.split(files["train"])[1],
                    "sens": len(train_data),
                    "pool_way": np.nan,
                    "templates": "_".join(template),
                    "epoch": epochs,
                    "step": step,
                    "dev_anisotropy": np.nan if supervised else dev_v,
                    "dev_corrcoef": dev_v if supervised else np.nan,
                    'test_anisotropy': np.nan if supervised else test_v,
                    'test_corrcoef': test_v if supervised else np.nan,
                }

    return df
