import pandas as pd

from utils import preprocess
from utils import utils
from baseline import baseline
from simcse import simcse
from whiten import whiten
from promptbaseline import prompt_baseline
from moprompt import moprompt


# 一键运行
def main():
    # 任务队列

    # 先将收集到的数据集处理成统一格式
    # preprocess.transfer_raw_into_dataset()

    # 创建本次运行结果保存文件夹
    path = utils.make_sub_folder()

    # 结果表
    df = pd.DataFrame(columns=["method", "check_point", "dataset", "sens", "pool_way", "epoch", "step", "dev_anisotropy",
                               "dev_corrcoef", "test_anisotropy", "test_corrcoef"])
    df_list = [df]

    # baseline 不用存储模型，就不用传入文件路径
    baseline_df = baseline()
    df_list.append(baseline_df)

    whiten_df = whiten(model_save_folder=path)
    df_list.append(whiten_df)

    simcse_df = simcse(model_save_folder=path)
    df_list.append(simcse_df)

    prompt_df = prompt_baseline(model_save_folder=path)
    df_list.append(prompt_df)

    moprompt_df = moprompt(model_save_folder=path)
    df_list.append(moprompt_df)

    df = pd.concat(df_list)

    # 保存结果
    utils.save_df(df, path=path)


'''

todo: 
    
    template search
    
    flow
    
    ablation study
'''

if __name__ == '__main__':
    main()
