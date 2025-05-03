import torch
from torch.utils import data
from torchvision import transforms, datasets
import pandas as pd
from collections import Counter
from .vocab import Vocab


def _split_data(datas: list, ratio: list[int]) -> list[data.Dataset]:
    '''将数据按比例随机分割'''
    ratio = [r/sum(ratio) for r in ratio]
    nums = [int(len(datas)*r) for r in ratio]
    nums[-1] = len(datas)-sum(nums[:-1])
    return data.random_split(datas, nums)


def _iter_data(datas: list, batch_size: int, shuffle: bool = True) -> tuple[data.DataLoader]:
    '''将批量数据转换为迭代器'''
    return (data.DataLoader(_data, batch_size=batch_size, shuffle=shuffle) for _data in datas)


def mnist(path: str, *, batch_size: int = 100, download: bool = False) -> tuple[data.DataLoader, data.DataLoader, data.DataLoader]:
    '''加载数据集MNIST, 返回训练集、验证集、测试集迭代器'''
    trans = transforms.ToTensor()  # 数据集格式转换
    train_data = datasets.MNIST(root=path, train=True, transform=trans, download=download)
    test_data = datasets.MNIST(root=path, train=False, transform=trans, download=download)
    train_data, val_data = _split_data(train_data, [9, 1])  # 训练集和验证集比例9：1
    return _iter_data([train_data, val_data, test_data], batch_size)  # 返回数据迭代器


def chn_senti_corp(path: str, *, batch_size: int = 100, step_size: int = 200) -> tuple[data.DataLoader, data.DataLoader, data.DataLoader, Vocab]:
    '''加载数据集ChnSentiCorp, 返回词表和训练集、验证集、测试集迭代器'''
    chn_senti_corp = pd.read_csv(path)  # 读数据集

    token_count = Counter()  # 将文本拆分为词元并统计频率
    for item in chn_senti_corp.iloc[:, 1].values:
        token_count.update(str(item))

    vocab = Vocab(token_count, min_freq=10, reserved_tokens=['<pad>'])  # 建立词表

    # 加载并划分数据集
    chn_senti_corp_feature = [vocab.to_indices(list(str(item))[:step_size]) for item in chn_senti_corp.iloc[:, 1].values]  # 读取评论转为数字列表，评论限制在200字内
    chn_senti_corp_feature = torch.tensor([item+[1]*(step_size-len(item)) for item in chn_senti_corp_feature])  # 将列表转为tensor，同时空内容用<pad>填充
    chn_senti_corp_label = torch.tensor(chn_senti_corp.iloc[:, 0].values)  # 将标签转为tensor
    chn_senti_corp_data = data.TensorDataset(chn_senti_corp_feature, chn_senti_corp_label)  # 生成Dataset
    train_data, val_data, test_data = _split_data(chn_senti_corp_data, [0.7, 0.15, 0.15])  # 划分训练集、验证集、测试集
    train_iter, val_iter, test_iter = _iter_data([train_data, val_data, test_data], batch_size)  # 产生迭代器

    return train_iter, val_iter, test_iter, vocab  # 返回词表和迭代器
