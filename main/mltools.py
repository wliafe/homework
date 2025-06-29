import torch
from torch.utils import data
from torchvision import transforms, datasets
import re
import json
import time
import httpx
import logging
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from IPython import display
from datetime import datetime
from collections import Counter
from matplotlib import pyplot as plt


# 数据保存器
# 继承数据保存器类可以获得保存和加载数据到json文件的功能
class DataSaveToJson:
    '''json数据保存器'''

    def save_data(path, label, datas):
        '''保存数据'''
        try:
            with open(path, 'r') as file:
                data = json.load(file)
        except FileNotFoundError:
            data = {}
        with open(path, 'w') as f:
            data[label] = datas
            json.dump(data, f, indent=4)

    def load_data(path, label):
        '''从json文件导入'''
        with open(path, 'r') as file:
            return json.load(file)[label]


# 分词器
# 分词器是一个字典，键是词元，值是索引
# 分词器的长度是词典的大小
# 分词器的索引是词元的索引
class Tokenizer:
    '''分词器'''

    def __init__(self, datas, min_freq=0):
        '''
        初始化

        datas: list[str] 数据集

        min_freq: int 最小词频, 默认值0
        '''
        tokens = Counter()  # 将文本拆分为词元并统计频率
        for item in datas:
            tokens.update(str(item))
        self.unk = 0  # 未知词元索引为0
        self.cls = 1  # 分类词元索引为1
        self.sep = 2  # 分隔词元索引为2
        self.pad = 3  # 填充词元索引为3
        tokens = [item[0] for item in tokens.items() if item[1] > min_freq]  # 删除低频词元
        self.idx_to_token = ['[UNK]', '[CLS]', '[SEP]', '[PAD]'] + tokens  # 建立词元列表
        # 建立词元字典
        tokens_dict = {value: index + 4 for index, value in enumerate(tokens)}
        self.token_to_idx = {'[UNK]': 0, '[CLS]': 1, '[SEP]': 2, '[PAD]': 3}
        self.token_to_idx.update(tokens_dict)

    def __call__(self, tokens, max_length=None):
        return self.encode(tokens, max_length)

    def __len__(self):
        '''返回词表大小'''
        return len(self.idx_to_token)

    def decode(self, indices):
        '''根据索引返回词元'''
        if isinstance(indices, torch.Tensor):
            if indices.dim() == 0:
                return []
            elif indices.dim() == 1:
                return ''.join([self.idx_to_token[index] for index in indices.tolist()])
            elif indices.dim() == 2:
                return [''.join([self.idx_to_token[item] for item in index]) for index in indices.tolist()]
        else:
            raise TypeError('indices must be torch.Tensor')

    def encode(self, texts, max_length=None):
        '''根据词元返回索引'''
        if isinstance(texts, str):
            if max_length:
                texts = list(texts)[:max_length] if len(texts) > max_length else list(texts) + ['[PAD]'] * (max_length - len(texts))
            return torch.tensor([self.token_to_idx.get(token, self.unk) for token in texts])
        elif isinstance(texts, (list, tuple)):
            if not max_length:
                max_length = max([len(text) for text in texts])
            return torch.stack([self.encode(text, max_length) for text in texts])
        else:
            raise TypeError(f'texts: {texts}\nThe type of texts is {type(texts)}, while texts must be of type str, tuple[str] or list[str]')

    def save(self, path, label='tokenizer'):
        '''保存数据'''
        DataSaveToJson.save_data(path, label, [self.idx_to_token, self.token_to_idx])

    def load(self, path, label='tokenizer'):
        '''加载数据'''
        self.idx_to_token, self.token_to_idx = DataSaveToJson.load_data(path, label)


# 自定义数据集
class MyDataset(data.Dataset):
    def __init__(self, datas):
        data.Dataset.__init__(self)
        self.data = datas

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# load_data 多个数据集加载器
# 数据集加载器是函数，用于加载数据集
# 数据集加载器的返回值是训练集、验证集、测试集的迭代器
# 数据集加载器的参数是数据集的路径、批量大小、是否下载数据集
def split_data(datas, ratio):
    '''将数据按比例随机分割'''
    ratio = [r / sum(ratio) for r in ratio]
    nums = [int(len(datas) * r) for r in ratio]
    nums[-1] = len(datas) - sum(nums[:-1])
    return data.random_split(datas, nums)


def iter_data(datas, batch_size, shuffle=True):
    '''将批量数据转换为迭代器'''
    return (data.DataLoader(_data, batch_size=batch_size, shuffle=shuffle) for _data in datas)


def download_file(url, save_path):
    '''文件下载'''
    file_name = re.search(r'(?<=/)[^/]+$', url).group()  # 从url中提取文件名
    if not Path(f'{save_path}/{file_name}').exists():  # 如果文件不存在则下载
        Path(save_path).mkdir(parents=True, exist_ok=True)  # 创建保存路径
        with httpx.Client() as client:
            with client.stream('GET', url) as response:
                response.raise_for_status()  # 检查响应状态码
                total_size = int(response.headers.get('Content-Length', 0))  # 获取文件大小
                with open(f'{save_path}/{file_name}', 'wb') as f, tqdm(desc=file_name, total=total_size, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                    for chuck in response.iter_bytes():
                        f.write(chuck)
                        pbar.update(len(chuck))
    return file_name


def mnist(path='../data', batch_size=100):
    '''
    加载数据集MNIST

    path: 数据集路径, 默认值'../data'

    batch_size: 批量大小, 默认值100

    download: 是否下载数据集, 默认值False

    返回训练集、验证集、测试集迭代器
    '''
    download = False if Path(f'{path}/MNIST').exists() else True
    trans = transforms.ToTensor()  # 数据集格式转换
    train_data = datasets.MNIST(root=path, train=True, transform=trans, download=download)
    test_data = datasets.MNIST(root=path, train=False, transform=trans, download=download)
    train_data, val_data = split_data(train_data, [9, 1])  # 训练集和验证集比例9：1
    return iter_data([train_data, val_data, test_data], batch_size)  # 返回数据迭代器


def chn_senti_corp(path='../data', batch_size=100):
    '''
    加载数据集ChnSentiCorp

    path: 数据集路径, 默认值'../data'

    batch_size: 批量大小, 默认值100

    返回词表和训练集、验证集、测试集迭代器
    '''
    url = 'https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/refs/heads/master/datasets/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv'
    file_name = download_file(url, path)
    chn_senti_corp = pd.read_csv(f'{path}/{file_name}')  # 读数据集
    chn_senti_corp_data = [(str(item.review), item.label) for item in chn_senti_corp.itertuples()]
    chn_senti_corp_data = MyDataset(chn_senti_corp_data)  # 生成Dataset
    train_data, val_data, test_data = split_data(chn_senti_corp_data, [0.7, 0.15, 0.15])  # 划分训练集、验证集、测试集
    train_iter, val_iter, test_iter = iter_data([train_data, val_data, test_data], batch_size)  # 产生迭代器
    tokenizer = Tokenizer(chn_senti_corp.iloc[:, 1].values, min_freq=10)  # 建立分词器
    return train_iter, val_iter, test_iter, tokenizer  # 返回迭代器和分词器


# Animator 动画器
# 动画器是类，用于绘制动画
# 动画器的返回值是动画对象
# 动画器的参数是x轴标签、y轴标签、x轴数据、y轴数据、x轴范围、y轴范围、图例
class Animator:
    """在动画中绘制数据"""

    def __init__(self, xlabel=None, ylabel=None, xlim=None, ylim=None, legend=None, fmts=None):
        self.fig, self.axes = plt.subplots()  # 生成画布
        self.set_axes = lambda: self.axes.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim)  # 初始化设置axes函数
        self.legend = legend  # 图例
        self.fmts = fmts if fmts else ('-', 'm--', 'g-.', 'r:')  # 格式

    def show(self, Y):
        '''展示动画'''
        X = [list(range(1, len(sublist) + 1)) for sublist in Y]
        self.axes.cla()  # 清除画布
        for x, y, fmt in zip(X, Y, self.fmts):
            self.axes.plot(x, y, fmt)
        self.set_axes()  # 设置axes
        if self.legend:
            self.axes.legend(self.legend)  # 设置图例
        self.axes.grid()  # 设置网格线
        display.display(self.fig)  # 画图
        display.clear_output(wait=True)  # 清除输出

    def save(self, path):
        '''保存动画'''
        self.fig.savefig(path)


def images(images, labels, shape):
    '''展示图片'''
    images = images.to(device='cpu')
    fig, axes = plt.subplots(*shape)
    axes = [element for sublist in axes for element in sublist]
    for ax, img, label in zip(axes, images, labels):
        ax.set_title(label)
        ax.set_axis_off()
        ax.imshow(img, cmap='gray')


# Accumulator 累加器
# 累加器是类，用于累加多个变量
# 累加器的返回值是累加器对象
# 累加器的参数是变量个数
class Accumulator:
    '''在n个变量上累加'''

    def __init__(self, n):
        '''初始化'''
        self.data = [0.0] * n

    def add(self, *args):
        '''添加'''
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        '''重置'''
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        '''返回第n个累加值'''
        return self.data[idx]


# Recorder 记录器
# 记录器是类，用于记录多个变量
# 记录器的返回值是记录器对象
# 记录器的参数是变量个数
class Recorder:
    '''n个记录器'''

    def __init__(self, n):
        '''初始化'''
        self.data = [[] for _ in range(n)]

    def get_latest_record(self):
        '''返回最新记录'''
        return (item[-1] for item in self.data)

    def max_record_size(self):
        '''返回最长记录长度'''
        return max((len(item) for item in self.data))

    def reset(self):
        '''重置'''
        self.data = [[] for _ in range(len(self.data))]

    def __getitem__(self, idx):
        '''返回第n个记录器'''
        return self.data[idx]

    def save(self, path, label='recorder'):
        '''保存数据'''
        DataSaveToJson.save_data(path, label, self.data)

    def load(self, path, label='recorder'):
        '''加载数据'''
        self.data = DataSaveToJson.load_data(path, label)


# Timer 计时器
# 计时器是类，用于记录多个变量
# 计时器的返回值是计时器对象
# 计时器的参数是变量个数
class Timer:
    '''记录多次运行时间'''

    def __init__(self):
        '''初始化'''
        self.times = []

    def start(self):
        '''启动计时器'''
        self.tik = time.time()

    def stop(self):
        '''停止计时器并将时间记录在列表中'''
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        '''返回平均时间'''
        if self.times:
            return sum(self.times) / len(self.times)
        else:
            return 0

    def sum(self):
        '''返回时间总和'''
        return sum(self.times)

    def save(self, path, label='timer'):
        '''保存数据'''
        DataSaveToJson.save_data(path, label, self.times)

    def load(self, path, label='timer'):
        '''加载数据'''
        self.times = DataSaveToJson.load_data(path, label)


# MachineLearning 机器学习
# 机器学习是类，用于训练模型
# 机器学习的返回值是机器学习对象
# 机器学习的参数是模型、训练集、验证集、测试集、设备
class MachineLearning:
    '''机器学习'''

    class AutoSaveLoader:
        '''自动保存加载器'''

        def __init__(self):
            '''
            初始化函数
            '''
            self.save_func = []  # 保存函数
            self.load_func = []  # 加载函数

        def add_save_func(self, func):
            '''添加保存函数'''
            self.save_func.append(func)

        def save(self, dir_path):
            '''保存数据'''
            for func in self.save_func:
                func(dir_path)

        def add_load_func(self, func):
            '''添加加载函数'''
            self.load_func.append(func)

        def load(self, dir_path):
            '''加载数据'''
            for func in self.load_func:
                func(dir_path)

    def __init__(self, file_name, **kwargs):
        '''
        初始化函数

        kwargs: 其他参数，自定义参数自动转化为属性
        '''
        # 定义时间字符串和文件名
        time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.dir_path = f'../results/{time_str}-{file_name}'
        self.file_name = file_name

        # 创建目录
        Path(self.dir_path).mkdir(parents=True, exist_ok=True)

        # 设置日志
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        # 定义日志格式
        formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
        # 创建文件处理器
        file_handler = logging.FileHandler(f'{self.dir_path}/{self.file_name}.log')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # 设置其他参数
        for key, value in kwargs.items():
            setattr(self, key, value)

        # 创建自动保存加载器
        self.data_manager = MachineLearning.AutoSaveLoader()

    def save(self, dir_name=None):
        '''保存数据'''
        dir_path = f'../results/{dir_name}' if dir_name else self.dir_path
        self.data_manager.save(dir_path)

    def load(self, dir_name=None):
        '''加载数据'''
        dir_path = f'../results/{dir_name}' if dir_name else self.dir_path
        self.data_manager.load(dir_path)

    def create_timer(self, label='timer'):
        '''创建计时器'''
        if hasattr(self, label):
            self.logger.warning(f'{label} already exists')
            return
        else:
            timer = Timer()
            setattr(self, label, timer)

        def save(dir_path):
            timer.save(f'{dir_path}/{self.file_name}.json', label)
            self.logger.debug(f'save Timer({label}) to {dir_path}/{self.file_name}.json')
        self.data_manager.add_save_func(save)

        def load(dir_path):
            timer.load(f'{dir_path}/{self.file_name}.json', label)
            self.logger.debug(f'load Timer({label}) from {dir_path}/{self.file_name}.json')
        self.data_manager.add_load_func(load)

        self.logger.debug(f'create Timer({label})')

    def create_recorder(self, recorder_num, label='recorder'):
        '''创建记录器'''
        if hasattr(self, label):
            self.logger.warning(f'{label} already exists')
            return
        else:
            recorder = Recorder(recorder_num)
            setattr(self, label, recorder)

        def save(dir_path):
            recorder.save(f'{dir_path}/{self.file_name}.json', label)
            self.logger.debug(f'save Recorder({label}) to {dir_path}/{self.file_name}.json')
        self.data_manager.add_save_func(save)

        def load(dir_path):
            recorder.load(f'{dir_path}/{self.file_name}.json', label)
            self.logger.debug(f'load Recorder({label}) from {dir_path}/{self.file_name}.json')
        self.data_manager.add_load_func(load)

        self.logger.debug(f'create Recorder({label})')

    def create_animator(self, xlabel=None, ylabel=None, xlim=None, ylim=None, legend=None, fmts=None, label='animator'):
        '''创建动画器'''
        if hasattr(self, label):
            self.logger.warning(f'{label} already exists')
            return
        else:
            animator = Animator(xlabel, ylabel, xlim, ylim, legend, fmts)
            setattr(self, label, animator)

        def save(dir_path):
            animator.save(f'{dir_path}/{self.file_name}.png')
            self.logger.debug(f'save Animator({label}) to {dir_path}/{self.file_name}.png')
        self.data_manager.add_save_func(save)

        self.logger.debug(f'create Animator({label})')

    def add_model(self, model, label='model'):
        '''添加模型保存'''
        def save(dir_path):
            torch.save(model.state_dict(), f'{dir_path}/{self.file_name}.pth')
            self.logger.debug(f'save model({label}) to {dir_path}/{self.file_name}.pth')
        self.data_manager.add_save_func(save)

        def load(dir_path):
            model.load_state_dict(torch.load(f'{dir_path}/{self.file_name}.pth'))
            self.logger.debug(f'load model({label}) from {dir_path}/{self.file_name}.pth')
        self.data_manager.add_load_func(load)

        self.logger.debug(f'add model({label})')
        self.logger.debug(f'model({label}) is {model}')


class SupervisedLearning(MachineLearning):
    '''机器学习'''

    def __init__(self, file_name, recorder_num=3, legend=['train loss', 'val loss', 'val acc'], **kwargs):
        '''
        初始化函数

        file_name: 文件名

        recorder_num: 记录器数量, 默认为空

        legend: 动画标签, 默认为空

        kwargs: 其他参数，自定义参数自动转化为属性
        '''
        MachineLearning.__init__(self, file_name, **kwargs)

        self.legend = legend  # 定义动画标签

        self.num_epochs = 0  # 定义总迭代次数

        self.create_timer()  # 创建计时器
        self.create_recorder(recorder_num)  # 创建记录器

    def trainer(self, func):
        '''
        训练装饰器

        args: 训练函数的参数

        num_epochs: 迭代次数

        kwargs: 其他参数
        '''

        def wrapper(*args, num_epochs, **kwargs):
            num_epoch = num_epochs - self.num_epochs if num_epochs > self.num_epochs else 0  # 计算迭代次数
            self.num_epochs = max(self.num_epochs, num_epochs)  # 计算总迭代次数

            self.create_animator(xlabel='epoch', xlim=[0, self.num_epochs + 1], ylim=-0.1, legend=self.legend)  # 创建动画器
            self.animator.show(self.recorder.data)  # 显示动画

            # 根据迭代次数产生日志
            if num_epoch:
                self.logger.debug(f'trained {num_epoch} epochs')
            else:
                self.logger.warning(f'num_epochs is {num_epochs}, but it is less than {self.num_epochs}, so it will not be trained')

            # 开始训练
            func(*args, num_epoch, **kwargs)

            self.logger.debug(f'total training epochs {self.num_epochs}')
            if self.timer.sum():
                self.logger.debug(f'total training time {time.strftime("%H:%M:%S", time.gmtime(self.timer.sum()))}')

            self.save()

        return wrapper

    def save(self):
        '''
        保存模型
        '''
        # 保存总迭代次数
        DataSaveToJson.save_data(f'{self.dir_path}/{self.file_name}.json', 'num_epochs', self.num_epochs)
        self.logger.debug(f'save num_epochs to {self.dir_path}/{self.file_name}.json')

        MachineLearning.save(self)

    def load(self, dir_name=None):
        '''
        加载模型

        dir_name: 模型保存文件名, 默认为最新文件名
        '''
        dir_path = f'../results/{dir_name}' if dir_name else self.dir_path

        # 加载总迭代次数
        self.num_epochs = DataSaveToJson.load_data(f'{dir_path}/{self.file_name}.json', 'num_epochs')
        self.logger.debug(f'load num_epochs from {dir_path}/{self.file_name}.json')

        MachineLearning.load(self, dir_name)
