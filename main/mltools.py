import torch
from torch.utils import data
from torchvision import transforms, datasets
import json
import time
import logging
import pandas as pd
from pathlib import Path
from IPython import display
from datetime import datetime
from collections import Counter
from matplotlib import pyplot as plt


# 数据保存器
# 定义数据保存器结构
class DataSave:
    '''数据保存器'''

    def __init__(self, data):
        self.data = data

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError


# 数据保存器
# 继承数据保存器类可以获得保存和加载数据到json文件的功能
class DataSaveToJson(DataSave):
    '''json数据保存器'''

    def save(self, path):
        '''保存数据'''
        try:
            with open(path, 'r') as file:
                data = json.load(file)
        except FileNotFoundError:
            data = {}
        with open(path, 'w') as f:
            data[self.__class__.__name__] = self.data
            json.dump(data, f, indent=4)

    def load(self, path):
        '''从json文件导入'''
        with open(path, 'r') as f:
            self.data = json.load(f)[self.__class__.__name__]


# 分词器
# 分词器是一个字典，键是词元，值是索引
# 分词器的长度是词典的大小
# 分词器的索引是词元的索引
class Tokenizer(DataSaveToJson):
    '''分词器'''

    def __init__(self, datas, min_freq=0):
        '''初始化'''
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
        DataSave.__init__(self, [self.idx_to_token, self.token_to_idx])

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
def _split_data(datas, ratio):
    '''将数据按比例随机分割'''
    ratio = [r / sum(ratio) for r in ratio]
    nums = [int(len(datas) * r) for r in ratio]
    nums[-1] = len(datas) - sum(nums[:-1])
    return data.random_split(datas, nums)


def _iter_data(datas, batch_size, shuffle=True):
    '''将批量数据转换为迭代器'''
    return (data.DataLoader(_data, batch_size=batch_size, shuffle=shuffle) for _data in datas)


def mnist(path, *, batch_size=100, download=False):
    '''加载数据集MNIST, 返回训练集、验证集、测试集迭代器'''
    trans = transforms.ToTensor()  # 数据集格式转换
    train_data = datasets.MNIST(root=path, train=True, transform=trans, download=download)
    test_data = datasets.MNIST(root=path, train=False, transform=trans, download=download)
    train_data, val_data = _split_data(train_data, [9, 1])  # 训练集和验证集比例9：1
    return _iter_data([train_data, val_data, test_data], batch_size)  # 返回数据迭代器


def chn_senti_corp(path, *, batch_size=100):
    '''加载数据集ChnSentiCorp, 返回词表和训练集、验证集、测试集迭代器'''
    # 数据集下载地址 https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/refs/heads/master/datasets/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv
    chn_senti_corp = pd.read_csv(path)  # 读数据集
    chn_senti_corp_data = [(str(item.review), item.label) for item in chn_senti_corp.itertuples()]
    chn_senti_corp_data = MyDataset(chn_senti_corp_data)  # 生成Dataset
    train_data, val_data, test_data = _split_data(chn_senti_corp_data, [0.7, 0.15, 0.15])  # 划分训练集、验证集、测试集
    train_iter, val_iter, test_iter = _iter_data([train_data, val_data, test_data], batch_size)  # 产生迭代器
    tokenizer = Tokenizer(chn_senti_corp.iloc[:, 1].values, min_freq=10)  # 建立分词器
    return train_iter, val_iter, test_iter, tokenizer  # 返回迭代器和分词器


# Animator 动画器
# 动画器是类，用于绘制动画
# 动画器的返回值是动画对象
# 动画器的参数是x轴标签、y轴标签、x轴数据、y轴数据、x轴范围、y轴范围、图例
class Animator:
    """在动画中绘制数据"""

    def __init__(self, *, xlabel=None, ylabel=None, xlim=None, ylim=None, legend=None, fmts=None):
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
class Recorder(DataSaveToJson):
    '''n个记录器'''

    def __init__(self, n):
        '''初始化'''
        self.data = [[] for _ in range(n)]
        DataSave.__init__(self, self.data)

    def add(self, *args):
        '''添加'''
        self.data = [a.append(b) for a, b in zip(self.data, args)]

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


# Timer 计时器
# 计时器是类，用于记录多个变量
# 计时器的返回值是计时器对象
# 计时器的参数是变量个数
class Timer(DataSaveToJson):
    '''记录多次运行时间'''

    def __init__(self):
        '''初始化'''
        self.times = []
        DataSave.__init__(self, self.times)

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


# MachineLearning 机器学习
# 机器学习是类，用于训练模型
# 机器学习的返回值是机器学习对象
# 机器学习的参数是模型、训练集、验证集、测试集、设备
class BaseMachineLearning:
    '''机器学习'''
    class AutoSave:
        '''自动保存'''

        def __init__(self, logger):
            '''初始化'''
            self.logger = logger  # 日志生成器
            self.items = []  # 保存项
            self.file_name = []  # 文件名
            self.can_load = []  # 能否加载

        def add(self, item, file_name, can_load=True):
            '''添加'''
            self.items.append(item)
            self.file_name.append(file_name)
            self.can_load.append(can_load)

        def save(self, dir_path):
            '''保存'''
            for item, file_name in zip(self.items, self.file_name):
                item.save(f'{dir_path}/{file_name}')
                self.logger.debug(f'save {item.__class__.__name__} to {dir_path}/{file_name}')

        def load(self, dir_path):
            '''加载'''
            for item, file_name, can_load in zip(self.items, self.file_name, self.can_load):
                if can_load:
                    if Path(f'{dir_path}/{file_name}').exists():
                        item.load(f'{dir_path}/{file_name}')
                        self.logger.debug(f'load {item.__class__.__name__} from {dir_path}/{file_name}')
                    else:
                        self.logger.warning(f'file {dir_path}/{file_name} not exists')

    def __init__(self, *, device=torch.device('cpu'), **kwargs):
        '''初始化函数'''
        # 定义时间字符串和文件名
        time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.dir_path = f'../results/{time_str}-{self.__class__.__name__}'
        self.file_name = f'{self.__class__.__name__}'

        # 创建目录
        for path in [f'../results', self.dir_path]:
            if not Path(path).exists():
                Path(path).mkdir()

        # 设置日志
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        # 创建文件处理器
        file_handler = logging.FileHandler(f'{self.dir_path}/{self.file_name}.log')
        file_handler.setLevel(logging.DEBUG)
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        # 设置日志格式
        formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        # 将处理器添加到日志记录器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        self.device = device  # 定义设备
        self.logger.debug(f'device is {self.device}')

        # 定义自动保存
        self.auto_save = self.AutoSave(self.logger)

        # 设置其他参数
        for key, value in kwargs.items():
            setattr(self, key, value)


class MachineLearning(BaseMachineLearning):
    '''机器学习'''

    def __init__(self, train_iter, val_iter, test_iter, *, model, loss, optimizer, recorder_num=3, legend=None, fmts=None, device=None, **kwargs):
        '''初始化函数'''
        BaseMachineLearning.__init__(self, device=device, **kwargs)

        # 设置训练集、验证集、测试集
        self.train_iter = train_iter
        self.val_iter = val_iter if val_iter else self.train_iter
        self.test_iter = test_iter if test_iter else self.val_iter

        model.to(device)
        self.model = model  # 设置模型
        self.logger.debug(f'model is {self.model}')

        self.loss = loss  # 设置损失函数
        self.logger.debug(f'loss function is {self.loss.__class__.__name__}')

        self.optimizer = optimizer  # 设置优化器
        self.logger.debug(f'optimizer is {self.optimizer.__class__.__name__}, learning rate is {self.optimizer.param_groups[0]["lr"]}')

        self.legend = legend  # 定义动画标签
        self.fmts = fmts  # 定义动画格式

        self.num_epochs = 0  # 定义总迭代次数

        self.timer = Timer()  # 设置计时器
        self.auto_save.add(self.timer, f'{self.file_name}.json')  # 自动保存计时器
        self.recorder = Recorder(recorder_num)  # 设置记录器
        self.auto_save.add(self.recorder, f'{self.file_name}.json')  # 自动保存记录器

    def trainer(func):
        '''训练装饰器'''

        def wrapper(self, *args, num_epochs, **kwargs):
            num_epoch = num_epochs - self.num_epochs if num_epochs > self.num_epochs else 0  # 计算迭代次数
            self.num_epochs = max(self.num_epochs, num_epochs)  # 计算总迭代次数

            # 初始化动画器
            self.animator = Animator(xlabel='epoch', xlim=[0, self.num_epochs + 1], ylim=-0.1, legend=self.legend, fmts=self.fmts)
            self.auto_save.add(self.animator, f'{self.file_name}.png', can_load=False)  # 自动保存动画
            self.animator.show(self.recorder.data)

            # 根据迭代次数产生日志
            if num_epoch:
                self.logger.debug(f'trained {num_epoch} times')
            else:
                self.logger.warning(f'num_epochs is {num_epochs}, but it is less than {self.num_epochs}, so it will not be trained')

            # 开始训练
            func(self, *args, num_epoch, **kwargs)

            # 自动保存
            self.auto_save.save(self.dir_path)

            # 保存模型参数
            model_parameters = {
                'num_epochs': self.num_epochs,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': self.loss,
            }
            torch.save(model_parameters, f'{self.dir_path}/{self.file_name}.pth')
            self.logger.debug(f'save model parameters to {self.dir_path}/{self.file_name}.pth')
        return wrapper

    def load(self, dir_name=None):
        '''加载模型'''
        dir_path = f'../results/{dir_name}' if dir_name else self.dir_path

        # 加载自动保存
        self.auto_save.load(dir_path)

        # 加载模型参数
        if Path(f'{dir_path}/{self.file_name}.pth').exists():
            model_parameters = torch.load(f'{dir_path}/{self.file_name}.pth', weights_only=False)
            self.model.load_state_dict(model_parameters['model_state_dict'])
            self.optimizer.load_state_dict(model_parameters['optimizer_state_dict'])
            self.num_epochs = model_parameters['num_epochs']
            self.loss = model_parameters['loss']
            self.logger.debug(f'load model parameters from {dir_path}/{self.file_name}.pth')
        else:
            self.logger.warning(f'file {dir_path}/{self.file_name}.pth not exists')

    def tester(func):
        '''测试装饰器'''

        def wrapper(self, *args, **kwargs):
            self.model.eval()  # 验证模式
            func(self, *args, **kwargs)
        return wrapper

    def predictor(func):
        '''预测装饰器'''

        def wrapper(self, *args, **kwargs):
            self.model.eval()  # 验证模式
            func(self, *args, **kwargs)
        return wrapper
