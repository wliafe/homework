import torch
from torch import nn
from torch.utils import data
from torchvision import transforms, datasets
import time
import math
import logging
import pandas as pd
from pathlib import Path
from IPython import display
from datetime import datetime
from collections import Counter
from matplotlib import pyplot as plt


# vocab 词元表，用于构建词典
# 词元表是一个字典，键是词元，值是索引
# 词元表的长度是词典的大小
# 词元表的索引是词元的索引
class Vocab:
    '''词元表'''

    def __init__(self, tokens, min_freq=0, reserved_tokens=None):
        '''初始化'''
        if not reserved_tokens:  # 保留词元
            reserved_tokens = []
        self.unk = 0  # 未知词元索引为0
        tokens = [item[0] for item in tokens.items() if item[1] > min_freq]  # 删除低频词元
        self.idx_to_token = ['<unk>']+reserved_tokens+tokens  # 建立词元列表
        # 建立词元字典
        reserved_tokens_dict = {value: index+1 for index, value in enumerate(reserved_tokens)}
        tokens_dict = {value: index+1+len(reserved_tokens_dict) for index, value in enumerate(tokens)}
        self.token_to_idx = {'<unk>': 0}
        self.token_to_idx.update(reserved_tokens_dict)
        self.token_to_idx.update(tokens_dict)

    def __len__(self):
        '''返回词表大小'''
        return len(self.idx_to_token)

    def __getitem__(self, indices):
        '''根据索引返回词元'''
        if isinstance(indices, (list, tuple)):
            return [self.__getitem__(index) for index in indices]
        return self.idx_to_token[indices]

    def to_indices(self, tokens):
        '''根据词元返回索引'''
        if isinstance(tokens, (list, tuple)):
            return [self.to_indices(token) for token in tokens]
        return self.token_to_idx.get(tokens, self.unk)


# load_data 多个数据集加载器
# 数据集加载器是函数，用于加载数据集
# 数据集加载器的返回值是训练集、验证集、测试集的迭代器
# 数据集加载器的参数是数据集的路径、批量大小、是否下载数据集
def _split_data(datas, ratio):
    '''将数据按比例随机分割'''
    ratio = [r/sum(ratio) for r in ratio]
    nums = [int(len(datas)*r) for r in ratio]
    nums[-1] = len(datas)-sum(nums[:-1])
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


def chn_senti_corp(path, *, batch_size=100, step_size=200):
    '''加载数据集ChnSentiCorp, 返回词表和训练集、验证集、测试集迭代器'''
    # 数据集下载地址 https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/refs/heads/master/datasets/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv
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


# 训练模型
# 训练模型是类
# 定义多种训练模型，适用于不同的训练任务
class RNNPModel(nn.Module):
    '''循环神经网络预测模型'''

    def __init__(self, embedding, rnn_layer, output_size, *args, **kwargs):
        '''初始化函数'''
        nn.Module.__init__(self, *args, **kwargs)
        self.embedding = embedding  # 定义嵌入层
        self.hidden_layer = rnn_layer  # 定义隐藏层
        self.output_layer = nn.Linear(rnn_layer.hidden_size, output_size)  # 定义输出层

    def forward(self, x, state=None):
        '''前向传播'''
        x = self.embedding(x)  # 将输入嵌入, x形状为(批量大小, 时间步数, 嵌入大小)
        x = x.to(torch.float32)
        x, state = self.hidden_layer(x, state)  # x形状为(批量大小, 时间步数, 隐藏大小), state形状为(隐藏层数, 批量大小, 隐藏大小)
        x = self.output_layer(x)  # 它的输出形状是(批量大小, 时间步数, 输出大小)
        x = x.permute(0, 2, 1)  # 交换时间步数和输出大小的维度, x形状为(批量大小, 输出大小, 时间步数)
        return x, state


class RNNCModel(nn.Module):
    '''循环神经网络分类模型'''

    def __init__(self, embedding, rnn_layer, output_size, *args, **kwargs):
        '''初始化函数'''
        nn.Module.__init__(self, *args, **kwargs)
        self.embedding = embedding  # 定义嵌入层
        self.hidden_layer = rnn_layer  # 定义隐藏层
        hidden_size = rnn_layer.hidden_size  # 定义时间步数和隐藏大小
        directions = 1 if not rnn_layer.bidirectional else 2  # 如果RNN是双向的, num_directions应该是2, 否则应该是1
        self.output_layer = nn.Linear(hidden_size * directions, output_size)  # 定义输出层

    def forward(self, x, state=None):
        '''前向传播'''
        x = self.embedding(x)  # 将输入嵌入, x形状为(批量大小, 时间步数, 嵌入大小)
        x = x.to(torch.float32)
        x, state = self.hidden_layer(x, state)  # x形状为(批量大小, 时间步数, 隐藏大小), state形状为(隐藏层数 * directions, 批量大小, 隐藏大小)
        x = x.mean(dim=1)  # 在时间步数维度取平均, x形状为(批量大小, 隐藏大小)
        x = self.output_layer(x)  # 它的输出形状是(批量大小, 输出大小)
        return x, state


class TransformerCModel(nn.Module):
    '''Transformer 分类模型'''

    def __init__(self, embedding, transformer_layer, input_size, output_size, *args, **kwargs):
        '''初始化函数'''
        nn.Module.__init__(self, *args, **kwargs)
        self.embedding = embedding  # 定义嵌入层
        self.hidden_layer = transformer_layer  # 定义隐藏层
        self.output_layer = nn.Linear(input_size, output_size)  # 定义输出层

    def forward(self, x):
        '''前向传播'''
        x = self.embedding(x)  # 将输入嵌入, x形状为(批量大小, 时间步数, 嵌入大小)
        x = self.hidden_layer(x)  # x形状为(批量大小, 时间步数, 隐藏大小)
        x = self.output_layer(x[:, 0, :])  # 它的输出形状是(批量大小, 输出大小)
        return x


# Animator 动画器
# 动画器是类，用于绘制动画
# 动画器的返回值是动画对象
# 动画器的参数是x轴标签、y轴标签、x轴数据、y轴数据、x轴范围、y轴范围
class Animator:
    """在动画中绘制数据"""

    def __init__(self, *, xlabel=None, ylabel=None, xlim=None, ylim=None):
        self.fig, self.axes = plt.subplots()  # 生成画布
        self.set_axes = lambda: self.axes.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim)  # 初始化设置axes函数

    def show(self, Y, legend=None):
        '''展示动画'''
        X = [list(range(1, len(sublist)+1)) for sublist in Y]
        self.axes.cla()  # 清除画布
        for x, y, fmt in zip(X, Y, ('-', 'm--', 'g-.', 'r:')):
            self.axes.plot(x, y, fmt)
        self.set_axes()  # 设置axes
        if legend:
            self.axes.legend(legend)  # 设置标签
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
        return sum(self.times) / len(self.times)

    def sum(self):
        '''返回时间总和'''
        return sum(self.times)


# MachineLearning 机器学习
# 机器学习是类，用于训练模型
# 机器学习的返回值是机器学习对象
# 机器学习的参数是模型、训练集、验证集、测试集、设备
class MachineLearning:
    '''机器学习'''

    def __init__(self, model, train_iter, val_iter, test_iter, *, recorder_num=3, device=torch.device('cpu')):
        '''初始化函数'''
        model.to(device)  # 将网络复制到device上
        self.model, self.train_iter, self.device = model, train_iter, device
        self.val_iter = val_iter if val_iter else self.train_iter  # 定义验证集
        self.test_iter = test_iter if test_iter else self.val_iter  # 定义测试集
        self.time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")  # 定义时间字符串
        self.file_name = f'../results/{self.time_str}-{self.__class__.__name__}/{self.time_str}-{self.__class__.__name__}'
        self._make_directories([f'../results', f'../results/{self.time_str}-{self.__class__.__name__}'])  # 创建目录
        self.set_timer()  # 设置计时器
        self.set_recorder(recorder_num)  # 设置记录器
        self.set_logger()  # 设置日志

    def set_timer(self):
        '''设置计时器'''
        self.timer = Timer()

    def set_recorder(self, recorder_num):
        '''设置记录器'''
        self.recorder = Recorder(recorder_num)

    def set_logger(self):
        '''设置日志'''
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        # 创建文件处理器
        file_handler = logging.FileHandler(f'{self.file_name}.log')
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

    def set_loss(self, loss):
        '''设置损失函数'''
        self.loss = loss
        self.logger.debug(f'loss function is {self.loss.__class__.__name__}')

    def set_optimizer(self, optimizer):
        '''设置优化器'''
        self.optimizer = optimizer
        self.logger.debug(f'optimizer is {self.optimizer.__class__.__name__}, learning rate is {self.optimizer.param_groups[0]["lr"]}')

    def train(self, num_epochs):
        '''训练模型'''
        rlim = num_epochs + self.recorder.max_record_size()  # 计算xlim的右边界
        self.animator = Animator(xlabel='epoch', xlim=[0, rlim + 1], ylim=-0.1)
        self.logger.debug(f'num_epochs is {num_epochs}')
        self.train_epochs(num_epochs)
        self.animator.save(f'{self.file_name}.png')
        torch.save(self.model.state_dict(), f'{self.file_name}.pth')

    def train_epochs(self, num_epochs):
        '''一个迭代周期'''
        raise NotImplementedError

    def test(self):
        '''测试模型'''
        raise NotImplementedError

    def predict(self):
        '''预测模型'''
        raise NotImplementedError

    def _make_directories(self, paths):
        '''创建目录'''
        if not isinstance(paths, list):
            paths = [paths]
        for path in paths:
            if not Path(path).exists():
                Path(path).mkdir()


class CNNLearning(MachineLearning):
    '''简单学习'''

    def train_epochs(self, num_epochs):
        '''迭代训练'''
        for num_epoch in range(1, num_epochs + 1):
            self.timer.start()

            # 计算训练集
            metric_train = Accumulator(2)  # 累加器：(train_loss, train_size)
            self.model.train()  # 训练模式
            for x, y in self.train_iter:
                x = x.to(self.device)  # 转换x
                y = y.to(self.device)  # 转换y
                y_train = self.model(x)  # 计算模型
                train_loss = self.loss(y_train, y)  # 计算训练损失

                # 梯度更新
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()

                metric_train.add(train_loss * len(y), len(y))
            self.recorder[0].append(metric_train[0] / metric_train[1])

            self.timer.stop()

            # 计算验证集
            metric_val = Accumulator(3)  # 累加器：(val_loss, val_acc, val_size)
            self.model.eval()  # 验证模式
            with torch.no_grad():
                for x, y in self.val_iter:
                    x = x.to(self.device)  # 转换x
                    y = y.to(self.device)  # 转换y
                    y_val = self.model(x)  # 计算模型
                    val_loss = self.loss(y_val, y)  # 计算验证损失
                    val_pred = y_val.argmax(dim=1)  # 计算预测值
                    val_acc = (val_pred == y).sum()  # 计算验证准确率
                    metric_val.add(val_loss * len(y), val_acc, len(y))
            self.recorder[1].append(metric_val[0] / metric_val[2])
            self.recorder[2].append(metric_val[1] / metric_val[2])

            # 打印输出值
            self.logger.info(f'train loss {self.recorder[0][-1]:.3f}, val loss {self.recorder[1][-1]:.3f}, val acc {self.recorder[2][-1]:.3f}, {self.timer.avg():.1f} sec/epoch on {str(self.device)}')
            self.animator.show(self.recorder.data, legend=['train loss', 'val loss', 'val acc'])  # 添加损失值
        else:
            # 打印输出值
            self.logger.info(f'train loss {self.recorder[0][-1]:.3f}, val loss {self.recorder[1][-1]:.3f}, val acc {self.recorder[2][-1]:.3f}, {self.timer.avg():.1f} sec/epoch on {str(self.device)}')

    def test(self):
        '''测试模型'''
        metric = Accumulator(2)  # 累加器：(test_acc, test_size)
        self.model.eval()  # 验证模式
        with torch.no_grad():
            for x, y in self.test_iter:
                x = x.to(self.device)  # 转换x
                y = y.to(self.device)  # 转换y
                y_test = self.model(x)  # 计算模型
                test_pred = y_test.argmax(dim=1)  # 计算准确率
                test_acc = (test_pred == y).sum()  # 计算测试准确率
                metric.add(test_acc, len(y))
        self.logger.info(f'test acc {metric[0] / metric[1]:.3f}')  # 计算测试准确率并输出

    def predict(self):
        '''预测模型'''
        self.model.eval()  # 验证模式
        x, y = next(iter(self.test_iter))  # 从测试中取一个批量
        x = x[:10].to(self.device)
        y = y[:10].to(self.device)
        y_pred = self.model(x)  # 计算模型
        y_pred = y_pred.argmax(dim=1)  # 计算预测
        self.show_pred(x, y_pred, y)  # 显示预测

    def show_pred(self, contents, preds, reals):
        '''显示预测'''
        raise NotImplementedError


class RNNPLearning(MachineLearning):
    '''RNN 预测学习'''

    def __init__(self, model, train_iter, *, device=torch.device('cpu')):
        MachineLearning.__init__(self, model, train_iter, None, None, recorder_num=1, device=device)

    def train_epochs(self, num_epochs):
        '''迭代训练'''
        for num_epoch in range(1, num_epochs + 1):
            self.timer.start()

            # 计算训练集
            metric_train = Accumulator(2)  # 累加器：(train_loss, train_size)
            self.model.train()  # 训练模式
            for x, y in self.train_iter:
                x = x.to(self.device)  # 转换x
                y = y.to(self.device)  # 转换y
                y_train, _ = self.model(x)  # 计算模型
                train_loss = self.loss(y_train, y)  # 计算训练损失

                # 梯度更新
                self.optimizer.zero_grad()
                train_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1, norm_type=2)
                self.optimizer.step()

                metric_train.add(train_loss * len(y), len(y))
            self.recorder[0].append(math.exp(metric_train[0] / metric_train[1]))

            self.timer.stop()

            # 打印输出值
            self.logger.info(f'train perplexity {self.recorder[0][-1]:.3f}, {self.timer.avg():.1f} sec/epoch on {str(self.device)}')
            self.animator.show(self.recorder.data, legend=['train perplexity'])  # 添加困惑度
        else:
            # 打印输出值
            self.logger.info(f'train perplexity {self.recorder[0][-1]:.3f}, {self.timer.avg():.1f} sec/epoch on {str(self.device)}')

    def test(self):
        '''测试模型'''
        metric = Accumulator(2)  # 累加器：(test_acc, test_size)
        self.model.eval()  # 验证模式
        with torch.no_grad():
            for x, y in self.test_iter:
                x = x.to(self.device)  # 转换x
                y = y.to(self.device)  # 转换y
                y_test, _ = self.model(x)  # 计算模型
                test_pred = y_test.argmax(dim=1)  # 计算准确率
                test_acc = (test_pred == y).sum()  # 计算测试准确率
                metric.add(test_acc, len(y))
        self.logger.info(f'test acc {metric[0] / metric[1]:.3f}')  # 计算测试准确率并输出


class RNNTXTCLearning(MachineLearning):
    '''RNN 文本分类学习'''

    def train_epochs(self, num_epochs):
        '''迭代训练'''
        for num_epoch in range(1, num_epochs + 1):
            self.timer.start()

            # 计算训练集
            metric_train = Accumulator(2)  # 累加器：(train_loss, train_size)
            self.model.train()  # 训练模式
            for x, y in self.train_iter:
                x = x.to(self.device)  # 转换x
                y = y.to(self.device)  # 转换y
                y_train, _ = self.model(x)  # 计算模型
                train_loss = self.loss(y_train, y)  # 计算训练损失

                # 梯度更新
                self.optimizer.zero_grad()
                train_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1, norm_type=2)
                self.optimizer.step()

                metric_train.add(train_loss * len(y), len(y))
            self.recorder[0].append(metric_train[0] / metric_train[1])

            self.timer.stop()

            # 计算验证集
            metric_val = Accumulator(3)  # 累加器：(val_loss, val_acc, val_size)
            self.model.eval()  # 验证模式
            with torch.no_grad():
                for x, y in self.val_iter:
                    x = x.to(self.device)  # 转换x
                    y = y.to(self.device)  # 转换y
                    y_val, _ = self.model(x)  # 计算模型
                    val_loss = self.loss(y_val, y)  # 计算验证损失
                    val_pred = y_val.argmax(dim=1)  # 计算预测值
                    val_acc = (val_pred == y).sum()  # 计算验证准确率
                    metric_val.add(val_loss * len(y), val_acc, len(y))
            self.recorder[1].append(metric_val[0] / metric_val[2])
            self.recorder[2].append(metric_val[1] / metric_val[2])

            # 打印输出值
            self.logger.info(f'train loss {self.recorder[0][-1]:.3f}, val loss {self.recorder[1][-1]:.3f}, val acc {self.recorder[2][-1]:.3f}, {self.timer.avg():.1f} sec/epoch on {str(self.device)}')
            self.animator.show(self.recorder.data, legend=['train loss', 'val loss', 'val acc'])  # 添加损失值
        else:
            # 打印输出值
            self.logger.info(f'train loss {self.recorder[0][-1]:.3f}, val loss {self.recorder[1][-1]:.3f}, val acc {self.recorder[2][-1]:.3f}, {self.timer.avg():.1f} sec/epoch on {str(self.device)}')

    def test(self):
        '''测试模型'''
        metric = Accumulator(2)  # 累加器：(test_acc, test_size)
        self.model.eval()  # 验证模式
        with torch.no_grad():
            for x, y in self.test_iter:
                x = x.to(self.device)  # 转换x
                y = y.to(self.device)  # 转换y
                y_test, _ = self.model(x)  # 计算模型
                test_pred = y_test.argmax(dim=1)  # 计算准确率
                test_acc = (test_pred == y).sum()  # 计算测试准确率
                metric.add(test_acc, len(y))
        self.logger.info(f'test acc {metric[0] / metric[1]:.3f}')  # 计算测试准确率并输出

    def predict(self):
        '''预测模型'''
        self.model.eval()  # 验证模式
        x, y = next(iter(self.test_iter))  # 从测试中取一个批量
        x = x[:10].to(self.device)
        y = y[:10].to(self.device)
        y_pred, _ = self.model(x)  # 计算模型
        y_pred = y_pred.argmax(dim=1)  # 计算预测
        self.show_pred(x, y_pred, y)  # 显示预测

    def show_pred(self, contents, preds, reals):
        '''显示预测'''
        raise NotImplementedError


class RNNIMGCLearning(MachineLearning):
    '''RNN 图片分类学习'''

    def train_epochs(self, num_epochs):
        '''迭代训练'''
        for num_epoch in range(1, num_epochs + 1):
            self.timer.start()

            # 计算训练集
            metric_train = Accumulator(2)  # 累加器：(train_loss, train_size)
            self.model.train()  # 训练模式
            for x, y in self.train_iter:
                x = x.to(self.device)  # 转换x
                x = x.permute(0, 2, 1, 3)  # 交换图片通道数和图片高度的维度, x形状为(批量大小, 图片高度, 图片通道数, 图片宽度)
                x = x.reshape(x.shape[0], x.shape[1], -1)  # 将图片展平, x形状为(批量大小, 图片高度, 图片通道数 * 图片宽度)
                y = y.to(self.device)  # 转换y
                y_train, _ = self.model(x)  # 计算模型
                train_loss = self.loss(y_train, y)  # 计算训练损失

                # 梯度更新
                self.optimizer.zero_grad()
                train_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1, norm_type=2)
                self.optimizer.step()

                metric_train.add(train_loss * len(y), len(y))
            self.recorder[0].append(metric_train[0] / metric_train[1])

            self.timer.stop()

            # 计算验证集
            metric_val = Accumulator(3)  # 累加器：(val_loss, val_acc, val_size)
            self.model.eval()  # 验证模式
            with torch.no_grad():
                for x, y in self.val_iter:
                    x = x.to(self.device)  # 转换x
                    x = x.permute(0, 2, 1, 3)  # 交换图片通道数和图片高度的维度, x形状为(批量大小, 图片高度, 图片通道数, 图片宽度)
                    x = x.reshape(x.shape[0], x.shape[1], -1)  # 将图片展平, x形状为(批量大小, 图片高度, 图片通道数 * 图片宽度)
                    y = y.to(self.device)  # 转换y
                    y_val, _ = self.model(x)  # 计算模型
                    val_loss = self.loss(y_val, y)  # 计算验证损失
                    val_pred = y_val.argmax(dim=1)  # 计算预测值
                    val_acc = (val_pred == y).sum()  # 计算验证准确率
                    metric_val.add(val_loss * len(y), val_acc, len(y))
            self.recorder[1].append(metric_val[0] / metric_val[2])
            self.recorder[2].append(metric_val[1] / metric_val[2])

            # 打印输出值
            self.logger.info(f'train loss {self.recorder[0][-1]:.3f}, val loss {self.recorder[1][-1]:.3f}, val acc {self.recorder[2][-1]:.3f}, {self.timer.avg():.1f} sec/epoch on {str(self.device)}')
            self.animator.show(self.recorder.data, legend=['train loss', 'val loss', 'val acc'])  # 添加损失值
        else:
            # 打印输出值
            self.logger.info(f'train loss {self.recorder[0][-1]:.3f}, val loss {self.recorder[1][-1]:.3f}, val acc {self.recorder[2][-1]:.3f}, {self.timer.avg():.1f} sec/epoch on {str(self.device)}')

    def test(self):
        '''测试模型'''
        metric = Accumulator(2)  # 累加器：(test_acc, test_size)
        self.model.eval()  # 验证模式
        with torch.no_grad():
            for x, y in self.test_iter:
                x = x.to(self.device)  # 转换x
                x = x.permute(0, 2, 1, 3)  # 交换图片通道数和图片高度的维度, x形状为(批量大小, 图片高度, 图片通道数, 图片宽度)
                x = x.reshape(x.shape[0], x.shape[1], -1)  # 将图片展平, x形状为(批量大小, 图片高度, 图片通道数 * 图片宽度)
                y = y.to(self.device)  # 转换y
                y_test, _ = self.model(x)  # 计算模型
                test_pred = y_test.argmax(dim=1)  # 计算准确率
                test_acc = (test_pred == y).sum()  # 计算测试准确率
                metric.add(test_acc, len(y))
        self.logger.info(f'test acc {metric[0] / metric[1]:.3f}')  # 计算测试准确率并输出

    def predict(self):
        '''预测模型'''
        self.model.eval()  # 验证模式
        x, y = next(iter(self.test_iter))  # 从测试中取一个批量
        x = x[:10].to(self.device)
        x = x.permute(0, 2, 1, 3)  # 交换图片通道数和图片高度的维度, x形状为(批量大小, 图片高度, 图片通道数, 图片宽度)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # 将图片展平, x形状为(批量大小, 图片高度, 图片通道数 * 图片宽度)
        y = y[:10].to(self.device)
        y_pred, _ = self.model(x)  # 计算模型
        y_pred = y_pred.argmax(dim=1)  # 计算预测
        self.show_pred(x, y_pred, y)  # 显示预测

    def show_pred(self, contents, preds, reals):
        '''显示预测'''
        raise NotImplementedError


class TransformerTXTCLearning(MachineLearning):
    '''Transformer 文本分类学习'''

    def train_epochs(self, num_epochs):
        '''迭代训练'''
        for num_epoch in range(1, num_epochs + 1):
            self.timer.start()

            # 计算训练集
            metric_train = Accumulator(2)  # 累加器：(train_loss, train_size)
            self.model.train()  # 训练模式
            for x, y in self.train_iter:
                x = x.to(self.device)  # 转换x
                y = y.to(self.device)  # 转换y
                y_train = self.model(x)  # 计算模型
                train_loss = self.loss(y_train, y)  # 计算训练损失

                # 梯度更新
                self.optimizer.zero_grad()
                train_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1, norm_type=2)
                self.optimizer.step()

                metric_train.add(train_loss * len(y), len(y))
            self.recorder[0].append(metric_train[0] / metric_train[1])

            self.timer.stop()

            # 计算验证集
            metric_val = Accumulator(3)  # 累加器：(val_loss, val_acc, val_size)
            self.model.eval()  # 验证模式
            with torch.no_grad():
                for x, y in self.val_iter:
                    x = x.to(self.device)  # 转换x
                    y = y.to(self.device)  # 转换y
                    y_val = self.model(x)  # 计算模型
                    val_loss = self.loss(y_val, y)  # 计算验证损失
                    val_pred = y_val.argmax(dim=1)  # 计算预测值
                    val_acc = (val_pred == y).sum()  # 计算验证准确率
                    metric_val.add(val_loss * len(y), val_acc, len(y))
            self.recorder[1].append(metric_val[0] / metric_val[2])
            self.recorder[2].append(metric_val[1] / metric_val[2])

            # 打印输出值
            self.logger.info(f'train loss {self.recorder[0][-1]:.3f}, val loss {self.recorder[1][-1]:.3f}, val acc {self.recorder[2][-1]:.3f}, {self.timer.avg():.1f} sec/epoch on {str(self.device)}')
            self.animator.show(self.recorder.data, legend=['train loss', 'val loss', 'val acc'])  # 添加损失值
        else:
            # 打印输出值
            self.logger.info(f'train loss {self.recorder[0][-1]:.3f}, val loss {self.recorder[1][-1]:.3f}, val acc {self.recorder[2][-1]:.3f}, {self.timer.avg():.1f} sec/epoch on {str(self.device)}')

    def test(self):
        '''测试模型'''
        metric = Accumulator(2)  # 累加器：(test_acc, test_size)
        self.model.eval()  # 验证模式
        with torch.no_grad():
            for x, y in self.test_iter:
                x = x.to(self.device)  # 转换x
                y = y.to(self.device)  # 转换y
                y_test = self.model(x)  # 计算模型
                test_pred = y_test.argmax(dim=1)  # 计算准确率
                test_acc = (test_pred == y).sum()  # 计算测试准确率
                metric.add(test_acc, len(y))
        self.logger.info(f'test acc {metric[0] / metric[1]:.3f}')  # 计算测试准确率并输出

    def predict(self):
        '''预测模型'''
        self.model.eval()  # 验证模式
        x, y = next(iter(self.test_iter))  # 从测试中取一个批量
        x = x[:10].to(self.device)
        y = y[:10].to(self.device)
        y_pred = self.model(x)  # 计算模型
        y_pred = y_pred.argmax(dim=1)  # 计算预测
        self.show_pred(x, y_pred, y)  # 显示预测

    def show_pred(self, contents, preds, reals):
        '''显示预测'''
        raise NotImplementedError


class TransformerIMGCLearning(MachineLearning):
    '''Transformer 图片分类学习'''

    def train_epochs(self, num_epochs):
        '''迭代训练'''
        for num_epoch in range(1, num_epochs + 1):
            self.timer.start()

            # 计算训练集
            metric_train = Accumulator(2)  # 累加器：(train_loss, train_size)
            self.model.train()  # 训练模式
            for x, y in self.train_iter:
                x = x.to(self.device)  # 转换x
                x = x.permute(0, 2, 1, 3)  # 交换图片通道数和图片高度的维度, x形状为(批量大小, 图片高度, 图片通道数, 图片宽度)
                x = x.reshape(x.shape[0], x.shape[1], -1)  # 将图片展平, x形状为(批量大小, 图片高度, 图片通道数 * 图片宽度)
                y = y.to(self.device)  # 转换y
                y_train = self.model(x)  # 计算模型
                train_loss = self.loss(y_train, y)  # 计算训练损失

                # 梯度更新
                self.optimizer.zero_grad()
                train_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1, norm_type=2)
                self.optimizer.step()

                metric_train.add(train_loss * len(y), len(y))
            self.recorder[0].append(metric_train[0] / metric_train[1])

            self.timer.stop()

            # 计算验证集
            metric_val = Accumulator(3)  # 累加器：(val_loss, val_acc, val_size)
            self.model.eval()  # 验证模式
            with torch.no_grad():
                for x, y in self.val_iter:
                    x = x.to(self.device)  # 转换x
                    x = x.permute(0, 2, 1, 3)  # 交换图片通道数和图片高度的维度, x形状为(批量大小, 图片高度, 图片通道数, 图片宽度)
                    x = x.reshape(x.shape[0], x.shape[1], -1)  # 将图片展平, x形状为(批量大小, 图片高度, 图片通道数 * 图片宽度)
                    y = y.to(self.device)  # 转换y
                    y_val = self.model(x)  # 计算模型
                    val_loss = self.loss(y_val, y)  # 计算验证损失
                    val_pred = y_val.argmax(dim=1)  # 计算预测值
                    val_acc = (val_pred == y).sum()  # 计算验证准确率
                    metric_val.add(val_loss * len(y), val_acc, len(y))
            self.recorder[1].append(metric_val[0] / metric_val[2])
            self.recorder[2].append(metric_val[1] / metric_val[2])

            # 打印输出值
            self.logger.info(f'train loss {self.recorder[0][-1]:.3f}, val loss {self.recorder[1][-1]:.3f}, val acc {self.recorder[2][-1]:.3f}, {self.timer.avg():.1f} sec/epoch on {str(self.device)}')
            self.animator.show(self.recorder.data, legend=['train loss', 'val loss', 'val acc'])  # 添加损失值
        else:
            # 打印输出值
            self.logger.info(f'train loss {self.recorder[0][-1]:.3f}, val loss {self.recorder[1][-1]:.3f}, val acc {self.recorder[2][-1]:.3f}, {self.timer.avg():.1f} sec/epoch on {str(self.device)}')

    def test(self):
        '''测试模型'''
        metric = Accumulator(2)  # 累加器：(test_acc, test_size)
        self.model.eval()  # 验证模式
        with torch.no_grad():
            for x, y in self.test_iter:
                x = x.to(self.device)  # 转换x
                x = x.permute(0, 2, 1, 3)  # 交换图片通道数和图片高度的维度, x形状为(批量大小, 图片高度, 图片通道数, 图片宽度)
                x = x.reshape(x.shape[0], x.shape[1], -1)  # 将图片展平, x形状为(批量大小, 图片高度, 图片通道数 * 图片宽度)
                y = y.to(self.device)  # 转换y
                y_test = self.model(x)  # 计算模型
                test_pred = y_test.argmax(dim=1)  # 计算准确率
                test_acc = (test_pred == y).sum()  # 计算测试准确率
                metric.add(test_acc, len(y))
        self.logger.info(f'test acc {metric[0] / metric[1]:.3f}')  # 计算测试准确率并输出

    def predict(self):
        '''预测模型'''
        self.model.eval()  # 验证模式
        x, y = next(iter(self.test_iter))  # 从测试中取一个批量
        x = x[:10].to(self.device)
        x = x.permute(0, 2, 1, 3)  # 交换图片通道数和图片高度的维度, x形状为(批量大小, 图片高度, 图片通道数, 图片宽度)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # 将图片展平, x形状为(批量大小, 图片高度, 图片通道数 * 图片宽度)
        y = y[:10].to(self.device)
        y_pred = self.model(x)  # 计算模型
        y_pred = y_pred.argmax(dim=1)  # 计算预测
        self.show_pred(x, y_pred, y)  # 显示预测

    def show_pred(self, contents, preds, reals):
        '''显示预测'''
        raise NotImplementedError
