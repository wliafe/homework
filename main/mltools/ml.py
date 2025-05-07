import torch
from torch import nn, optim
from torch.utils import data
from .draw import Animator
from .timer import Timer
from .accumulator import Accumulator


class MachineLearning:
    def __init__(self, model, train_iter, val_iter=None, test_iter=None, *, device=torch.device('cpu')):
        '''初始化函数'''
        model.to(device)  # 将网络复制到device上
        self.model, self.train_iter, self.device = model, train_iter, device
        self.val_iter = val_iter if val_iter else self.train_iter  # 定义验证集
        self.test_iter = test_iter if test_iter else self.val_iter  # 定义测试集

    def train(self, num_epochs, learning_rate):
        '''训练模型'''
        self.set_timer()  # 设置计时器
        self.set_loss()  # 设置损失函数
        self.set_optimizer(learning_rate)  # 定义优化器
        self.set_animator(num_epochs)  # 设置Animator
        for self.num_epoch in range(1, num_epochs+1):
            self.train_epoch()
        else:
            self.output_print()

    def set_timer(self):
        self.timer = Timer()  # 设置计时器

    def set_loss(self):
        '''设置损失函数'''
        self.loss = nn.CrossEntropyLoss()  # 定义损失函数

    def set_optimizer(self, learning_rate):
        '''设置优化器'''
        self.optimizer = optim.SGD(self.model.parameters(), learning_rate)  # 定义优化器

    def set_animator(self, num_epochs):
        '''设置Animator'''
        self.animator = Animator(line_num=3, xlabel='epoch', xlim=[0, num_epochs+1], ylim=-0.1, legend=['train loss', 'val loss', 'val acc'])

    def train_epoch(self):
        '''一个迭代周期'''
        self.timer.start()
        self.calculate_train_iter()  # 计算训练集
        self.timer.stop()
        self.calculate_val_iter()  # 计算验证集
        self.output_print()
        self.animator.add(self.train_loss, self.val_loss, self.val_acc)  # 添加损失值

    def calculate_train_iter(self):
        '''计算训练集'''
        metric = Accumulator(2)  # 累加器：(train_loss, train_size)
        self.model.train()  # 训练模式
        for x, y in self.train_iter:
            x = self.transform_x(x)  # 转换x
            y_train = self.calculate_model(x)  # 计算模型
            y = self.transform_y(y)  # 转换y
            train_loss = self.calculate_loss(y_train, y)  # 计算训练损失
            self.grad_update(train_loss)  # 梯度更新
            metric.add(train_loss * len(y), len(y))
        self.train_loss = metric[0] / metric[1]

    def calculate_val_iter(self):
        '''计算验证集'''
        metric = Accumulator(3)  # 累加器：(val_loss, val_acc, val_size)
        self.model.eval()  # 验证模式
        with torch.no_grad():
            for x, y in self.val_iter:
                x = self.transform_x(x)  # 转换x
                y_val = self.calculate_model(x)  # 计算模型
                y = self.transform_y(y)  # 转换y
                val_loss = self.calculate_loss(y_val, y)  # 计算验证损失
                val_acc = self.calculate_acc(y_val, y)  # 计算验证准确率
                metric.add(val_loss * len(y), val_acc, len(y))
        self.val_loss = metric[0] / metric[2]
        self.val_acc = metric[1] / metric[2]

    def transform_x(self, x):
        '''转换x'''
        return x.to(self.device)

    def transform_y(self, y):
        '''转换y'''
        return y.to(self.device)

    def calculate_model(self, x):
        '''计算神经网络'''
        return self.model(x)

    def calculate_loss(self, y_hat, y):
        '''计算损失函数'''
        return self.loss(y_hat, y)

    def calculate_acc(self, y_hat, y):
        '''计算准确率'''
        y_hat = y_hat.argmax(dim=1)
        return (y_hat == y).sum()

    def grad_update(self, loss):
        '''梯度更新'''
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def output_print(self):
        '''打印输出值'''
        print(f'train loss {self.train_loss:.3f}, val loss {self.val_loss:.3f}, val acc {self.val_acc:.3f}')
        print(f'{self.timer.sum() / self.num_epoch:.1f} sec/epoch on {str(self.device)}')

    def test(self):
        '''测试模型'''
        metric = Accumulator(2)  # 累加器：(test_acc, test_size)
        self.model.eval()  # 验证模式
        with torch.no_grad():
            for x, y in self.test_iter:
                x = self.transform_x(x)  # 转换x
                y_test = self.calculate_model(x)  # 计算模型
                y = self.transform_y(y)  # 转换y
                test_acc = self.calculate_acc(y_test, y)  # 计算测试准确率
                metric.add(test_acc, len(y))
        print(f'Accuracy rate {metric[0] / metric[1]}')  # 计算测试准确率并输出
