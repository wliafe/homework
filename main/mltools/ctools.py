import torch
import time
from .draw import Animator


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


class MachineLearning:
    def __init__(self, model, train_iter, val_iter=None, test_iter=None, *, device=torch.device('cpu')):
        '''初始化函数'''
        model.to(device)  # 将网络复制到device上
        self.model, self.train_iter, self.device = model, train_iter, device
        self.val_iter = val_iter if val_iter else self.train_iter  # 定义验证集
        self.test_iter = test_iter if test_iter else self.val_iter  # 定义测试集
        self.set_timer()  # 设置计时器
        self.set_recorder()  # 设置损失记录器

    def set_timer(self):
        '''设置计时器'''
        self.timer = Timer()

    def set_recorder(self):
        '''设置损失记录器'''
        self.recorder = Recorder(3)

    def set_loss(self, loss):
        '''设置损失函数'''
        self.loss = loss

    def set_optimizer(self, optimizer):
        '''设置优化器'''
        self.optimizer = optimizer

    def train(self, num_epochs):
        '''训练模型'''
        self.set_animator(num_epochs)  # 设置Animator
        for self.num_epoch in range(1, num_epochs+1):
            self.train_epoch()
        else:
            self.output_print()

    def set_animator(self, num_epochs):
        '''设置Animator'''
        rlim = num_epochs + self.recorder.max_record_size()  # 计算xlim的右边界
        self.animator = Animator(xlabel='epoch', xlim=[0, rlim+1], ylim=-0.1)

    def train_epoch(self):
        '''一个迭代周期'''
        self.timer.start()
        self.calculate_train_iter()  # 计算训练集
        self.timer.stop()
        self.calculate_val_iter()  # 计算验证集
        self.output_print()
        self.animator.show(self.recorder.data, legend=['train loss', 'val loss', 'val acc'])  # 添加损失值

    def calculate_train_iter(self):
        '''计算训练集'''
        metric = Accumulator(2)  # 累加器：(train_loss, train_size)
        self.model.train()  # 训练模式
        for x, y in self.train_iter:
            x = self.transform_x(x)  # 转换x
            y_train = self.calculate_model(x)  # 计算模型
            y = self.transform_y(y)  # 转换y
            y_train = self.transform_model_result(y_train)  # 转换模型结果
            train_loss = self.calculate_loss(y_train, y)  # 计算训练损失
            self.grad_update(train_loss)  # 梯度更新
            metric.add(train_loss * len(y), len(y))
        self.recorder[0].append(metric[0] / metric[1])

    def calculate_val_iter(self):
        '''计算验证集'''
        metric = Accumulator(3)  # 累加器：(val_loss, val_acc, val_size)
        self.model.eval()  # 验证模式
        with torch.no_grad():
            for x, y in self.val_iter:
                x = self.transform_x(x)  # 转换x
                y_val = self.calculate_model(x)  # 计算模型
                y = self.transform_y(y)  # 转换y
                y_val = self.transform_model_result(y_val)  # 转换模型结果
                val_loss = self.calculate_loss(y_val, y)  # 计算验证损失
                val_pred = self.calculate_pred(y_val)  # 计算预测值
                val_acc = self.calculate_acc(val_pred, y)  # 计算验证准确率
                metric.add(val_loss * len(y), val_acc, len(y))
        self.recorder[1].append(metric[0] / metric[2])
        self.recorder[2].append(metric[1] / metric[2])

    def transform_x(self, x):
        '''转换x'''
        return x.to(self.device)

    def transform_y(self, y):
        '''转换y'''
        return y.to(self.device)

    def calculate_model(self, x):
        '''计算神经网络'''
        return self.model(x)

    def transform_model_result(self, y):
        '''转换模型结果'''
        return y

    def calculate_loss(self, y_hat, y):
        '''计算损失函数'''
        return self.loss(y_hat, y)

    def calculate_pred(self, y):
        '''计算预测值'''
        return y.argmax(dim=1)

    def calculate_acc(self, pred, real):
        '''计算准确率'''
        return (pred == real).sum()

    def grad_update(self, loss):
        '''梯度更新'''
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def output_print(self):
        '''打印输出值'''
        print(f'train loss {self.recorder[0][-1]:.3f}, val loss {self.recorder[1][-1]:.3f}, val acc {self.recorder[2][-1]:.3f}')
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
                y_test = self.transform_model_result(y_test)  # 转换模型结果
                test_pred = self.calculate_pred(y_test)  # 计算准确率
                test_acc = self.calculate_acc(test_pred, y)  # 计算测试准确率
                metric.add(test_acc, len(y))
        print(f'Accuracy rate {metric[0] / metric[1]:.3f}')  # 计算测试准确率并输出

    def predict(self):
        '''预测模型'''
        self.model.eval()  # 验证模式
        x, y = next(iter(self.test_iter))  # 从测试中取一个批量
        x = self.transform_x(x[:10])
        y = self.transform_y(y[:10])
        y_pred = self.calculate_model(x)  # 计算模型
        y_pred = self.calculate_pred(y_pred)  # 计算预测
        self.show_pred(x, y_pred, y)  # 显示预测

    def show_pred(self, contents, preds, reals):
        '''显示预测'''
        raise NotImplementedError
