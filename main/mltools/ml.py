import torch
from torch import nn,optim
from .draw import Animator
from .timer import Timer
from .accumulator import Accumulator

class MachineLearning:
    def __init__(self, model, train_iter, val_iter=None, test_iter=None, *, device=torch.device('cpu')):
        '''初始化函数'''
        model.to(device) # 将网络复制到device上
        self.model, self.train_iter, self.val_iter, self.device = model, train_iter, val_iter, device
        self.test_iter = test_iter if test_iter else val_iter # 定义测试集
        self.setTimer() # 设置计时器
        self.setLoss() # 设置损失函数

    def train(self,num_epochs,learning_rate):
        '''训练模型'''
        self.setOptimizer(learning_rate) # 定义优化器
        self.setAnimator(num_epochs) # 设置Animator
        for num_epoch in range(1,num_epochs+1):
            self.train_epoch(num_epoch)

    def setTimer(self):
        '''设置计时器'''
        self.timer = Timer()

    def setLoss(self, loss=None):
        '''设置损失函数'''
        self.loss = loss if loss else nn.CrossEntropyLoss()  # 定义损失函数

    def setOptimizer(self, learning_rate):
        '''设置优化器'''
        self.optimizer = optim.SGD(self.model.parameters(), learning_rate) # 定义优化器

    def setAnimator(self, num_epochs):
        '''设置Animator'''
        self.animator = Animator(line_num=2,xlabel='epoch',ylabel='loss',xlim=[0, num_epochs+1],ylim=-0.1,legend=['train loss','val loss'])

    def train_epoch(self, num_epoch):
        '''一个迭代周期'''
        self.timer.start()
        train_loss=self.calculate_train_iter() # 计算训练集            
        self.timer.stop()
        val_loss=self.calculate_val_iter() # 计算验证集            
        print(f'train loss {train_loss:.3f}, val loss {val_loss:.3f}')
        print(f'{self.timer.sum() / num_epoch:.1f} sec/epoch on {str(self.device)}')
        self.animator.add(train_loss.detach().cpu(),val_loss.detach().cpu()) # 添加损失值

    def calculate_train_iter(self):
        '''计算训练集'''
        for x, y in self.train_iter:
            train_loss = self.calculate_loss(x, y) # 计算训练损失
            self.grad_update(train_loss) # 梯度更新
        return train_loss

    def calculate_val_iter(self):
        '''计算验证集'''
        with torch.no_grad():
            for x, y in self.val_iter:
                val_loss = self.calculate_loss(x, y)
        return val_loss

    def calculate_loss(self, x, y):
        '''计算损失函数'''
        y_train = self.calculate_y(x)
        y = y.to(self.device)
        return self.loss(y_train, y)

    def calculate_y(self, x):
        '''计算神经网络'''
        x = x.to(self.device)
        return self.model(x)

    def grad_update(self, loss):
        '''梯度更新'''
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def test(self):
        '''测试模型'''
        accumulate=Accumulator(2) # 定义测试数量和预测真实数量
        # 测试
        for x,y in self.test_iter:
            pred=self.calculate_acc(x, y)
            accumulate.add(pred.sum(),len(pred))
        print(f'Accuracy rate {accumulate[0] / accumulate[1]}') # 计算测试准确率并输出

    def calculate_acc(self, x, y):
        '''计算准确率'''
        y_test=self.calculate_y(x)            
        y_test=y_test.argmax(dim=1)            
        y = y.to(self.device)            
        return y==y_test