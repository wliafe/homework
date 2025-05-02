import torch
from torch import nn
from torch.nn import functional as F

class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size, step_size=1, output_size=None, one_hot=True, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn, self.vocab_size, self.step_size, self.one_hot = rnn_layer, vocab_size, step_size, one_hot
        self.output_size = output_size if output_size else vocab_size
        self.hidden_size = self.rnn.hidden_size
        self.directions = 1 if not self.rnn.bidirectional else 2 # 如果RNN是双向的，num_directions应该是2，否则应该是1
        self.linear = nn.Linear(self.step_size * self.hidden_size * self.directions, self.output_size) # 定义输出层

    def forward(self, inputs):
        X = F.one_hot(inputs.long(), self.vocab_size) if self.one_hot else inputs # 将输入独热编码，X形状为(批量大小, 时间步数, 词表大小)
        X = X.to(torch.float32)
        Y, state = self.rnn(X) # Y形状为(批量大小, 时间步数, 隐藏大小),state形状为(隐藏层数*num_directions, 批量大小, 隐藏大小)
        output = self.linear(Y.reshape(-1, self.step_size * self.hidden_size * self.directions)) # 它的输出形状是(批量大小, 输出大小)。
        return output, state