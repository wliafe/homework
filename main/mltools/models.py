import torch
from torch import nn


class MyModel(nn.Module):
    '''自定义模型'''

    def __init__(self, embedding, hidden_layer, output_layer, *args, **kwargs):
        '''初始化函数'''
        nn.Module.__init__(self, *args, **kwargs)
        self.embedding = embedding  # 定义嵌入层
        self.hidden_layer = hidden_layer  # 定义隐藏层
        self.output_layer = output_layer  # 定义输出层


class RNNModel(MyModel):
    '''循环神经网络模型'''

    def __init__(self, embedding, rnn_layer, step_size: int, output_size: int,  *args, **kwargs):
        '''初始化函数'''
        hidden_size = rnn_layer.hidden_size  # 定义时间步数和隐藏大小
        directions = 1 if not rnn_layer.bidirectional else 2  # 如果RNN是双向的，num_directions应该是2，否则应该是1
        self.input_size = step_size * hidden_size * directions
        output_layer = nn.Linear(self.input_size, output_size)  # 定义输出层
        MyModel.__init__(self, embedding, rnn_layer, output_layer, *args, **kwargs)

    def forward(self, x, state=None):
        '''前向传播'''
        x = self.embedding(x)  # 将输入独热编码，x形状为(批量大小, 时间步数, 词表大小)
        x = x.to(torch.float32)
        x, state = self.hidden_layer(x, state)  # x形状为(批量大小, 时间步数, 隐藏大小),state形状为(隐藏层数*directions, 批量大小, 隐藏大小)
        x = self.output_layer(x.reshape(-1, self.input_size))  # 它的输出形状是(批量大小, 输出大小)
        return x, state


class TransformerEncodeModel(MyModel):
    '''Transformer模型'''

    def __init__(self, vocab_size: int, output_size: int, d_model: int = 256, nhead: int = 8, batch_first: bool = True, encode_num_layers: int = 6, *args, **kwargs):
        '''初始化函数'''
        embedding = nn.Embedding(vocab_size, d_model, padding_idx=1)  # 定义嵌入函数
        encode_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=batch_first)
        hidden_layer = nn.TransformerEncoder(encode_layer, encode_num_layers)  # 定义TransformerEncoder
        output_layer = nn.Linear(d_model, output_size)  # 定义输出层
        MyModel.__init__(self, embedding, hidden_layer, output_layer, *args, **kwargs)

    def forward(self, x):
        '''前向传播'''
        x = self.embedding(x)
        x = self.hidden_layer(x)
        x = self.output_layer(x[:, 0, :])
        return x
