# homework

## 介绍

利用 MLP、CNN、LSTM 和 Transformer 对 MNIST 数据集进行图片分类，同时运用 LSTM 和 Transformer 开展文本情感分类。

## 软件架构

文件结构如下：

```text
homework
├── data
│   ├── mnist.npz
│   └── sentiment.csv
├── main
│   ├── mltools.py
│   └── <file_name>.ipynb
├── results
│   ├── <time_str>-<file_name>
│   │   ├── <file_name>.json
│   │   ├── <file_name>.log
│   │   ├── <file_name>.png
│   │   └── <file_name>.pth
│   └── <time_str>-<file_name>
│       ├── <file_name>.json
│       ├── <file_name>.log
│       ├── <file_name>.png
│       └── <file_name>.pth
├── .gitignore
└── README.md
```

其中，`main` 目录下存放的是各个实验的代码，`results` 目录下存放的是各个实验的结果，`data` 目录下存放的是数据集。

## 环境配置

使用uv安装依赖项。

```bash
uv sync --extra cu124
```

## 使用说明

1. 每次运行时都将生成一个结果文件夹，其中包含模型参数、训练日志、训练曲线、训练结果等。
2. 结果文件夹的命名格式为 `时间-文件名`，其中时间格式为 `年-月-日-时-分-秒`。
3. 结果文件夹中包含的文件如下：
    - `文件名.json`：训练结果
    - `文件名.log`：训练日志
    - `文件名.png`：训练曲线
    - `文件名.pth`：模型参数
4. 如果想要重现结果，可以通过ml.load(dir_name)函数加载模型参数。
5. 模型训练参数num_epochs为训练迭代总次数，并不是训练次数，因此当你已经迭代20次想再迭代20次时，num_epochs应该为40。
