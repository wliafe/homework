from IPython import display
from matplotlib import pyplot as plt


class Animator:
    """在动画中绘制数据"""

    def __init__(self, *, xlabel: str = None, ylabel: str = None, xlim=None, ylim=None):
        self.fig, self.axes = plt.subplots()  # 生成画布
        self.set_axes = lambda: self.axes.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim)  # 初始化设置axes函数

    def show(self, Y: list[list[int]], legend: list[str] = None):
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


def images(images, labels: list[str], shape: tuple[int, int]):
    '''展示图片'''
    images = images.to(device='cpu')
    fig, axes = plt.subplots(*shape)
    axes = [element for sublist in axes for element in sublist]
    for ax, img, label in zip(axes, images, labels):
        ax.set_title(label)
        ax.set_axis_off()
        ax.imshow(img, cmap='gray')
