from IPython import display
from matplotlib import pyplot as plt

class Animator:
    """在动画中绘制数据"""
    def __init__(self, *, line_num, xlabel=None, ylabel=None, xlim=None, ylim=None, legend=None):
        self.fig, self.axes = plt.subplots() # 生成画布
        self.set_axes= lambda : self.axes.set(xlabel=xlabel,ylabel=ylabel,xlim=xlim,ylim=ylim) # 初始化设置axes函数
        self.legend=legend # 初始化标签
        self.X,self.Y=[[] for _ in range(line_num)],[[] for _ in range(line_num)] # 初始化数据容器
        
    def add(self, *y):
        '''添加数据'''
        for index,item in enumerate(y):
            self.Y[index].append(item)
            self.X[index].append(len(self.Y[index]))
        self.axes.cla() # 清除画布
        for x,y,fmt in zip(self.X,self.Y,('-', 'm--', 'g-.', 'r:')):
            self.axes.plot(x,y,fmt)
        self.set_axes() # 设置axes
        if self.legend:self.axes.legend(self.legend) # 设置标签
        self.axes.grid() # 设置网格线
        display.display(self.fig) # 画图
        display.clear_output(wait=True) # 清除输出

def images(images,labels,shape):
    '''展示图片'''
    images=images.squeeze(1).to(device='cpu')
    fig,axes=plt.subplots(*shape)
    axes = [element for sublist in axes for element in sublist]
    for index,(ax,img,label) in enumerate(zip(axes,images,labels)):
        ax.set_title(label)
        ax.set_axis_off()
        ax.imshow(img ,cmap='gray')