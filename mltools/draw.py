from IPython import display
from matplotlib import pyplot as plt

class Animator:
    """在动画中绘制数据"""
    def __init__(self, *, line_num, xlabel=None, ylabel=None, xlim=None, ylim=None, legend=None, fmts=('-', 'm--', 'g-.', 'r:')):
        # 增量地绘制多条线
        self.fig, self.axes = plt.subplots()
        self.xlabel,self.ylabel,self.xlim,self.ylim,self.legend=xlabel,ylabel,xlim,ylim,legend
        self.X,self.Y=[[] for _ in range(line_num)],[[] for _ in range(line_num)]
        self.fmts=fmts
        
    def add(self, *y):
        for index,item in enumerate(y):
            self.Y[index].append(item)
            self.X[index].append(len(self.Y[index]))
        self.axes.cla()
        for index,(x,y) in enumerate(zip(self.X,self.Y)):
            self.axes.plot(x,y,self.fmts[index])
        self.set_axes()
        display.display(self.fig)
        display.clear_output(wait=True)

    def set_axes(self):
        self.axes.set_xlabel(self.xlabel)
        self.axes.set_ylabel(self.ylabel)
        self.axes.set_xlim(self.xlim)
        self.axes.set_ylim(self.ylim)
        if self.legend:self.axes.legend(self.legend)
        self.axes.grid()

def images(images,labels,shape):
    '''展示图片'''
    images=images.squeeze(1).to(device='cpu')
    fig,axes=plt.subplots(*shape)
    axes = [element for sublist in axes for element in sublist]
    for index,(ax,img,label) in enumerate(zip(axes,images,labels)):
        ax.set_title(label)
        ax.set_axis_off()
        ax.imshow(img ,cmap='gray')