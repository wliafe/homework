from matplotlib import pyplot as plt

def plot(x=None,y=None,xlabel=None,ylabel=None,fmts=('-', 'm--', 'g-.', 'r:')):
    '''画折线图'''
    if not y:
        y=[[]]
    elif not isinstance(y[0], list):
        y=[y]
    if not x:
        x=list(range(1,len(y[0])+1))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for index,y in enumerate(y):
        plt.plot(x,y,fmts[index])
    plt.grid()
    plt.show()

def images(images,labels,shape):
    '''展示图片'''
    images=images.to(device='cpu')
    for index,(img,label) in enumerate(zip(images,labels)):
        plt.subplot(*shape,index+1)
        plt.title(label)
        plt.axis('off')
        plt.imshow(img[0],cmap='gray')
    plt.show()