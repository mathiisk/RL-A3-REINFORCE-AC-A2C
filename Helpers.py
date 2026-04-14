import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


class LearningCurvePlot:

    def __init__(self,title=None):
        self.fig,self.ax = plt.subplots()
        self.ax.set_xlabel('Timestep')
        self.ax.set_ylabel('Episode Return')
        if title is not None:
            self.ax.set_title(title)

    def add_curve(self,x,y, std=None, label=None):
        ''' y: vector of average reward results
        label: string to appear as label in plot legend '''
        if label is not None:
            line, = self.ax.plot(x, y, label=label)
        else:
            line, = self.ax.plot(x, y)

        if std is not None:
            y = np.array(y)
            std = np.array(std)
            self.ax.fill_between(x, y - std, y + std, alpha=0.3, color=line.get_color())

    def set_ylim(self,lower,upper):
        self.ax.set_ylim([lower,upper])

    def add_hline(self,height,label):
        self.ax.axhline(height,ls='--',c='k',label=label)

    def save(self,name='test.png'):
        ''' name: string for filename of saved figure '''
        self.ax.legend()
        self.fig.savefig(name,dpi=300)

def smooth(y, window, poly=2):
    '''
    y: vector to be smoothed
    window: size of the smoothing window '''
    return savgol_filter(y,window,poly)

