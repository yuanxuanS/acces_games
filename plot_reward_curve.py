import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
linestyle = ['-', '--', ':', '-.']
color = ['r', 'g', 'b', 'k']
label = ['algo1', 'algo2', 'algo3', 'algo4']

def smooth(data, sm=1):
    if sm > 1:
        smooth_data = []
        for d in data:
            y = np.ones(sm)*1.0/sm
            d = np.convolve(y, d, "same")
 
            smooth_data.append(d)
 
    return smooth_data

data = np.array([1.,3,4,6,5,7,8,7,9])
# 下载好数据后，根据自己文件路径，修改np.load()中的代码路径
y_data = smooth(data, 2)
x_data = np.arange(len(y_data))
print(x_data, y_data)
sns.set(style="darkgrid", font_scale=1.5)
ax =sns.tsplot(time=x_data, data=y_data, color=color[2], linestyle=linestyle[0])
fig = ax.get_figure()
fig.savefig("curve.png")