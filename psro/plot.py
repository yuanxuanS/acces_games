import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import seaborn as sns
from scipy.interpolate import make_interp_spline

plt.rcParams['figure.figsize'] = (6.5, 5)
plt.rcParams['axes.labelsize'] = 15
def inter(x, y):
    x_smooth = np.linspace(x.min(), x.max(), 300)  # np.linspace 等差数列,从x.min()到x.max()生成300个数，便于后续插值
    y_smooth = make_interp_spline(x, y)(x_smooth)
    return x_smooth, y_smooth

linestyle = ['-', '--', ':', '-.']
color = ['r', 'g', 'b', 'k']

save_dir = "/home/panpan/rl4co/psro/graphs4/"
def load_psro_info(log_pth):
     
    data = np.load(log_pth+ '/info.npz')  # 加载
    nashconv = data['nashconv_lst']  # 引用保存好的数组，他的格式默认是numpy.array
    payoffs = data['payoffs']
    return nashconv, payoffs


def draw_jpc(env_name, pth, save_format):
    _, payoffs = load_psro_info(pth)
    # plot JPC
    plt.figure()
    plt.rcParams['font.size'] = 14
    plt.imshow(payoffs, cmap='coolwarm', norm=PowerNorm(3))
    plt.title("utility of PSRO ("+env_name[:-2]+" "+env_name[-2:]+")")
    plt.xlabel("adversary")
    plt.ylabel("protagonist")
    plt.colorbar()
    graph_name =  "JPC"  #
    name = graph_name +"_log_"+env_name
    global save_dir
    plt.savefig(save_dir+name+"."+save_format, format=save_format)
    plt.close()
    # compute JPC

    print(payoffs.shape)
    total_sum = np.sum(payoffs)

    diagonal_sum = 0
    size = payoffs.shape[0]
    for i in range(size):
        diagonal_sum += payoffs[i][i]

    nondiag_sum = total_sum - diagonal_sum

    jpc = 1 - (diagonal_sum / size) / (nondiag_sum / (size*size - size))
    print(f"total_sum is {total_sum}, diag sum is {diagonal_sum}, nondiag sum is {nondiag_sum}, jpc is {jpc}")

def plot_nashconv(env_name, pth, window=10, type="inter", save_format="jpg"):
    nashconv, _ = load_psro_info(pth)
    # plot nashconv
    # plt.plot(nashconv)
    # # plt.yscale("log")
    # plt.title("NashConv of PSRO("+env_name+")")
    # plt.xlabel("iteration")
    # plt.ylabel("NashConv")

    graph_name =  "nashconv"    #"JPC"  #
    name = graph_name +"_log_"+env_name
    # global save_dir
    # plt.savefig(save_dir+name+".jpg")

    # smooth 
    data = np.array(nashconv)
    # 下载好数据后，根据自己文件路径，修改np.load()中的代码路径
    # y_data = smooth(data, window)
    if type== "conv":
        y_sm = moving_average(data, window)
        x_data = np.arange(len(y_sm))
    elif type == "inter":
        x_data = np.arange(len(data))
        x_sm, y_sm = inter(x_data, data)
    
    print(y_sm)
    print(f"len y {len(y_sm)}, len x {len(data)}")
    print(x_data, y_sm)
    
    plt.figure()
    plt.rcParams['font.size'] = 14
    # sns.set(style="darkgrid")
    if type == "inter":
        ax =sns.tsplot(time=x_sm, data=y_sm, color=color[2], linestyle=linestyle[0])
    elif type == "conv":
        ax =sns.tsplot(time=x_data, data=y_sm, color=color[2], linestyle=linestyle[0])
    # ax =sns.tsplot(time=x_data, data=data, color=color[2], linestyle=linestyle[0], alpha=0.3)
    plt.xlabel("Iterations", )
    plt.ylabel("Exploitability")
    plt.title(f"{env_name[:-2]} {env_name[-2:]}")
    # sns.set(font_scale=3)
    # fig = ax.get_figure()
    plt.savefig(save_dir+name+"_sm_"+str(window)+"."+save_format, format=save_format)
    plt.close()

def moving_average(interval, windowsize):
    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(interval, window, 'same')
    return re

def smooth(data, sm=1):
    if sm > 1:
        smooth_data = []
        for d in data:
            y = np.ones(sm)*1.0/sm
            d = np.convolve(y, d, "same")
 
            smooth_data.append(d)
 
    return smooth_data


envs = ["csp20", "opsa20", "opsa50", "svrp20"]
pths = ["/home/panpan/rl4co/logs/train_psro/runs/csp20/am-csp20/2024-05-20_03-59-47",
        "/home/panpan/rl4co/logs/train_psro/runs/opsa20/am-opsa20/2024-05-20_14-19-12",
        "/home/panpan/rl4co/logs/train_psro/runs/opsa50/am-opsa50/2024-05-20_21-15-34",
        "/home/panpan/rl4co/logs/train_psro/runs/svrp20/am-svrp20/2024-05-19_22-38-12",
        ]
window=7
type="conv"
save_format = "eps"
for env_name, log_pth in zip(envs, pths):
    # env_name = "svrp20"  #"csp20" 
    log_pth = log_pth + "/psro/"
    # log_pth = "/home/panpan/rl4co/logs/train_psro/runs/svrp20/am-svrp20/2024-05-19_22-38-12/psro/"
    

    plot_nashconv(env_name, log_pth, window, type, save_format)
    draw_jpc(env_name, log_pth, save_format)
