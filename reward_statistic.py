import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# 读取csv作为local logger，绘制指定key的曲线

# 读取CSV文件
def read_npz(file_path):
    data = np.load(file_path)
    return data["rewards"]

# 提取指定key对应的值
def extract_values(dataframe, key):
    return dataframe[key]

# 绘制曲线
def plot_curve(x_values, y_values, title="Curve Plot", save_path="./"):
    plt.plot(x_values, y_values, marker='o', linestyle='-', color='b')
    plt.title(title)
    plt.xlabel("X Axis Label")
    plt.ylabel("Y Axis Label")
    plt.grid(True)
    plt.show()
    plt.savefig(save_path+"/val_reward2.jpg")

def statis_reward(y, save_path, prefix, color):
    if not isinstance(y, list):
        y = list(y)
    # print(y)
    y_sort = sorted(y)
    
    # 统计最大最小值
    print(f"{prefix}, max: {max(y_sort)}, min: {min(y_sort)}")
    
    x_axis = [x/ len(y_sort) * 100 for x in list(range(len(y_sort)))] 
    plt.plot(x_axis, y_sort, linestyle='-', color=color, label=prefix)
    plt.title("percentile-val/reward"+"_"+prefix)
    plt.xlabel("percentile")
    plt.ylabel("val/reward")
    plt.legend()
    plt.show()
    
    plt.savefig(save_path+"/percentile-reward.jpg")

        
# 主函数
def main():
    file_path = "/home/panpan/rl4co/logs/train_rarl/runs/svrp_fix20/am-svrp_fix20/2024-01-12_21-33-13"
    # 替换为你的CSV文件路径
    npz_rarl_p = file_path + '/evalu_5adv_rarltrainedprog_greedy.npz'
    npz_p = file_path + '/evalu_5adv_nonrarltrainedprog_greedy.npz'
    # 替换为你想提取的key
    target_key = 'val/reward'

    # 读取CSV文件
    rarl_data = read_npz(npz_rarl_p)
    nonrarl_data = read_npz(npz_p)

    
    
    # 绘制percentile-reward
    rarl_prefix = "rarl-prog"
    rarl_color = "r"
    nonrarl_prefix = "nonrarl-prog"
    nonrarl_color = "b"
    statis_reward(rarl_data, file_path, rarl_prefix, rarl_color)
    statis_reward(nonrarl_data, file_path, nonrarl_prefix, nonrarl_color)

if __name__ == "__main__":
    main()
