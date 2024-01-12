import pandas as pd
import matplotlib.pyplot as plt
# 读取csv作为local logger，绘制指定key的曲线

# 读取CSV文件
def read_csv(file_path):
    return pd.read_csv(file_path)

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
    plt.savefig(save_path+"/val_reward_concat.jpg")

# 主函数
def main():
    
    former_file_path = "/home/panpan/rl4co/logs/train_rarl/runs/svrp20/am-svrp20/2024-01-10_21-14-54/csv/version_0"
    former_csv_file_path = former_file_path + "/metrics.csv"
    # 替换为你想提取的key
    target_key = 'val/reward'

    # 读取CSV文件
    former_data = read_csv(former_csv_file_path)

    # 提取指定key对应的值
    former_x_values = former_data.index  # 使用行索引作为X轴值
    former_y_values = extract_values(former_data, target_key)
    
    latter_file_path = "/home/panpan/rl4co/logs/train_rarl/runs/svrp20/am-svrp20/2024-01-11_09-26-02/csv/version_0"
    last_file_path = "/home/panpan/rl4co/logs/train_rarl/runs/svrp20/am-svrp20/2024-01-11_11-11-48/csv/version_0"
    _f = "/home/panpan/rl4co/logs/train_rarl/runs/svrp20/am-svrp20/2024-01-11_14-26-10/csv/version_0"
    # 一定要step按顺序写路径
    file_paths = [latter_file_path, last_file_path, _f]
    
    y_values = former_y_values
    for fp in file_paths:
        csv_fp = fp + "/metrics.csv"
        csv_data = read_csv(csv_fp)
        tmp_y_values = extract_values(csv_data, target_key)
        y_values = pd.concat([y_values, tmp_y_values])
        
    x_values = range(len(y_values))
    
    
    # 
    plot_curve(x_values, y_values, title=f"{target_key} Curve", save_path=_f)

if __name__ == "__main__":
    main()
