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
    plt.savefig(save_path+"/val_reward2.jpg")

# 主函数
def main():
    file_path = "/home/panpan/rl4co/logs/train_rarl/runs/svrp20/am-svrp20/2024-01-06_20-28-27/csv/version_0"
    # 替换为你的CSV文件路径
    csv_file_path = file_path + '/metrics.csv'
    
    # 替换为你想提取的key
    target_key = 'val/reward'

    # 读取CSV文件
    data = read_csv(csv_file_path)

    # 提取指定key对应的值
    x_values = data.index  # 使用行索引作为X轴值
    y_values = extract_values(data, target_key)

    # 绘制曲线
    plot_curve(x_values, y_values, title=f"{target_key} Curve", save_path=file_path)

if __name__ == "__main__":
    main()
