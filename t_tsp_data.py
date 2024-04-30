import pandas as pd
import numpy as np
import os
import time

from tensordict.tensordict import TensorDict
def load_npz_to_tensordict(filename):
    """Load a npz file directly into a TensorDict
    We assume that the npz file contains a dictionary of numpy arrays
    This is at least an order of magnitude faster than pickle
    """
    x = np.load(filename)
    x_dict = dict(x)
    batch_size = x_dict[list(x_dict.keys())[0]].shape[0]
    return TensorDict(x_dict, batch_size=batch_size)

def process_swtwtsp_data(batch_size):
    # 直接读取提前生成的数据
    # CUSTNO,XCOORD,YCOORD,TW_LOW,TW_HIGH,PRIZE,MAXTIME
    dir_path = "/home/panpan/td-opswtw-competition-rl-main/td-opswtw-competition-rl-main/data/generated/n_nodes_20/instances/"
    
    # seed_lst = list(range(2, 1289997))     # 取这些种子的数据，
    path_lst = os.listdir(dir_path)
    path_lst_slice = [path_lst[-10000:]]
                    # path_lst[320001:400001], path_lst[400001:480001], path_lst[480001:560001],
                    # path_lst[560001:640001], path_lst[640001:720001], path_lst[720001:800001],
                    # path_lst[800001:880001], path_lst[880001:960001], path_lst[960001:1040001],
                    # path_lst[1040001:1120001], path_lst[1120001:1200001], path_lst[1200001:1280001]]

    def concat_csv_data(path_lst, idx):
        st = time.time()
        data = pd.read_csv(dir_path+path_lst[0])
        data = data.values[:, 1:]     # 去掉第一列node 序号
        data = data[None, ...]
        cnt = 0
        for pth in path_lst[1:]:
            # pth_name = str(seed_name)
            tmp = pd.read_csv(dir_path+pth)
            tmp = tmp.values[:, 1:]     # 去掉第一列node 序号
            tmp = tmp[None, ...]
            data = np.concatenate((data, tmp), axis=0)

            # 
            cnt += 1
            if cnt % 10000 == 0:
                print(cnt)
        locs = np.concatenate((data[:, :, 0][..., None], data[:, :, 1][..., None]), axis=2)
        # print(locs[0, :, :])
        data_dict = {
            "locs": locs.astype(np.float32),
            "tw_low": data[:, :, 2].astype(np.float32), # data_size, size
            "tw_high": data[:, :, 3].astype(np.float32),
            "prize": data[:, :, 4].astype(np.float32),
            "maxtime": data[:, :, 5].astype(np.float32)
        }
        print("time: ", time.time() - st)
        save_pth = "/home/panpan/rl4co/data/opswtw/opswtw_val.npz"
        np.savez(save_pth, **data_dict)

    idx = 3
    for plst in path_lst_slice:
        concat_csv_data(plst, idx)
        idx += 1
        print("concat data lst done:", idx)
    
    
    return 

def generate_opswtw_data(batch_size):
    '''
        load batch_size number data from presaved data
    '''
    data_pth = "/home/panpan/rl4co/data/opswtw_train/opswtw_train.npz"
    noise_pth = "/home/panpan/rl4co/data/opswtw_train/noise.npz"

    x = np.load(data_pth)
    x_dict = dict(x)
    noise_data = np.load(noise_pth)
    noise_dict = dict(noise_data)
    whole_size = x_dict[list(x_dict.keys())[0]].shape[0]

    assert whole_size >= batch_size, "Required size exceeds data size"

    locs = x_dict["locs"][:batch_size, ...]
    tw_low = x_dict["tw_low"][:batch_size, ...]
    tw_high = x_dict["tw_high"][:batch_size, ...]
    prize = x_dict["prize"][:batch_size, ...]
    maxtime = x_dict["maxtime"][:batch_size, ...]
    noise = noise_dict["noise"][:batch_size, ...]       # batch, size+1

    return {
        "locs": locs,
        "tw_low": tw_low,
        "tw_high": tw_high,
        "prize": prize,
        "maxtime": maxtime,
        "noise": noise
    }




def generate_stochastoc_factor():
    # 为每个节点生成一个随机因子
    size = 20
    dataset_size = 10000
    noise = np.random.randint(1, 101, size=(dataset_size, size)) / 100
    noise_dict = {
        "noise": noise.astype(np.float32)
    }
    save_pth = "/home/panpan/rl4co/data/opswtw/noise_val.npz"
    np.savez(save_pth, **noise_dict)

def concat_npz_data(dir):
    dir = "/home/panpan/rl4co/data/swtwtsp_train/"
    npz_pth = "/home/panpan/rl4co/data/swtwtsp_train/opswtw_train_0.npz"
    # npz_pth2 = "/home/panpan/rl4co/data/opswtw_train/opswtw_train_4.npz"

    data = dict(np.load(npz_pth))
    for idx in range(1, 16):
        p = dir + "opswtw_train_" + str(idx) + ".npz"

        tmp = dict(np.load(p))
        # print(data["locs"].shape, tmp["locs"].shape)
        for key in data.keys():
            # print(key)
            data[key]  = np.concatenate((data[key], tmp[key]), axis=0)
            # print(data[key].shape)
    # print(data["locs"].shape)
    save_pth = "/home/panpan/rl4co/data/opswtw_train/opswtw_train.npz"
    # print(data)
    np.savez(save_pth, **data)

def generate_and_save_opsa_data(batch_size, num_loc):
    attack_prob = np.random.uniform(0, 1, size=(batch_size, num_loc)).astype(np.float32)
    attack_prob[:, 0] =0
    real_prob = np.random.uniform(0, 1, size=(batch_size, num_loc)).astype(np.float32)
    real_prob[:, 0] =0
    weather = np.random.uniform(0, 1, size=(batch_size, 3)).astype(np.float32)
    
    data = {"attack_prob": attack_prob, 
                "real_prob": real_prob,
                "weather": weather}
    val_pth = "/home/panpan/rl4co/data/opsa/val_part_data.npz"
    np.savez(val_pth, **data)

generate_and_save_opsa_data(10000, 20)
# process_swtwtsp_data(0)
# generate_opswtw_data(1280000)
# concat_npz_data(0)
# td = load_npz_to_tensordict("/home/panpan/rl4co/data/opswtw_train/opswtw_train.npz")
# print(td["maxtime"].max(), td["tw_high"].max())
# print(td.batch_size, td.keys())
# generate_stochastoc_factor()