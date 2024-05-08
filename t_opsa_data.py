import pandas as pd
import numpy as np
import os
import time

from tensordict.tensordict import TensorDict

STOCH_PARAMS = {
    0: [0.6, 0.2, 0.2],
    1: [0.8, 0.2, 0.0],
    2: [0.8, 0.,  0.2],
    3: [0.4, 0.3, 0.3]
}
STOCH_IDX = 0

SEED=1234
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
    '''
    每80000条数据拼在一起，索引idx存起来
    '''
    # 直接读取提前生成的数据
    # CUSTNO,XCOORD,YCOORD,TW_LOW,TW_HIGH,PRIZE,MAXTIME
    dir_path = "/home/panpan/td-opswtw-competition-rl-main/td-opswtw-competition-rl-main/data/generated/n_nodes_20/instances/"
    
    # seed_lst = list(range(2, 1289997))     # 取这些种子的数据，
    path_lst = os.listdir(dir_path)
    print(len(path_lst))
    path_lst_slice = [path_lst[-20000:-10000]]
        # val: [path_lst[-10000:]]
        # path_lst[:80001], path_lst[80001:160001], path_lst[160001:240001],
        #             path_lst[240001:320001], path_lst[320001:400001], path_lst[400001:480001],
        #             path_lst[480001:560001], path_lst[560001:640001], path_lst[640001:720001],
        #             path_lst[720001:800001], path_lst[800001:880001], path_lst[880001:960001], 
        #             path_lst[960001:1040001], path_lst[1040001:1120001], path_lst[1120001:1200001],
        #               path_lst[1200001:1280001]]
    # [path_lst[-10000:]]
    def concat_csv_data(path_lst, idx):
        '''
        将path_lst里数据concat起来
        '''
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
        save_pth = "/home/panpan/rl4co/data/opsa/opsa_test.npz"
        np.savez(save_pth, **data_dict)

    idx = 0
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

    return {
        "locs": locs,
        "tw_low": tw_low,
        "tw_high": tw_high,
        "prize": prize,
        "maxtime": maxtime,
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
    dir = "/home/panpan/rl4co/data/opsa/"
    npz_pth = "/home/panpan/rl4co/data/opsa/opsa_train_0.npz"
    # npz_pth2 = "/home/panpan/rl4co/data/opswtw_train/opswtw_train_4.npz"

    data = dict(np.load(npz_pth))
    for idx in range(1, 16):
        p = dir + "opsa_train_" + str(idx) + ".npz"

        tmp = dict(np.load(p))
        # print(data["locs"].shape, tmp["locs"].shape)
        for key in data.keys():
            # print(key)
            data[key]  = np.concatenate((data[key], tmp[key]), axis=0)
            # print(data[key].shape)
    # print(data["locs"].shape)
    save_pth = "/home/panpan/rl4co/data/opsa/opsa_train.npz"
    # print(data)
    np.savez(save_pth, **data)

def generate_and_save_stoch_data(batch_size, num_loc):
    data = np.load("/home/panpan/rl4co/data"+str(STOCH_IDX)+"/opsa/opsa20_val.npz")
    locs = data["locs"]

    attack_prob = np.random.uniform(0, 1, size=(batch_size, num_loc)).astype(np.float32)
    attack_prob[:, 0] =0
    weather = np.random.uniform(-1, 1, size=(batch_size, 3)).astype(np.float32)
    
    real_prob = get_stoch_var(attack_prob[..., np.newaxis],
                              locs.copy(),
                                np.repeat(weather[:, np.newaxis, :], num_loc, axis=1),
                                None).squeeze(-1).astype(np.float32)
    real_prob = np.clip(real_prob, a_min=0., a_max=1.)
    real_prob[:, 0] =0
    
    data = {"attack_prob": attack_prob, 
                "real_prob": real_prob,
                "weather": weather}
    val_pth = "/home/panpan/rl4co/data"+str(STOCH_IDX)+"/opsa/opsa20_val_part_data.npz"
    np.savez(val_pth, **data)


def get_stoch_var(inp, locs, w, alphas, A=0.6, B=0.2, G=0.2):
    A, B, G = STOCH_PARAMS[STOCH_IDX]   # 和env的stoch_idx一致
    print(f"ABG in generate data is {A} {B} {G}")
    n_problems,n_nodes,shape = inp.shape
    T = inp/A
    
    var_noise = T*G
    # noise = np.random.randn(n_problems,n_nodes, shape)      #=np.rand.randn, normal dis(0, 1)
    # noise = var_noise*noise     # multivariable normal distr, var_noise mean
    # noise = np.clip(noise, a_min=-var_noise, a_max=var_noise)
    
    noise = np.sqrt(var_noise)*np.random.randn(n_problems,n_nodes, shape)      #=np.rand.randn, normal dis(0, 1)
    noise = np.clip(noise, a_min=-var_noise, a_max=var_noise)
    
    var_w = np.sqrt(T*B)
    # sum_alpha = var_w[:, :, np.newaxis, :]*4.5      #? 4.5
    # alphas = np.random.random((n_problems, n_nodes, 9, shape))      # =np.random.random, uniform dis(0, 1)
    # alphas /= alphas.sum(axis=2)[:, :, np.newaxis, :]       # normalize alpha to 0-1
    # alphas *= sum_alpha     # alpha value [4.5*var_w]
    # alphas = np.sqrt(alphas)        # alpha value [sqrt(4.5*var_w)]
    # signs = np.random.random((n_problems, n_nodes, 9, shape))
    # signs = np.where(signs > 0.5)
    # alphas[signs] *= -1     # half negative: 0 mean, [sqrt(-4.5*var_w) ,s sqrt(4.5*var_w)]
    
    # sum_alpha = var_w[:, :, None, :]*4.5      #? 4.5
    sum_alpha = var_w[:, :, np.newaxis, :]*9      #? 4.5
    # alphas = np.random.random((n_problems, n_nodes, 9, shape))      # =np.random.random, uniform dis(0, 1)
    alphas = np.random.random((n_problems, 1, 9, shape))      # =np.random.random, uniform dis(0, 1)
    alphas_loc = locs.sum(-1)[..., np.newaxis, np.newaxis]/2 * alphas
    
    
    # alphas_loc /= alphas_loc.sum(axis=2)[:, :, np.newaxis, :]       # normalize alpha to 0-1
    alphas_loc *= sum_alpha     # alpha value [4.5*var_w]
    alphas_loc = np.sqrt(alphas_loc)        # alpha value [sqrt(4.5*var_w)]
    signs = np.random.random((n_problems, n_nodes, 9, shape))
    alphas_loc[np.where(signs > 0.5)] *= -1     # half negative: 0 mean, [sqrt(-4.5*var_w) ,s sqrt(4.5*var_w)]
        
    # w1 = np.repeat(w, 3, axis=2)[..., np.newaxis]       # [batch, nodes, 3*repeat3=9, 1]
    # # roll shift num in axis: [batch, nodes, 3] -> concat [batch, nodes, 9,1]
    # w2 = np.concatenate([w, np.roll(w,shift=1,axis=2), np.roll(w,shift=2,axis=2)], 2)[..., np.newaxis]
    
    # tot_w = (alphas*w1*w2).sum(2)       # alpha_i * wm * wn, i[1-9], m,n[1-3], [batch, nodes, 9]->[batch, nodes,1]
    # tot_w = np.clip(tot_w, a_min=-var_w, a_max=var_w)
    
   
    tot_w = (alphas_loc*
             np.repeat(w, 3, axis=2)[..., np.newaxis]*
             np.concatenate([w, np.roll(w,shift=1,axis=2), 
                             np.roll(w,shift=2,axis=2)], 2)[..., np.newaxis]
            ).sum(2)       # alpha_i * wm * wn, i[1-9], m,n[1-3], [batch, nodes, 9]->[batch, nodes,1]
    tot_w = np.clip(tot_w, a_min=-var_w, a_max=var_w)
    out = inp + tot_w + noise
    out = np.clip(out, a_min=0.01, a_max=1e5)       # a_min a_max必须都是标量
    
    del sum_alpha, alphas_loc, signs, tot_w
    del T, noise, var_w
    # gc.collect()
        
    return out

generate_and_save_stoch_data(10000, 20)
# process_swtwtsp_data(0) # 先将所有数据每80000条分别存下来
# generate_opswtw_data(1280000)
# concat_npz_data(0)

# td = load_npz_to_tensordict("/home/panpan/rl4co/data0/opsa/opsa20_test.npz")
# print(len(dict(td)["locs"]))
# print(dict(td)["locs"][0,:10])
# print(td["maxtime"].max(), td["tw_high"].max())
# print(td.batch_size, td.keys())
# generate_stochastoc_factor()