import numpy as np

locs = np.array([[0.0708, 0.8151],
        [0.7679, 0.2864],
        [0.1931, 0.9789],
        [0.4062, 0.7578],
        [0.0892, 0.3099],
        [0.6189, 0.4599],
        [0.2183, 0.6635],
        [0.6787, 0.9503],
        [0.2813, 0.6199],
        [0.3833, 0.4004],
        [0.9427, 0.9299],
        [0.9484, 0.3755],
        [0.3423, 0.6648],
        [0.0423, 0.2322],
        [0.4301, 0.0779],
        [0.7666, 0.8539],
        [0.1504, 0.1011],
        [0.2709, 0.0302],
        [0.8378, 0.5976],
        [0.9311, 0.4955]])

def distance(p1, p2):
    return np.sqrt(np.sum(np.square(p1 - p2)))

def adjacency(locs):
    n = locs.shape[0]
    adjs = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dis = distance(locs[i], locs[j])
            adjs[i, j] = dis
    return adjs


adj = adjacency(locs)
# print(adj)
nodes =[ 7, 15, 10, 18, 19, 11,  5,  9,  8, 12,  3,  4, 13, 16, 14, 14, 14, 14,
        14]
# nodes = [13,  4, 16, 14,  9,  5,  1, 11, 18, 15, 10,  7,  3,  8, 12, 12, 12, 12,
#         12]
dises = []
for i in range(len(nodes) - 1):
    dis = adj[nodes[i], nodes[1+i]]
    dises.append(dis)
    print(f"dis of node {nodes[i]} - node {nodes[i+1]}", dis)
print(sum(dises))