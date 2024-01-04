import argparse
from scipy.sparse.csgraph import shortest_path
from sklearn import metrics
import numpy as np
import pandas as pd
import torch
import dgl
import math

from ogb.linkproppred import DglLinkPropPredDataset, Evaluator

import scipy.io as scio


def parse_arguments():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser(description='SEAL')
    parser.add_argument('--dataset', type=str, default='1-CircR2Disease')# 1-CircR2Disease/2-circAtlas/3-Circ2Disease/4-CircRNADisease
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--hop', type=int, default=3)  # 1
    parser.add_argument('--model', type=str, default='gnn')
    parser.add_argument('--gnn_type', type=str, default='sagemagna')
    parser.add_argument('--use_attribute',default=True)
    parser.add_argument('--num_layers', type=int, default=3)  # 3
    parser.add_argument('--alpha', type=float, default=0.4)
    parser.add_argument('--hop_num', type=int, default=5)
    parser.add_argument('--in_dim', type=int, default=32)
    parser.add_argument('--hidden_units', type=int, default=64)
    parser.add_argument('--sort_k', type=int, default=30)
    parser.add_argument('--pooling', type=str, default='sum')
    parser.add_argument('--dropout', type=str, default=0.5)
    # parser.add_argument('--hits_k', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--neg_samples', type=int, default=1)
    parser.add_argument('--subsample_ratio', type=float, default=1)  #1
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)  #32
    parser.add_argument('--num_workers', type=int, default=0)  # 32
    parser.add_argument('--random_seed', type=int, default=2022)

    parser.add_argument('--load_dir', type=str, default='./data/circRNA-disease/1-CircR2Disease/基于circRNA序列信息的circRNA-disease关联关系/All Species/')
    parser.add_argument('--save_dir', type=str, default='./processed')
    args = parser.parse_args()

    return args


def load_cda_dataset_have_unknown (dataset, random_seed, directory, device):  # 有负边
    IC_DO = scio.loadmat(directory + 'DO/integrated similarities/integrated_circ_sim.mat')['integrated_circ_sim']
    ID_DO = scio.loadmat(directory + 'DO/integrated similarities/integrated_dise_sim.mat')['integrated_dise_sim']

    IC_G = scio.loadmat(directory + 'Gaussian interaction profile kernel/circ_gipk.mat')['circ_gipk']
    ID_G = scio.loadmat(directory + 'Gaussian interaction profile kernel/dis_gipk.mat')['dis_gipk']

    IC_sequence = scio.loadmat(directory + 'CircRNA sequence similarity/Levenshtein/LevenshteinSimilar.mat')['LevenshteinSimilar']

    IC_mesh = scio.loadmat(directory + 'MeSH/integrated similarities/integrated_circ_sim.mat')['integrated_circ_sim']
    ID_mesh = scio.loadmat(directory + 'MeSH/integrated similarities/integrated_dise_sim.mat')['integrated_dise_sim']

    IC = np.hstack((IC_mesh,IC_G))
    ID = np.hstack((ID_mesh, ID_G))

    associations = pd.read_excel(directory + 'Association Matrix.xlsx', sheet_name='Association Matrix',header=None)
    associations.columns = range(1,ID.shape[0]+1)  # 更新索引
    associations.index = range(1,IC.shape[0]+1)

    known_asss = associations.where(associations > 0).stack()  # 筛选数据
    unknown_asss = associations.where(associations == 0).stack()
    # 构建新的df
    known_associations = pd.concat([pd.DataFrame([[known_asss.index[i][0],known_asss.index[i][1],'1']],columns=['circrna','diseases','label']) for i in range(known_asss.__len__())])
    unknown_asss =unknown_asss.sample(n = known_asss.shape[0],random_state = random_seed, axis = 0)  # 随机抽取
    unknown_associations = pd.concat([pd.DataFrame([[unknown_asss.index[i][0],unknown_asss.index[i][1],'0']],columns=['circrna','diseases','label']) for i in range(unknown_asss.__len__())])
    all_associations = known_associations.append(unknown_associations)
    all_associations.reset_index(drop=True, inplace=True)  # 重置索引

    samples = all_associations.values      # 获得重新编号的新样本,.values可以得到数组array([[...]],dtype=int64)
    g = dgl.DGLGraph().to(device)
    g.add_nodes(ID.shape[0] + IC.shape[0])
    node_type = torch.zeros(g.number_of_nodes(), dtype=torch.int64).to(device)
    node_type[: ID.shape[0]] = 1
    g.ndata['type'] = node_type

    d_feat = torch.zeros(g.number_of_nodes(), ID.shape[1]).to(device)  # [878,383]
    d_feat[: ID.shape[0], :] = torch.from_numpy(ID.astype('float32'))
    g.ndata['d_feat'] = d_feat

    c_feat = torch.zeros(g.number_of_nodes(), IC.shape[1]).to(device)
    c_feat[ID.shape[0]: ID.shape[0] + IC.shape[0], :] = torch.from_numpy(IC.astype('float32')).to(device)
    g.ndata['c_feat'] = c_feat

    disease_ids = list(range(1, ID.shape[0] + 1))  # 1...383
    circrna_ids = list(range(1, IC.shape[0] + 1))  # 1...495

    disease_ids_invmap = {id_: i for i, id_ in enumerate(disease_ids)}  # {1:0,2:1,...,383:382}
    circrna_ids_invmap = {id_: i for i, id_ in enumerate(circrna_ids)}

    sample_disease_vertices = [disease_ids_invmap[id_] for id_ in samples[:, 1]]
    sample_circrn_vertices = [circrna_ids_invmap[id_] + ID.shape[0] for id_ in samples[:, 0]]

    g.add_edges(sample_disease_vertices, sample_circrn_vertices,
                data={'label': torch.from_numpy(samples[:, 2].astype('float32')).to(device)})
    g = dgl.add_reverse_edges(g)  # 添加反向边，即转为无向图
    g.readonly()
    split_edge = get_split_edge(g=g, train_ratio=0.85, val_ratio=0.05, test_ratio=0.1)
    return g, split_edge


def build_graph(device, directory):
    IC_DO = scio.loadmat(directory + 'DO/integrated similarities/integrated_circ_sim.mat')['integrated_circ_sim']
    ID_DO = scio.loadmat(directory + 'DO/integrated similarities/integrated_dise_sim.mat')['integrated_dise_sim']

    IC_G = scio.loadmat(directory + 'Gaussian interaction profile kernel/circ_gipk.mat')['circ_gipk']
    ID_G = scio.loadmat(directory + 'Gaussian interaction profile kernel/dis_gipk.mat')['dis_gipk']

    IC_sequence = scio.loadmat(directory + 'CircRNA sequence similarity/Levenshtein/LevenshteinSimilar.mat')[
        'LevenshteinSimilar']

    IC_mesh = scio.loadmat(directory + 'MeSH/integrated similarities/integrated_circ_sim.mat')['integrated_circ_sim']
    ID_mesh = scio.loadmat(directory + 'MeSH/integrated similarities/integrated_dise_sim.mat')['integrated_dise_sim']

    IC = np.hstack((IC_mesh, IC_G))
    ID = np.hstack((ID_mesh, ID_G))

    associations = pd.read_excel(directory + 'Association Matrixs.xlsx', sheet_name='Association Matrix', header=None)


    known_asss = associations.where(associations > 0).stack()  # 筛选数据

    known_associations = pd.concat(
        [pd.DataFrame([[known_asss.index[i][0], known_asss.index[i][1] + IC.shape[0]]], columns=['circrna', 'diseases'])
         for i in range(known_asss.__len__())])

    g = dgl.DGLGraph().to(device)
    g.add_nodes(IC.shape[0] + ID.shape[0])
    node_type = torch.zeros(g.number_of_nodes(), dtype=torch.int64).to(device)
    node_type[: ID.shape[0]] = 1
    g.ndata['type'] = node_type

    d_feat = torch.zeros(g.number_of_nodes(), ID.shape[1]).to(device)  # [878,383]
    d_feat[: ID.shape[0], :] = torch.from_numpy(ID.astype('float32'))
    g.ndata['d_feat'] = d_feat

    c_feat = torch.zeros(g.number_of_nodes(), IC.shape[1]).to(device)
    c_feat[ID.shape[0]: ID.shape[0] + IC.shape[0], :] = torch.from_numpy(IC.astype('float32')).to(device)
    g.ndata['c_feat'] = c_feat

    disease_ids = list(range(1, ID.shape[0] + 1))  # 1...383
    circrna_ids = list(range(1, IC.shape[0] + 1))  # 1...495

    g.add_edges(torch.tensor(known_associations['circrna'].values).to(device),
                torch.tensor(known_associations['diseases'].values).to(device))

    g = dgl.add_reverse_edges(g)  # 添加反向边，即转为无向图
    return g, IC, ID


def load_cda_dataset_only_known (directory, device):  # 有负边
    IC_DO = scio.loadmat(directory + 'DO/integrated similarities/integrated_circ_sim.mat')['integrated_circ_sim']
    ID_DO = scio.loadmat(directory + 'DO/integrated similarities/integrated_dise_sim.mat')['integrated_dise_sim']

    IC_G = scio.loadmat(directory + 'Gaussian interaction profile kernel/circ_gipk.mat')['circ_gipk']
    ID_G = scio.loadmat(directory + 'Gaussian interaction profile kernel/dis_gipk.mat')['dis_gipk']

    IC_sequence = scio.loadmat(directory + 'CircRNA sequence similarity/Levenshtein/LevenshteinSimilar.mat')['LevenshteinSimilar']

    IC_mesh = scio.loadmat(directory + 'MeSH/integrated similarities/integrated_circ_sim.mat')['integrated_circ_sim']
    ID_mesh = scio.loadmat(directory + 'MeSH/integrated similarities/integrated_dise_sim.mat')['integrated_dise_sim']

    IC = np.hstack((IC_mesh,IC_G))
    ID = np.hstack((ID_mesh, ID_G))

    associations = pd.read_excel(directory + 'Association Matrix.xlsx', sheet_name='Association Matrix',header=None)


    known_asss = associations.where(associations > 0).stack()  # 筛选数据

    known_associations = pd.concat([pd.DataFrame([[known_asss.index[i][0],known_asss.index[i][1]+IC.shape[0]]],columns=['circrna','diseases']) for i in range(known_asss.__len__())])

    g = dgl.DGLGraph().to(device)
    g.add_nodes(IC.shape[0] + ID.shape[0])
    node_type = torch.zeros(g.number_of_nodes(), dtype=torch.int64).to(device)
    node_type[: ID.shape[0]] = 1
    g.ndata['type'] = node_type

    d_feat = torch.zeros(g.number_of_nodes(), ID.shape[1]).to(device)  # [878,383]
    d_feat[: ID.shape[0], :] = torch.from_numpy(ID.astype('float32'))
    g.ndata['d_feat'] = d_feat

    c_feat = torch.zeros(g.number_of_nodes(), IC.shape[1]).to(device)
    c_feat[ID.shape[0]: ID.shape[0] + IC.shape[0], :] = torch.from_numpy(IC.astype('float32')).to(device)
    g.ndata['c_feat'] = c_feat

    disease_ids = list(range(1, ID.shape[0] + 1))  # 1...383
    circrna_ids = list(range(1, IC.shape[0] + 1))  # 1...495

    g.add_edges(torch.tensor(known_associations['circrna'].values).to(device),torch.tensor(known_associations['diseases'].values).to(device))

    g = dgl.add_reverse_edges(g)  # 添加反向边，即转为无向图
    split_edge = get_split_edge(g=g, train_ratio=0.85, val_ratio=0.05, test_ratio=0.1)
    return g, split_edge


def get_split_edge(g, train_idx, test_idx):
    src, dst = g.edges()
    mask = src < dst  # 前一半为true，mask[606]=true, mask[607]=false
    src, dst = src[mask], dst[mask]


    s,d = src[test_idx[:math.floor(len(test_idx)/2)]],dst[test_idx[:math.floor(len(test_idx)/2)]]
    val_pos_edge_index = torch.cat([torch.stack([s, d], dim=0),torch.stack([d, s], dim=0)],dim=1)
    s,d = src[test_idx[math.floor(len(test_idx)/2):]],dst[test_idx[math.floor(len(test_idx)/2):]]
    test_pos_edge_index = torch.cat([torch.stack([s, d], dim=0),torch.stack([d, s], dim=0)],dim=1)
    s, d = src[train_idx], dst[train_idx]
    train_pos_edge_index = torch.cat([torch.stack([s, d], dim=0),torch.stack([d, s], dim=0)],dim=1)
    # Negative edges (cannot guarantee (i,j) and (j,i) won't both appear)
    n_src, n_dst = dgl.sampling.global_uniform_negative_sampling(g, int(g.num_edges()/2))
    val_neg_edge_index = torch.cat([torch.stack([n_src[test_idx[:math.floor(len(test_idx)/2)]], n_dst[test_idx[:math.floor(len(test_idx)/2)]]]), torch.stack([n_dst[test_idx[:math.floor(len(test_idx)/2)]], n_src[test_idx[:math.floor(len(test_idx)/2)]]], dim=0)],dim=1)
    test_neg_edge_index = torch.cat([torch.stack([n_src[test_idx[math.floor(len(test_idx)/2)]:], n_dst[test_idx[math.floor(len(test_idx)/2)]:]]),torch.stack([ n_dst[test_idx[math.floor(len(test_idx)/2)]:], n_src[test_idx[math.floor(len(test_idx)/2)]:]], dim=0)],dim=1)
    train_neg_edge_index = torch.cat([torch.stack([n_src[train_idx], n_dst[train_idx]]),torch.stack([n_dst[train_idx], n_src[train_idx]], dim=0)],dim=1)

    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    split_edge['train']['edge'] = train_pos_edge_index.t()
    split_edge['train']['edge_neg'] = train_neg_edge_index.t()
    split_edge['valid']['edge'] = val_pos_edge_index.t()
    split_edge['valid']['edge_neg'] = val_neg_edge_index.t()
    split_edge['test']['edge'] = test_pos_edge_index.t()
    split_edge['test']['edge_neg'] = test_neg_edge_index.t()

    return split_edge


def get_split_old_edge(g, train_ratio, val_ratio, test_ratio):
    src, dst = g.edges()
    mask = src < dst  # 前一半为true，mask[606]=true, mask[607]=false
    src, dst = src[mask], dst[mask]

    n_v = int(math.floor(val_ratio * src.size(0)))
    n_t = int(math.floor(test_ratio * src.size(0)))

    perm = torch.randperm(src.size(0))
    src, dst = src[perm], dst[perm]
    s, d = src[:n_v], dst[:n_v]
    val_pos_edge_index = torch.cat([torch.stack([s, d], dim=0),torch.stack([d, s], dim=0)],dim=1)
    s, d = src[n_v:n_v + n_t], dst[n_v:n_v + n_t]
    test_pos_edge_index = torch.cat([torch.stack([s, d], dim=0),torch.stack([d, s], dim=0)],dim=1)
    s, d = src[n_v + n_t:], dst[n_v + n_t:]
    train_pos_edge_index = torch.cat([torch.stack([s, d], dim=0),torch.stack([d, s], dim=0)],dim=1)
    # Negative edges (cannot guarantee (i,j) and (j,i) won't both appear)
    n_src, n_dst = dgl.sampling.global_uniform_negative_sampling(g, int(g.num_edges()/2))
    val_neg_edge_index = torch.cat([torch.stack([n_src[:n_v], n_dst[:n_v]]), torch.stack([n_dst[:n_v], n_src[:n_v]], dim=0)],dim=1)
    test_neg_edge_index = torch.cat([torch.stack([n_src[n_v:n_v + n_t], n_dst[n_v:n_v + n_t]]),torch.stack([ n_dst[n_v:n_v + n_t], n_src[n_v:n_v + n_t]], dim=0)],dim=1)
    train_neg_edge_index = torch.cat([torch.stack([n_src[n_v + n_t:], n_dst[n_v + n_t:]]),torch.stack([n_dst[n_v + n_t:], n_src[n_v + n_t:]], dim=0)],dim=1)

    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    split_edge['train']['edge'] = train_pos_edge_index.t()
    split_edge['train']['edge_neg'] = train_neg_edge_index.t()
    split_edge['valid']['edge'] = val_pos_edge_index.t()
    split_edge['valid']['edge_neg'] = val_neg_edge_index.t()
    split_edge['test']['edge'] = test_pos_edge_index.t()
    split_edge['test']['edge_neg'] = test_neg_edge_index.t()

    return split_edge


def load_ogb_dataset(dataset):

    dataset = DglLinkPropPredDataset(name=dataset)
    split_edge = dataset.get_edge_split()
    graph = dataset[0]

    return graph, split_edge


def drnl_node_labeling(subgraph, src, dst):

    adj = subgraph.adj().to_dense().numpy()
    src, dst = (dst, src) if src > dst else (src, dst)

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst - 1)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = dist2src + dist2dst
    dist_over_2, dist_mod_2 = dist // 2, dist % 2

    z = 1 + torch.min(dist2src, dist2dst)
    z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
    z[src] = 1.
    z[dst] = 1.
    z[torch.isnan(z)] = 0.

    return z.to(torch.long)


def de_node_labeling(subgraph, src, dst, max_dist=3):

    adj = subgraph.adj().to_dense().numpy()

    src, dst = (dst, src) if src > dst else (src, dst)

    dist, predecessors = shortest_path(adj, directed=False, unweighted=True, indices=[src, dst])
    dist = torch.from_numpy(dist)

    dist[dist > max_dist] = max_dist
    dist[torch.isnan(dist)] = max_dist + 1

    return dist.to(torch.long).t()


def de_plus_node_labeling(subgraph, src, dst, max_dist=100):

    adj = subgraph.adj().to_dense().numpy()

    src, dst = (dst, src) if src > dst else (src, dst)

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src= shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst= shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst-1)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = torch.cat([dist2src.view(-1, 1), dist2dst.view(-1, 1)], 1)
    dist[dist > max_dist] = max_dist
    dist[torch.isnan(dist)] = max_dist + 1

    return dist.to(torch.long)


def best_threshold(fpr,tpr,thresholds):
    ths = thresholds
    diffs = list(tpr - fpr)
    max_diff = max(diffs)
    optimal_idx = diffs.index(max_diff)
    optimal_th = ths[optimal_idx]
    return optimal_th


def evaluate_hits(name, pos_pred, neg_pred, K):

    evaluator = Evaluator(name)
    evaluator.K = K
    hits = evaluator.eval({
        'y_pred_pos': pos_pred,
        'y_pred_neg': neg_pred,
    })[f'hits@{K}']

    return hits




def evaluate_roc(y_pred_pos,y_pred_neg):
    y_pred_pos_numpy = y_pred_pos.cpu().detach().numpy()
    y_pred_neg_numpy = y_pred_neg.cpu().detach().numpy()
    y_true = np.concatenate([np.ones(len(y_pred_pos_numpy)), np.zeros(len(y_pred_neg_numpy))]).astype(np.int32)
    y_pred = np.concatenate([y_pred_pos_numpy, y_pred_neg_numpy])
    # y_pred = Abnormal_data_handling(y_pred)

    fpr_, tpr_, thresholds = metrics.roc_curve(y_true, y_pred)
    best_th = best_threshold(fpr_,tpr_,thresholds)
    auc = metrics.roc_auc_score(y_true, y_pred)

    pred_label = [0 if j < best_th else 1 for j in y_pred]

    acc = metrics.accuracy_score(y_true, pred_label)
    pre = metrics.precision_score(y_true, pred_label)
    recall = metrics.recall_score(y_true, pred_label)

    f1 = metrics.f1_score(y_true, pred_label)
    precision_, recall_, _ = metrics.precision_recall_curve(y_true, y_pred)
    prc = metrics.auc(recall_, precision_)
    return acc, pre, recall, precision_, recall_, f1, auc, prc, fpr_, tpr_,

