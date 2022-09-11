

import warnings


#方法二：
warnings.filterwarnings("ignore")  #忽略告警


import json
import os
import torch
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor

from sklearn.metrics import accuracy_score

import networkx as nx

import pickle


data_path = './cikm22_data/CIKM22Competition'
# nx_stat_store = './cikm22_data/nx_stat'
tmp_res_path = './cikm22_data/tmp_res'

# os.makedirs(nx_stat_store, exist_ok=True)
os.makedirs(tmp_res_path, exist_ok=True)


submit = True
debug = False


class LgbCls:
    def __init__(self):

        self.models = [
            LGBMClassifier(random_state=2022, n_jobs=4),

        ]

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)

    def predict_proba(self, X):
        pred = None
        for model in self.models:
            model_prob = model.predict_proba(X)
            if pred is None:
                pred = model_prob / len(self.models)
            else:
                pred += model_prob / len(self.models)

        return pred


class LgbReg:
    def __init__(self):
        self.models = [

            LGBMRegressor(random_state=2023, n_jobs=4, n_estimators=1000, num_leaves=38),
            LGBMRegressor(random_state=2023, n_jobs=4, n_estimators=1000, num_leaves=40),
            LGBMRegressor(random_state=2023, n_jobs=4, n_estimators=1000, num_leaves=38, learning_rate=0.09),

        ]

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)

    def predict(self, X):
        pred = None
        for model in self.models:
            model_prob = model.predict(X)
            if pred is None:
                pred = model_prob / len(self.models)
            else:
                pred += model_prob / len(self.models)

        return pred


def get_metric(score_dict):
    bl_err = {1: 0.263789, 2: 0.289617, 3: 0.355404, 4: 0.176471,
              5: 0.396825, 6: 0.261580, 7: 0.302378, 8: 0.211538,
              9: 0.059199, 10: 0.007083, 11: 0.734011, 12: 1.361326,
              13: 0.004389}

    # err_dict = {k: 1 - v for k, v in acc_dict.items()}
    imp_score_dict = {k: (bl_err[k] - score_dict[k]) / bl_err[k] * 100 for k in score_dict}

    imp_score_list = list(imp_score_dict.values())

    imp_score = np.sum(imp_score_list) / len(imp_score_list)
    print(imp_score_dict)
    print(imp_score)
    return imp_score, imp_score_dict


def get_score(did, task, config, ):
    data_dict = {}
    drop_cols = []

    data_split_list = ['train', 'val', ]
    if submit:
        data_split_list = ['train', 'val', 'test']
    for split in data_split_list:
        split_data = torch.load(os.path.join(data_path, str(did), f'{split}.pt'))
        data_dict[split] = split_data


        y = []

        stat_dict = {}
        graph_meta = []
        x_node_mean = []
        x_node_max = []
        x_node_min = []
        x_node_sum = []
        x_edge_mean = []


        for k in config:
            stat_dict[k] = []


        # already_exist = []
        # for k in config:
        #     file_path = os.path.join(nx_stat_store, f'{did}-{split}-{k}.pkl')
        #     if os.path.exists(file_path):
        #         stat_dict[k] = pickle.load(open(file_path, 'rb'))
        #         already_exist.append(k)
        #         print('already exist : ', k,)

        for gid, d in enumerate(split_data):

            if debug and gid > 100:
                continue

            node_matrix = d.x.numpy()

            num_nodes = d.num_nodes
            num_edges = d.num_edges
            graph_meta.append([num_nodes, num_edges])

            node_mean = node_matrix.mean(axis=0)
            node_max = node_matrix.max(axis=0)
            node_min = node_matrix.min(axis=0)
            node_sum = node_matrix.sum(axis=0)


            x_node_mean.append(node_mean)
            x_node_max.append(node_max)
            x_node_min.append(node_min)
            x_node_sum.append(node_sum)

            if d.edge_attr is not None:
                edge_matrix = d.edge_attr.numpy()
                d_edge_mean = edge_matrix.mean(axis=0)
                x_edge_mean.append(d_edge_mean)

            dy = d.y.numpy()[0]

            y.append(dy)

            edge_index = d.edge_index.numpy().T
            G = nx.Graph()

            edge_index = [(l, r) for l, r in edge_index]
            G.add_edges_from(edge_index)

            for k in config:
                # if k in already_exist:
                #     continue
                func = k
                # print(func)
                node_scores = eval(f'nx.algorithms.{func}')(G)
                ns_sr = pd.Series(node_scores)
                ns_stat = [np.mean(ns_sr), np.sum(ns_sr), np.max(ns_sr), np.min(ns_sr), np.std(ns_sr)]
                stat_dict[k].append(ns_stat)

        # for k in config:
        #     if k in already_exist:
        #         continue
        #     file_path = os.path.join(nx_stat_store, f'{did}-{split}-{k}.pkl')
        #     pickle.dump(stat_dict[k], open(file_path, 'wb'))

        df_dict = {}
        for k in config:
            nss_df = pd.DataFrame(stat_dict[k], )
            nss_df.columns = [f'{k}-mean', f'{k}-sum', f'{k}-max', f'{k}-min', f'{k}-std', ]
            df_dict[k] = nss_df

        graph_meta_df = pd.DataFrame(graph_meta)
        graph_meta_df.columns = ['num_nodes', 'num_edges']

        x_node_mean = pd.DataFrame(x_node_mean)
        x_node_mean.columns = [f'node_mean_{i}' for i in range(x_node_mean.shape[1])]

        x_node_max = pd.DataFrame(x_node_max)
        x_node_max.columns = [f'node_max_{i}' for i in range(x_node_max.shape[1])]

        x_node_min = pd.DataFrame(x_node_min)
        x_node_min.columns = [f'node_min_{i}' for i in range(x_node_min.shape[1])]

        x_node_sum = pd.DataFrame(x_node_sum)
        x_node_sum.columns = [f'node_min_{i}' for i in range(x_node_sum.shape[1])]

        x_edge_mean = pd.DataFrame(x_edge_mean, )
        x_edge_mean.columns = [f'edge_{i}' for i in range(x_edge_mean.shape[1])]


        node_stat_df_list = [df_dict[k] for k in config]
        x = pd.concat([x_node_mean, x_edge_mean, graph_meta_df] + node_stat_df_list, axis=1)


        if split == 'train':
            for col in x:
                if x[col].nunique() == 1:
                    drop_cols.append(col)
        if drop_cols:
            x = x.drop(columns=drop_cols)

        y = pd.DataFrame(y)
        data_dict[split] = {'x': x, 'y': y}

    score = None
    test_pred = None
    if task == 'cls':
        score, test_pred = cls_train(data_dict)
    elif task == 'reg':
        score, test_pred = reg_train(data_dict)

    return score, test_pred




def cls_train(data_dict):

    tr_y = data_dict['train']['y'].values
    print(tr_y.shape)
    model = LgbCls()

    model.fit(data_dict['train']['x'], tr_y[:, 0])

    val_pred_prob = model.predict_proba(data_dict['val']['x'])[:, 1]


    test_pred_label = None
    y_true = data_dict['val']['y'].values[:, 0]

    if submit:

        model_all = LgbCls()

        model_all.fit(data_dict['train']['x'].append(data_dict['val']['x']),
                      np.hstack([tr_y[:, 0], y_true]))

        test_pred_prob = model_all.predict_proba(data_dict['test']['x'])[:, 1]
        test_pred_label = test_pred_prob > 0.5
        test_pred_label = test_pred_label.astype(int)

    val_pred_label = val_pred_prob > 0.5
    y_true = data_dict['val']['y'].values[:, 0]

    acc = accuracy_score(y_true, val_pred_label)
    err = 1 - acc
    print('acc : ', acc, 'err : ', err)
    return err, test_pred_label


def reg_train(data_dict):

    tr_y = data_dict['train']['y'].values
    print(tr_y.shape)

    val_pred_list = []
    test_pred_list = []
    y_true = data_dict['val']['y'].values

    for idx in range(tr_y.shape[1]):
        model = LgbReg()

        model.fit(data_dict['train']['x'], tr_y[:, idx])

        val_pred = model.predict(data_dict['val']['x'])
        val_pred_list.append(val_pred.reshape(-1, 1))

        if submit:
            model_all = LgbReg()

            model_all.fit(data_dict['train']['x'].append(data_dict['val']['x']),
                          np.hstack([tr_y[:, idx], y_true[:, idx]]))


            test_pred = model_all.predict(data_dict['test']['x'])
            test_pred_list.append(test_pred.reshape(-1, 1))

    y_prob = np.hstack(val_pred_list)

    print('y_prob ', y_prob.shape)
    print('y_true', y_true.shape)
    mse = np.mean(np.power(y_true - y_prob, 2))
    all_test_pred = None

    if submit:
        all_test_pred = np.hstack(test_pred_list)
    print(mse)
    return mse, all_test_pred


get_task_type = lambda did : 'cls' if did <= 8 else 'reg'


# cfg is from feat_select_nx_1/te2_kfold
cfg_store = {
    1: {k: True for k in ['subgraph_centrality_exp', 'pagerank', 'pagerank_scipy', 'betweenness_centrality', 'katz_centrality', 'core_number', 'number_of_cliques']},
    2: {k: True for k in ['subgraph_centrality_exp', 'average_neighbor_degree', 'clustering', 'node_clique_number', 'square_clustering', 'triangles', 'harmonic_centrality', 'degree_centrality', 'subgraph_centrality', 'katz_centrality_numpy']},
    3: {k: True for k in ['subgraph_centrality_exp', 'load_centrality', 'average_neighbor_degree', 'closeness_centrality', 'onion_layers']},
    4: {k: True for k in ['number_of_cliques', 'clustering', 'node_clique_number', 'square_clustering', 'triangles', 'closeness_centrality', 'load_centrality']},
    5: {k: True for k in ['average_neighbor_degree', 'greedy_color', 'core_number']},
    6: {k: True for k in
        ['subgraph_centrality_exp', 'square_clustering', 'harmonic_centrality', 'clustering', 'subgraph_centrality']},
    7: {k: True for k in
        ['load_centrality', 'harmonic_centrality', 'square_clustering', 'greedy_color', 'degree_centrality',
         'onion_layers', 'clustering']},
    8: {k: True for k in
        ['katz_centrality', 'subgraph_centrality_exp', 'clustering', 'core_number', 'node_clique_number']},

    9: {k: True for k in
        ['subgraph_centrality', 'betweenness_centrality', 'core_number', 'onion_layers', 'pagerank', 'find_cores',
         'pagerank_scipy', 'betweenness_centrality_source', 'number_of_cliques', 'average_neighbor_degree']},
    10: {k: True for k in
         ['subgraph_centrality', 'betweenness_centrality', 'number_of_cliques', 'onion_layers', 'core_number',
          'degree_centrality', 'pagerank', 'square_clustering', 'triangles', 'clustering']},
    11: {k: True for k in ['betweenness_centrality', 'onion_layers', 'subgraph_centrality']},
    12: {k: True for k in ['degree_centrality', 'onion_layers', 'node_clique_number']},
    13: {k: True for k in
         ['closeness_centrality', 'subgraph_centrality', 'average_neighbor_degree', 'load_centrality', 'onion_layers',
          'square_clustering', 'core_number', 'number_of_cliques', 'clustering', 'degree_centrality']},

}


score_dict = {}

test_pred_dict = {}

for did in range(1, 14):
    print('######'*7, did)

    task_ = get_task_type(did)
    config = cfg_store[did]

    score, test_pred = get_score(did, task_, config)
    score_dict[did] = score
    test_pred_dict[did] = test_pred


metric = get_metric(score_dict)

if submit:
    # out_path = r'G:\work\code\competition\cikm22\submit\prediction_0911_3.csv'
    out_path = os.path.join(tmp_res_path, 'lgb_prediction.csv')
    for did in range(1, 14):
        test_pred = test_pred_dict[did]

        with open(out_path, 'a') as file:
            for y_ind in range(test_pred.shape[0]):
                if did <= 8:
                    line = [did, y_ind] + [test_pred[y_ind]]
                else:
                    line = [did, y_ind] + list(test_pred[y_ind])

                file.write(','.join([str(_) for _ in line]) + '\n')
