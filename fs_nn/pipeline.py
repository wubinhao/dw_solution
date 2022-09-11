
import os
import random
import numpy as np

from yacs.config import CfgNode
from sklearn.model_selection import KFold, StratifiedKFold


import torch
from torch_geometric.loader import DataLoader
from .model.graph_level import GNN_Net_Graph
from .trainer.graphtrainer import GraphMiniBatchTrainer
from .trainer.monitor import Monitor



# data_path = r'G:\work\code\competition\cikm22\datas\CIKM22Competition'
# monitor_outdir = r'G:\work\code\competition\cikm22\tmp'


data_path = './cikm22_data/CIKM22Competition'
monitor_outdir = './cikm22_data/monitor_tmp'
os.makedirs(monitor_outdir, exist_ok=True)

backend = 'torch'


did_meta = {
    1: {'out_channels': 2, 'task': 'graphClassification', 'criterion': 'CrossEntropyLoss', 'base': 0.263789},
    2: {'out_channels': 2, 'task': 'graphClassification', 'criterion': 'CrossEntropyLoss', 'base': 0.289617},
    3: {'out_channels': 2, 'task': 'graphClassification', 'criterion': 'CrossEntropyLoss', 'base': 0.355404},
    4: {'out_channels': 2, 'task': 'graphClassification', 'criterion': 'CrossEntropyLoss', 'base': 0.176471},
    5: {'out_channels': 2, 'task': 'graphClassification', 'criterion': 'CrossEntropyLoss', 'base': 0.396825},
    6: {'out_channels': 2, 'task': 'graphClassification', 'criterion': 'CrossEntropyLoss', 'base': 0.261580},
    7: {'out_channels': 2, 'task': 'graphClassification', 'criterion': 'CrossEntropyLoss', 'base': 0.302378},
    8: {'out_channels': 2, 'task': 'graphClassification', 'criterion': 'CrossEntropyLoss', 'base': 0.211538},

    9: {'out_channels': 1, 'task': 'graphRegression', 'criterion': 'MSELoss', 'base': 0.059199},
    10: {'out_channels': 10, 'task': 'graphRegression', 'criterion': 'MSELoss', 'base': 0.007083},
    11: {'out_channels': 1, 'task': 'graphRegression', 'criterion': 'MSELoss', 'base': 0.734011},
    12: {'out_channels': 1, 'task': 'graphRegression', 'criterion': 'MSELoss', 'base': 1.361326},
    13: {'out_channels': 12, 'task': 'graphRegression', 'criterion': 'MSELoss', 'base': 0.004389},

}


class NNPipeline:
    def __init__(self, pp_config, did):
        self.pp_config = pp_config
        self.did = did

        self.model_param = None
        self.monitor_config = None
        self.trainer_config = None

        batch_size = self.pp_config['batch_size']

        self.dataloader_dict = get_data_loader(self.did, batch_size)
        self.parse_cfg()

    def parse_cfg(self):
        self.update_model_param()
        self.update_monitor_config()
        self.update_trainer_config()

    def update_model_param(self):
        tmp_data = next(iter(self.dataloader_dict['train']))
        in_channels = tmp_data.x.shape[-1]
        out_channels = did_meta[self.did]['out_channels']
        hidden = self.pp_config['hidden']
        max_depth = self.pp_config['max_depth']
        dropout = self.pp_config['dropout']
        model_type = self.pp_config['model_type']
        pooling = self.pp_config['pooling']

        self.model_param = {
            'in_channels': in_channels,
            'out_channels': out_channels,
            'hidden': hidden,
            'max_depth': max_depth,
            'dropout': dropout,
            'model_type': model_type,
            'pooling': pooling,
        }

    def update_monitor_config(self):
        monitor_config = CfgNode()
        monitor_config['outdir'] = monitor_outdir
        monitor_config['wandb'] = CfgNode({'use': False, })
        self.monitor_config = monitor_config

    def update_trainer_config(self):
        batch_size = self.pp_config['batch_size']
        get_metrics = lambda dataset_id: 'acc' if dataset_id <= 8 else 'mse'
        metrics = get_metrics(self.did)
        grad_clip = self.pp_config['grad_clip']
        epoch = self.pp_config['epoch']
        lr = self.pp_config['lr']
        optimizer = self.pp_config.get('optimizer', 'SGD')
        print('optimizer : ', optimizer)
        regularizer = self.pp_config.get('regularizer', '')
        print('regularizer : ', regularizer)
        mu = self.pp_config.get('mu', 0)
        print('mu : ', mu)
        weight_decay = self.pp_config.get('weight_decay', None)
        print('weight_decay : ', weight_decay)

        if weight_decay is not None:
            opt_dic = {'type': optimizer, 'lr': lr, 'weight_decay': weight_decay}
        else:
            opt_dic = {'type': optimizer, 'lr': lr, }

        config = CfgNode()
        config['eval'] = CfgNode({'metrics': [metrics], 'count_flops': False})
        config['backend'] = backend

        if 'criterion' in self.pp_config:
            criterion = self.pp_config['criterion']
        else:
            criterion = did_meta[self.did]['criterion']
        print('criterion : ', criterion)
        config['criterion'] = CfgNode({'type': criterion})
        config['regularizer'] = CfgNode({'type': regularizer, 'mu': mu})
        config['grad'] = CfgNode({'grad_clip': grad_clip})
        config['train'] = CfgNode(
            {'local_update_steps': epoch, 'batch_or_epoch': 'epoch', 'optimizer': CfgNode(opt_dic)})
        config['data'] = CfgNode({'batch_size': batch_size, 'drop_last': False})
        config['finetune'] = CfgNode({'before_eval': False, })
        config['federate'] = CfgNode(
            {'mode': 'standalone', 'use_diff': False, 'share_local_model': False, 'method': 'local'})
        config['model'] = CfgNode({'task': did_meta[self.did]['task'], })
        self.trainer_config = config

    def run(self):

        model = get_model(**self.model_param)
        monitor = Monitor(self.monitor_config)

        trainer = get_trainer(model, monitor, self.dataloader_dict, self.trainer_config, )

        sample_size, model_para_all, results = trainer.train_early_stop(target_data_split_name="train",
                                                                        valid_mode='val',
                                                                        early_stop_rounds=50,
                                                                        base_score=did_meta[self.did]['base']
                                                                        )

        print(sample_size)
        print(model_para_all)
        print(results)

        eval_metrics = trainer.evaluate(target_data_split_name='val')

        print('val', eval_metrics)

        print('val', trainer.ctx.val_y_prob.shape, trainer.ctx.val_y_true.shape)
        val_y_prob = trainer.ctx.val_y_prob
        val_y_true = trainer.ctx.val_y_true

        test_eval_metrics = trainer.evaluate(target_data_split_name='test')
        print('test', test_eval_metrics)
        print('test', trainer.ctx.test_y_prob.shape, trainer.ctx.test_y_true.shape)

        test_y_prob = trainer.ctx.test_y_prob
        test_y_true = trainer.ctx.test_y_true

        return val_y_prob, val_y_true, test_y_prob, test_y_true

    def run_kfold(self, n_splits=5, random_state=2022):
        batch_size = self.pp_config['batch_size']

        dataloader_kfold_dict, val_idx_list, val_num = \
            get_data_loader_kfold(self.did,
                                  batch_size,
                                  n_splits=n_splits,
                                  random_state=random_state
                                  )

        kfold_res = []
        for fid in dataloader_kfold_dict:
            dataloader_dict = dataloader_kfold_dict[fid]

            model = get_model(**self.model_param)
            monitor = Monitor(self.monitor_config)

            trainer = get_trainer(model, monitor, dataloader_dict, self.trainer_config, )
            sample_size, model_para_all, results = trainer.train_early_stop(target_data_split_name="train",
                                                                            valid_mode='val',
                                                                            early_stop_rounds=50,
                                                                            base_score=did_meta[self.did]['base']
                                                                            )
            print(sample_size)
            print(model_para_all)
            print(results)

            eval_metrics = trainer.evaluate(target_data_split_name='val')

            print('val', eval_metrics)
            print('val', trainer.ctx.val_y_prob.shape, trainer.ctx.val_y_true.shape)
            val_y_prob = trainer.ctx.val_y_prob
            val_y_true = trainer.ctx.val_y_true

            test_eval_metrics = trainer.evaluate(target_data_split_name='test')
            print('test', test_eval_metrics)
            print('test', trainer.ctx.test_y_prob.shape, trainer.ctx.test_y_true.shape)

            test_y_prob = trainer.ctx.test_y_prob
            test_y_true = trainer.ctx.test_y_true

            kfold_res.append((val_y_prob, val_y_true, test_y_prob, test_y_true))

        all_val_num = np.sum([d[0].shape[0] for d in kfold_res])
        out_channels = kfold_res[0][0].shape[1]
        assert out_channels == self.model_param['out_channels']
        print('all_val_num : ', all_val_num)

        all_y_prob = np.zeros((all_val_num, out_channels))
        if self.did <= 8:
            all_y_true = np.zeros(all_val_num)
        else:
            all_y_true = np.zeros((all_val_num, out_channels))

        test_y_prob = None
        test_y_true = None

        for fid in dataloader_kfold_dict:
            val_idx = val_idx_list[fid]
            val_y_prob_, val_y_true_, test_y_prob_, test_y_true_ = kfold_res[fid]
            all_y_prob[val_idx] = val_y_prob_
            all_y_true[val_idx] = val_y_true_
            if test_y_true is None:
                test_y_true = test_y_true_
            if test_y_prob is None:
                test_y_prob = test_y_prob_ / n_splits
            else:
                test_y_prob += test_y_prob_ / n_splits

        val_y_prob, val_y_true = all_y_prob[-val_num:], all_y_true[-val_num:]
        train_y_prob, train_y_true = all_y_prob[:-val_num], all_y_true[:-val_num]

        return train_y_prob, train_y_true, val_y_prob, val_y_true, test_y_prob, test_y_true


def get_data_loader(did, batch_size):
    def worker_init(worked_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    data_dict = {}
    dataloader_dict = {}
    for split in ['train', 'val', 'test']:
        split_data = torch.load(os.path.join(data_path, str(did), f'{split}.pt'))
        data_dict[split] = split_data
        shuffle = True if split == 'train' else False

        # if split == 'train':
        #     node_feat_num = split_data[0].x.shape[1]

        g = torch.Generator()
        g.manual_seed(0)

        dataloader_dict[split] = DataLoader(split_data, batch_size=batch_size, shuffle=shuffle,
                                            worker_init_fn=worker_init, num_workers=0,
                                            # generator=g,
                                            )
    # dataloader_dict['num_label'] = 0

    return dataloader_dict


def get_data_loader_kfold(did, batch_size, n_splits=5, random_state=2022):
    data_dict = {}
    for split in ['train', 'val', 'test']:
        split_data = torch.load(os.path.join(data_path, str(did), f'{split}.pt'))
        data_dict[split] = split_data

    all_data = data_dict['train'] + data_dict['val']
    val_num = len(data_dict['val'])
    tr_idx_list, val_idx_list = [], []

    # fake_all_data = np.array(range(len(all_data)))
    if did <= 8:
        all_y = [d.y.item() for d in all_data]
        all_y = np.array(all_y)
        print(all_y)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for tr_idx, val_idx in skf.split(all_data, all_y):
            tr_idx_list.append(tr_idx)
            val_idx_list.append(val_idx)
    else:

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for tr_idx, val_idx in kf.split(all_data):
            tr_idx_list.append(tr_idx)
            val_idx_list.append(val_idx)

    dataloader_kfold_dict = {}
    for fid in range(n_splits):
        tr_idx = tr_idx_list[fid]
        val_idx = val_idx_list[fid]

        dataloader_kfold_dict[fid] = {}

        tr_data = [all_data[idx] for idx in tr_idx]
        val_data = [all_data[idx] for idx in val_idx]
        test_data = [d for d in data_dict['test']]

        dataloader_kfold_dict[fid]['train'] = DataLoader(tr_data, batch_size=batch_size, shuffle=True)
        dataloader_kfold_dict[fid]['val'] = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        dataloader_kfold_dict[fid]['test'] = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return dataloader_kfold_dict, val_idx_list, val_num


def get_model(in_channels,
              out_channels,
              hidden,
              max_depth,
              dropout,
              model_type,
              pooling
              ):
    model = GNN_Net_Graph(in_channels,
                          out_channels,
                          hidden=hidden,
                          max_depth=max_depth,
                          dropout=dropout,
                          gnn=model_type,
                          pooling=pooling)
    return model


def get_trainer(model, monitor, dataloader_dict, config, device='cuda:0'):
    trainer = GraphMiniBatchTrainer(model=model,
                                    data=dataloader_dict,
                                    device=device,
                                    config=config,
                                    monitor=monitor)
    return trainer


# torch.optim.SGD