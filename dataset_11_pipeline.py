

import os

import pickle

import numpy as np
from fs_nn.pipeline import NNPipeline


data_path = './cikm22_data/CIKM22Competition'
monitor_outdir = './cikm22_data/monitor_tmp'
nn_pkl_path = './cikm22_data/nn_pkl'

os.makedirs(monitor_outdir, exist_ok=True)
os.makedirs(nn_pkl_path, exist_ok=True)


did = 11
pp_cfg = {
    'model_type': 'gin',
    'pooling': 'mean',
    'hidden': 128,
    'max_depth': 4,
    'dropout': 0.,

    'grad_clip': -1,
    'epoch': 1000,
    'lr': 0.02,
    'batch_size': 128,

}

train_y_true_ = None
train_y_prob_ = None

val_y_true_ = None
val_y_prob_ = None
test_y_true_ = None

test_y_prob_ = None

repeat = 10
n_splits = 5

for i in range(repeat):
    print('#############', i)
    pp = NNPipeline(pp_cfg, did)
    train_y_prob, train_y_true, val_y_prob, val_y_true, test_y_prob, test_y_true = pp.run_kfold(n_splits=n_splits,
                                                                                                random_state=i+2021)
    if val_y_true_ is None:
        val_y_true_ = val_y_true
        test_y_true_ = test_y_true
        train_y_true_ = train_y_true

    if val_y_prob_ is None:
        val_y_prob_ = val_y_prob / repeat
        test_y_prob_ = test_y_prob / repeat
        train_y_prob_ = train_y_prob / repeat

    else:
        val_y_prob_ += val_y_prob / repeat
        test_y_prob_ += test_y_prob / repeat
        train_y_prob_ += train_y_prob / repeat

    mse = np.mean(np.power(val_y_true - val_y_prob, 2))
    print( i, mse)

mse = np.mean(np.power(val_y_true_ - val_y_prob_, 2))
print(mse)


pickle.dump((train_y_prob_, train_y_true_, val_y_prob_, val_y_true_, test_y_prob_, test_y_true_),
            open(os.path.join(nn_pkl_path, 'k_fold_nn_res_repeat{}_{}.pkl'.format(repeat, did)), 'wb'))