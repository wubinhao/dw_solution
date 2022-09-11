
import os

import pickle

import numpy as np
from fs_nn.pipeline import NNPipeline
from sklearn.metrics import accuracy_score


data_path = './cikm22_data/CIKM22Competition'
monitor_outdir = './cikm22_data/monitor_tmp'
nn_pkl_path = './cikm22_data/nn_pkl'

os.makedirs(monitor_outdir, exist_ok=True)
os.makedirs(nn_pkl_path, exist_ok=True)


def get_acc(val_y_true_, val_y_prob_):
    acc = accuracy_score(val_y_true_, np.argmax(val_y_prob_, axis=1))
    return acc

did = 5
pp_cfg = {
    'model_type': 'gin',
    'pooling': 'mean',
    'hidden': 64,
    'max_depth': 2,
    'dropout': 0.1,

    'grad_clip': -1,
    'epoch': 1000,
    'lr':  0.03,
    'batch_size': 32,

}


train_y_true_ = None
train_y_prob_ = None

val_y_true_ = None
val_y_prob_ = None
test_y_true_ = None
test_y_prob_ = None



repeat = 11
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

    score = get_acc(val_y_true, val_y_prob)

    print(val_y_prob)
    print( i, score)

print(val_y_prob_)

score = get_acc(val_y_true_, val_y_prob_)
print(score)

pickle.dump((train_y_prob_, train_y_true_, val_y_prob_, val_y_true_, test_y_prob_, test_y_true_),
            open(os.path.join(nn_pkl_path, 'k_fold_nn_res_repeat{}_{}.pkl'.format(repeat, did)), 'wb'))