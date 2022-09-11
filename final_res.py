
import os

import csv
import pickle
import numpy as np
import pandas as pd

tmp_res_path = './cikm22_data/tmp_res'
nn_pkl_path = './cikm22_data/nn_pkl'
final_res_path = './cikm22_data/final_res'

os.makedirs(tmp_res_path, exist_ok=True)
os.makedirs(nn_pkl_path, exist_ok=True)
os.makedirs(final_res_path, exist_ok=True)

lgb_file = os.path.join(tmp_res_path, 'lgb_prediction.csv')


nl_list = []
with open(lgb_file, "r") as csvfile:

    reader=csv.reader(csvfile)

    for line in reader:


        nl = line

        nl_list.append(nl)


did = 11
repeat = 10
_, _, _, _, test_y_prob, test_y_true = \
    pickle.load(open(os.path.join(nn_pkl_path, 'k_fold_nn_res_repeat{}_{}.pkl'.format(repeat, did)), 'rb'))
test_y_pred = test_y_prob[:, 0]

print(test_y_pred)
nl_list1 = []
count = 0

for line in nl_list:

    if line[0] == str(did):

        print(line)
        nl = [item for item in line]
        nl[-1] = test_y_pred[count]
        count += 1
        print(nl)

    else:
        nl = line

    nl_list1.append(nl)
print('_')



repeat = 11
did = 5
train_y_prob, train_y_true, val_y_prob, val_y_true, test_y_prob, test_y_true = \
    pickle.load(open(os.path.join(nn_pkl_path, 'k_fold_nn_res_repeat{}_{}.pkl'.format(repeat, did)), 'rb'))
test_y_pred = np.argmax(test_y_prob, axis=1)

print(test_y_pred)
nl_list2 = []
count = 0

for line in nl_list1:

    if line[0] == str(did):

        print(line)
        nl = [item for item in line]
        nl[-1] = test_y_pred[count]
        count += 1
        print(nl)

    else:
        nl = line

    nl_list2.append(nl)
print('_')


out_path = os.path.join(final_res_path, 'prediction_final.csv')

with open(out_path, 'a') as file:
    for line in nl_list2:


        file.write(','.join([str(_) for _ in line]) + '\n')