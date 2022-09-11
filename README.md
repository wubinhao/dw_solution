# dw2022 方案


## lgb_pipeline (运行几十分钟)

1. 数据集 1， 2，3，4，6，7，8，9，10，12，13 使用 lgb pipeline
2. 计算每个 graph 的 node featrure 平均值，edge feature 平均值，node 数量， edge 数量
3. 计算每个 graph, 一些 networkx 统计特征，每个数据集的 networkx 统计特征是不一样的, 见 'lgb_pipeline.py' 下的 cfg_store
4. 最后基于上述特征，训练 lightgbm 模型

## dataset_5_pipeline (运行十几分钟)
1. 数据集 5 使用 dataset_5_pipeline
2. 使用 gin 模型，使用 5折交叉验证 和 早停 训练，使用不同随机种子重复训练 11 次，对预测结果做平均

## dataset_11_pipeline (运行几个小时)
1. 数据集 5 使用 dataset_5_pipeline
2. 使用 gin 模型，使用 5折交叉验证 和 早停 训练，使用不同随机种子重复训练 10 次，对预测结果做平均

## final_res
1. 结合3个 pipeline 预测结果作为最终预测结果，(用 dataset_5_pipeline 和 dataset_11_pipeline 的预测结果 对 lgb_pipeline 的预测结果做一个替换)

## 运行
1. 将数据集放到 './cikm22_data/CIKM22Competition' 
2. run.sh
3. 最终预测结果会放到 './cikm22_data/final_res/prediction_final.csv'

## 注意
- 所有数据集都是独立建模的，互不影响

## 环境依赖
- Windows 10
- python 3.9
- 运行内存 32G
- cpu核数 6 个