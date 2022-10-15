import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import r2_score

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn.svm import LinearSVC

rng = np.random.RandomState(2021)

root_train = '/home/jljl/file/cas/df_info_train_FFA.csv'
root_valid = '/home/jljl/file/cas/df_info_valid_FFA.csv'

features = ['depth_sim', 'residual', 'rate']
norm = ['depth_sim', 'residual', 'rate', 'PM2.5']

df_train = pd.read_csv(root_train)
df_valid = pd.read_csv(root_valid)

scale = df_valid['PM2.5'].max() - df_valid['PM2.5'].min()

df_train[norm] = (df_train[norm] - df_train[norm].min()) / (df_train[norm].max() - df_train[norm].min())
df_valid[norm] = (df_valid[norm] - df_valid[norm].min()) / (df_valid[norm].max() - df_valid[norm].min())

train_features, train_label = df_train[features].values, df_train['PM2.5'].values
valid_features, valid_label = df_valid[features].values, df_valid['PM2.5'].values

# 合并训练集和验证集到同一numpy列表中，后续可以根据test_fold对列表设定训练集和验证集的范围
train_val_features = np.concatenate((train_features, valid_features), axis=0)
train_val_labels = np.concatenate((train_label, valid_label), axis=0)

regr = SVR()

test_fold = np.zeros(train_val_features.shape[0])  # 将所有index初始化为0,0表示第一轮的验证集
test_fold[:train_features.shape[0]] = -1  # 将训练集对应的index设为-1，表示永远不划分到验证集中
ps = PredefinedSplit(test_fold = test_fold)
# backup
# param_grid = {
#     "C": np.arange(5, 5.1, 0.1),
#     "degree": np.arange(3, 4, 1),
#     "epsilon": np.arange(0.01, 0.02, 0.01),
# }  # 输出信息，数字越大输出信息越多

param_grid= {
    "C" : np.arange(29, 100, 3),
    "degree": [3, 4, 5],
    "epsilon": [0.05, 0.08, 0.1, 0.2],
    "gamma": [0.01, 0.001]
}

print('train feature shape:', train_features.shape)
print('train label shape', train_label.shape)
grid = GridSearchCV(regr, param_grid=param_grid, cv=ps, n_jobs=-1, verbose=10)  # 包含交叉验证
grid.fit(train_val_features, train_val_labels)
print('best_params: ', grid.best_params_)
print('best_score:', grid.best_score_)
print('best_estimator:', grid.best_estimator_)
print('best_index:', grid.best_index_)

bst = grid.best_estimator_
valid_features = train_features
df_valid = df_train
# preds = bst.predict(valid_features)
preds = bst.predict(valid_features) * scale
list_pred = list(preds)
# list_gt = list(df_valid['PM2.5'])
list_gt = list(df_valid['PM2.5'] * scale)
list_file = list(df_valid['file_Id'])
df_pred = pd.DataFrame({'file_Id': list_file, 'PM2.5': list_gt, 'pred': list_pred})
mae = np.abs(df_pred['pred'] - df_pred['PM2.5'])
df_pred['mae'] = mae
bestMAE = np.round(mae.mean(), 3)
print('MAE:', bestMAE)
print('r2_score:', r2_score(df_pred['pred'], df_pred['PM2.5']))
df_pred.to_csv(f'PM2.5_Pred_MAE:{bestMAE}_{str(grid.best_params_)}.csv', sep=',', index=False)


