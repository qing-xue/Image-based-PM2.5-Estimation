# %% [markdown]
# mlp for PM2.5 Estimation

# %%
import os
import numpy as np
import pandas as pd

import _utils as utils

import warnings

warnings.filterwarnings('ignore')

# %%
np.set_printoptions(formatter={'float': '{: 0.1f}'.format})

def get_filename(filename):
    (filepath, tempfilename) = os.path.split(filename);
    (shotname, extension) = os.path.splitext(tempfilename);
    # return filepath, shotname, extension
    return shotname


def excel_to_df(xlsx):
    df = pd.DataFrame(pd.read_excel(xlsx, sheet_name=get_filename(xlsx)))
    return df


# %%
file_PhotoRGB_AQI_Met = 'PhotoRGB_AQI_Met.xlsx'
# file_PhotoRGB_AQI_Met = 'Photo_Heshan.xlsx'

PhotoRGB_AQI_Met = excel_to_df(file_PhotoRGB_AQI_Met)

# %%
Va_Met_Ground = ['T2m', 'BLH', 'U10', 'V10', 'TP', 'SP']

# %%
dfs = [PhotoRGB_AQI_Met]

for df in dfs:
    T_500 = df['T_500']
    T_850 = df['T_850']
    Ro_500 = T_500.map(lambda x: 50000 * 29 / (8314 * (x + 273.15)))
    Ro_850 = T_850.map(lambda x: 85000 * 29 / (8314 * (x + 273.15)))

    U_500 = df['U_500']
    V_500 = df['V_500']
    U_850 = df['U_850']
    V_850 = df['V_850']

    KE_500 = 0.5 * Ro_500 * (U_500 ** 2 + V_500 ** 2)
    KE_850 = 0.5 * Ro_850 * (U_850 ** 2 + V_850 ** 2)

    GP_500 = df['GP_500']
    GP_850 = df['GP_850']
    GE_500 = Ro_500 * GP_500
    GE_850 = Ro_850 * GP_850

    df['GE_500'] = GE_500
    df['KE_500'] = KE_500
    df['GE_850'] = GE_850
    df['KE_850'] = KE_850


# %%
def getdata_day(df):
    va = df.columns.values.tolist()
    day = df[va][df['R_R_M'] > 100]
    return day


def getdata_night(df):
    va = df.columns.values.tolist()
    night = df[va][df['R_R_M'] < 100]
    return night


# %%
R_G_Sky = dfs[0]['R_R_M'] / dfs[0]['G_R_M']
R_B_Sky = dfs[0]['R_R_M'] / dfs[0]['B_R_M']
RGB_Sky = dfs[0]['R_R_M'] + dfs[0]['G_R_M'] + dfs[0]['B_R_M']

R_G_Ground = dfs[0]['R_L_M'] / dfs[0]['G_L_M']
R_B_Ground = dfs[0]['R_L_M'] / dfs[0]['B_L_M']
RGB_Ground = dfs[0]['R_L_M'] + dfs[0]['G_L_M'] + dfs[0]['B_L_M']

dfs[0]['R_G_Sky'] = R_G_Sky
dfs[0]['R_B_Sky'] = R_B_Sky
dfs[0]['RGB_Sky'] = RGB_Sky
dfs[0]['R_G_Ground'] = R_G_Ground
dfs[0]['R_B_Ground'] = R_B_Ground
dfs[0]['RGB_Ground'] = RGB_Ground

PM25 = dfs[0]['PM2.5']

GE_500 = dfs[0]['GE_500']
BLH = dfs[0]['BLH']
# file_Id = dfs[0]['file_Id']

dfs1_day = getdata_day(dfs[0])
dfs1_night = getdata_night(dfs[0])
print(dfs1_day.shape, dfs1_night.shape)

# %%
df = dfs1_day
# df.drop(['Time'], axis=1, inplace=True)
# dfn=df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

y_cols = ['PM2.5']

x_cols = [
             'KE_850',
             'GE_500',
             'R_B_Sky',
             'R_B_Ground',
             'RGB_Sky',
             'RGB_Ground',
             # 'file_Id'
         ] + Va_Met_Ground

df = df.dropna()

X = df[x_cols]
y = df[y_cols]

# X_train,y_train=X[0:239],y[0:239]
# X_test, y_test =X[239:],y[239:]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
# print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# %% [markdown]
# ### 2. Multi-Layer Perceptron (MLP)

# %%
df.drop(['Time'], axis=1, inplace=True)
# df.drop(['file_Id'], axis=1, inplace=True) # 如果是heshan 请保留这一行
dfn = df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
# df['file_Id'] = file_Id

# %%
# 以下只需修改训练集和测试集
################################################################
X = dfn[x_cols]
y = dfn[y_cols]

X_train, y_train = X[0:239], y[0:239]
X_test, y_test = X[239:], y[239:]
train_df = pd.concat([X_train, y_train], axis=1)
val_df = pd.concat([X_test, y_test], axis=1)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

Y_range = df['PM2.5'].max() - df['PM2.5'].min()
print(Y_range)

# %%

import os.path
import shutil
import time

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from box import Box
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import classifier

torch.autograd.set_detect_anomaly(True)

# %%
# config

config = {
    'root_checkpoint': r'D:\vscoding\mlp',
    'epoch': 1000,
    'train_loader': {
        'batch_size': 500,
        'shuffle': True,
        'num_workers': 0,
        'pin_memory': True,
        'drop_last': False,
    },
    'val_loader': {
        'batch_size': 256,
        'shuffle': False,
        'num_workers': 0,
        'pin_memory': True,
        'drop_last': False,
    },
    'model': {
        'name': 'multilayer_perceptron'
    },
    'optimizer': {
        'name': 'optim.AdamW',
        'params': {
            'lr': 1e-3,
            # 'weight_decay': 3e-4
        },
    },
    'scheduler': {
        'name': 'optim.lr_scheduler.CosineAnnealingWarmRestarts',
        'params': {
            'T_0': 100,
            'eta_min': 1e-2,
        },
    },
    'loss': 'nn.MSELoss',
}

config = Box(config)
print(config)


class myDataset(Dataset):
    def __init__(self, df):
        self._X = df[x_cols].values
        self._X = np.float32(self._X)
        self._y = None
        if "PM2.5" in df.keys():
            self._y = df["PM2.5"].values
            self._y = np.float32(self._y)

    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):
        features = self._X[idx]
        label = self._y[idx]
        return label, features


class DataModule():
    def __init__(self, train_df, val_df, cfg):
        super().__init__()
        self._train_df = train_df
        self._val_df = val_df
        self._cfg = cfg

    def __create_dataset(self, train=True):
        return (
            myDataset(self._train_df) if train
            else myDataset(self._val_df)
        )

    def train_dataloader(self):
        dataset = self.__create_dataset(True)
        return DataLoader(dataset, **self._cfg.train_loader)

    def val_dataloader(self):
        dataset = self.__create_dataset(False)
        return DataLoader(dataset, **self._cfg.val_loader)


# %%
class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# %%

def save_checkpoint(state, is_best, path='checkpoint', filename='last.pth'):
    if not os.path.exists(path):
        os.makedirs(path)
    full_path = os.path.join(path, filename)
    torch.save(state, full_path)
    if is_best:
        shutil.copyfile(full_path, os.path.join(path, 'best.pth'))
        print("Save best model at %s==" %
              os.path.join(path, 'best.pth'))


# %%

# %%

class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.__build_model()
        self._criterion = eval(self.cfg.loss)()
        self.optimizer = None
        self.scheduler = None

    def __build_model(self):
        self.model = classifier(12, 1)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        self.optimizer = eval(self.cfg.optimizer.name)(
            self.parameters(), **self.cfg.optimizer.params
        )
        self.scheduler = eval(self.cfg.scheduler.name)(
            self.optimizer, **self.cfg.scheduler.params
        )


# %%
datamodule = DataModule(train_df, val_df, config)

iter_count_t = 0
iter_count_v = 0
writer = SummaryWriter(comment='Perceptron', filename_suffix='')

model = Model(config)
model = model.cuda()
model.configure_optimizers()
# model.load_state_dict(torch.load(os.path.join(config.root_checkpoint, config.model.name, 'best.pth'))['state_dict'])
# model.optimizer.load_state_dict(torch.load(os.path.join(config.root_checkpoint, config.model.name, 'best.pth'))['optimizer'])

train_loader = datamodule.train_dataloader()
valid_loader = datamodule.val_dataloader()

bestMSE = 100.
bestMSE = torch.load(os.path.join(config.root_checkpoint, config.model.name, 'best.pth'))['best_MSE']
PM_Range = 100.
print("bestMSE: ", bestMSE)

for epoch in range(0, config.epoch):
    data_time = AverageMeter()
    MSELoss = AverageMeter()

    model.train()
    end = time.time()
    preds_list = []
    labels_list = []
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (labels, features) in pbar:
        iter_count_t = iter_count_t + 1

        data_time.update(time.time() - end)
        labels, features = labels.cuda(), features.cuda()
        labels = labels.float()

        logits = model(features)
        logits = logits.squeeze(1)
        loss = model._criterion(logits, labels)

        # _pred = logits.sigmoid().detach().cpu() * 262.
        # _pred = logits.detach().cpu() * Y_range + 11.0769 # heshan最小11.0769
        # _labels = labels.detach().cpu() * Y_range + 11.0769
        _pred = logits.detach().cpu() * Y_range + 1. # beijing 最小是1
        _labels = labels.detach().cpu() * Y_range + 1.

        outputs = {
            'loss': loss,
            'pred': _pred,
            'labels': _labels
        }

        pred, label = outputs['pred'], outputs['labels']
        preds_list.append(pred), labels_list.append(label)

        # metrics = torch.abs(preds - labels).mean()

        MSELoss.update(loss.item(), features.size(0))
        # MSELoss.update(metrics.item(), features.size(0))

        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()

        mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9)
        s = ('%10.4s' * 2 + '%10.6g' * 2) % ('%g/%g' % (epoch, config.epoch - 1), mem, MSELoss.avg, 1.)
        pbar.set_description(s)

    writer.add_scalars("MAELoss", {"Train": MSELoss.avg}, epoch)

    data_time.reset()
    MSELoss.reset()
    model.eval()
    with torch.no_grad():
        preds_list = []
        labels_list = []
        pbar = tqdm(enumerate(valid_loader), total=len(valid_loader))
        for i, (labels, features) in pbar:
            iter_count_v = iter_count_v + 1

            data_time.update(time.time() - end)
            labels, features = labels.cuda(), features.cuda()
            labels = labels.float()

            logits = model(features)
            logits = logits.squeeze(1)
            loss = model._criterion(logits, labels)

            # _pred = logits.sigmoid().detach().cpu() * 262.
            # _pred = logits.detach().cpu() * Y_range + 11.0769
            # _labels = labels.detach().cpu() * Y_range + 11.0769
            _pred = logits.detach().cpu() * Y_range + 1.
            _labels = labels.detach().cpu() * Y_range + 1.

            outputs = {
                'loss': loss,
                'preds': _pred,
                'labels': _labels
            }

            pred, label = outputs['preds'], outputs['labels']
            preds_list.append(pred), labels_list.append(label)
            # metrics = torch.abs(preds - labels).mean()

            MSELoss.update(loss.item(), features.size(0))
            # MSELoss.update(metrics.item(), features.size(0))

            mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9)
            s = ('%10.4s' * 2 + '%10.6g' * 2) % ('%g/%g' % (epoch, config.epoch - 1), mem, MSELoss.avg, 1.)
            pbar.set_description(s)
        labels_list = torch.cat(labels_list)
        preds_list = torch.cat(preds_list)
        rmse, mae, r2_score, nmge = utils.get_metrics(labels_list, preds_list)
        writer.add_scalars("MAELoss", {"Valid": MSELoss.avg}, epoch)
        writer.add_scalars("MAE", {"Valid": mae}, epoch)
        writer.add_scalars("RMSE", {"Valid": rmse}, epoch)
        writer.add_scalars("R2_Score", {"Valid": r2_score}, epoch)
        writer.add_scalars("NMGE", {"Valid": nmge}, epoch)
        is_best = MSELoss.avg < bestMSE
        bestMSE = min(bestMSE, MSELoss.avg)

        dataframe = pd.DataFrame({'label': labels_list, 'pred': preds_list})
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_MSE': MSELoss.avg,
            'optimizer': model.optimizer.state_dict()
        }, is_best, os.path.join(config.root_checkpoint, config.model.name))
