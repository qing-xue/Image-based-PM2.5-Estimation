import os
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')


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
df.drop(['Time'], axis=1, inplace=True)
# df.drop(['file_Id'], axis=1, inplace=True) # 如果是heshan 请保留这一行

X = df[x_cols]
y = df[y_cols]
print(X.head())
print(X.min().values)
print(X.max().values)

# TODO：only for BLH Beijing
BLH_BJ_min = X.min().values
BLH_BJ_max = X.max().values

# KE_850, GE_500, R_B_Sky, R_B_Ground, RGB_Sky, RGB_Ground, T2m, BLH, U10, V10, TP, SP

# 列特征归一化
Xn = X.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
print(Xn.head())
print(Xn.min())
print(Xn.max())