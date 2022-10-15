import os
import pandas as pd


def get_BLH_Dataset(data_root=""):
    anno_name = 'anno'
    img_prefix = os.path.join(data_root, 'images_rotate')
    anno_dir = os.path.join(data_root, anno_name)

    df_train = pd.read_csv(os.path.join(anno_dir, 'df_day_train.csv'))
    df_test = pd.read_csv(os.path.join(anno_dir, 'df_day_valid.csv'))

    df_train_Xys = df_train[['file_Id', 'PM2.5']]
    df_test_Xys = df_test[['file_Id', 'PM2.5']]
    print('Original(may have bad images) Train len: %d, Test len: %d' % (len(df_train_Xys), len(df_test_Xys)))

    df_train_Xys['file_Id'] = df_train_Xys.file_Id.map(lambda x: os.path.join(img_prefix, x))
    df_test_Xys['file_Id'] = df_test_Xys.file_Id.map(lambda x: os.path.join(img_prefix, x))
    return df_train_Xys, df_test_Xys


def get_Heshan_Dataset(data_root=""):
    img_prefix = os.path.join(data_root, 'all_images')
    df_train = pd.read_csv(os.path.join(data_root, 'Heshan_Daytime_train.csv'))
    df_test = pd.read_csv(os.path.join(data_root, 'Heshan_Daytime_valid.csv'))

    # 如有必要需同一不同数据集的列名
    df_train_Xys = df_train[['IMG_ID', 'PM2.5']]
    df_test_Xys = df_test[['IMG_ID', 'PM2.5']]
    print('Original(may have bad images) Train len: %d, Test len: %d' % (len(df_train_Xys), len(df_test_Xys)))

    df_train_Xys['IMG_ID'] = df_train_Xys.IMG_ID.map(lambda x: os.path.join(img_prefix, x))
    df_test_Xys['IMG_ID'] = df_test_Xys.IMG_ID.map(lambda x: os.path.join(img_prefix, x))
    return df_train_Xys, df_test_Xys


def get_PM_Dataset(data_root=""):
    if "BLH" in data_root:
        return get_BLH_Dataset(data_root), "BLH"
    elif "Heshan" in data_root:
        return get_Heshan_Dataset(data_root), "Heshan"
    else:
        raise Exception("No such dataset!")