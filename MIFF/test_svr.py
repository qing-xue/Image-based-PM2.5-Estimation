import os
from PIL import Image
from joblib import load
import numpy as np
import pandas as pd

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from svr_prepare import f1f2f3


def process_input(im):
    """Feature extracting..."""
    (width, height) = (im.width // 2, im.height // 2)
    im = im.resize((width, height))  # PIL 先 Resize 减少计算量

    rate, depth_sim, residual_sim = f1f2f3(im)
    # 注意特征顺序：features = ['depth_sim', 'residual', 'rate']
    valid_features = np.asarray([[depth_sim, residual_sim, rate]])
    valid_features = (valid_features - SERIES_MIN) / (SERIES_MAX - SERIES_MIN)

    return valid_features


# Just for Beijing dataset ---------------
PM_MIN, PM_MAX = 1.0, 262.0
SCALE = PM_MAX - PM_MIN
SERIES_MAX = np.asarray([0.993714, 0.432661, 1.019228])  # 顺序：depth_sim, residual, rate
SERIES_MIN = np.asarray([0.859622, 0.039976, 0.967000])  # 顺序：depth_sim, residual, rate

# Image 目录-----------------------------------
test_imgs = r'D:\workplace\dataset\Various-Pic\V-picture-3\imgs'
img_dir = test_imgs + '/'
output_dir = img_dir
print("pred_dir:", output_dir)

# Test --------------------------------------------------
bst = load('miff.joblib')  # TODO：模型
all_pred = np.empty((0, 1), float)
img_names = []

for img_name in os.listdir(img_dir):
    if not img_name.endswith(('jpg', 'png', 'jpeg')):
        continue
    print(f'\r {img_name}', end='\n', flush=True)
    img_names += [img_name]

    im = Image.open(img_dir + img_name).convert('RGB')
    valid_features = process_input(im)

    pred_nor = bst.predict(valid_features) * SCALE + PM_MIN
    print('Estimate: ', pred_nor)

    all_pred = np.vstack((all_pred, pred_nor))

df = pd.DataFrame({'IMG': img_names, 'Preds': all_pred.flatten()})
df.to_excel(os.path.join(output_dir, 'results.xlsx'), index=False)