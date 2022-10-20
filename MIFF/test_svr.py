import os
from PIL import Image
from joblib import load
import numpy as np

from config import PM_PATH
from svr_prepare import f1f2f3


# Just for Beijing dataset ---------------
PM_MIN, PM_MAX = 1.0, 262.0
SCALE = PM_MAX - PM_MIN
SERIES_MAX = np.asarray([0.993714, 0.432661, 1.019228])  # 顺序：depth_sim, residual, rate
SERIES_MIN = np.asarray([0.859622, 0.039976, 0.967000])  # 顺序：depth_sim, residual, rate

# Image -----------------------------------
img_name = os.path.join(PM_PATH, r'imgs/Beijing_20190530051212642_PM=22.jpg')
im = Image.open(img_name).convert('RGB')
(width, height) = (im.width // 2, im.height // 2)
im = im.resize((width, height))  # PIL 先 Resize 减少计算量

print('Feature extracting...')
rate, depth_sim, residual_sim = f1f2f3(im)
# 注意特征顺序：features = ['depth_sim', 'residual', 'rate']
valid_features = np.asarray([[depth_sim, residual_sim, rate]])
valid_features = (valid_features - SERIES_MIN) / (SERIES_MAX - SERIES_MIN)

# Test --------------------------------------------------
bst = load('miff.joblib')
preds = bst.predict(valid_features) * SCALE + PM_MIN
print('Estimate: ', preds)