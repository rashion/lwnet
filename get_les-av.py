import os
import os.path as osp
import shutil
import pandas as pd
from tqdm import tqdm
from PIL import Image
from skimage import io
import numpy as np
from torchvision.transforms.functional import resize

########################################################################################################################
# What you can see below are the old instructions to download the LES-AV dataset. Since the release of this codebase,
# LES-AV has been removed from the public url we used to employ for downloading it and hosted at figshare, which allows
# free downloading, but as far as I know it needs to be done manually. Please head to the following url:
#
# https://figshare.com/articles/dataset/LES-AV_dataset/11857698
#
########################################################################################################################

# download the LES-AV.zip file and then reproduce the steps below accordingly, sorry for the inconvenience. Adrian.

call = '(unzip LES-AV.zip -d data/LES-AV ' \
       '&& rm LES-AV.zip && mv data/LES-AV data/LES_AV' \
       '&& rm -r data/LES_AV/__MACOSX)'
os.system(call)

call = '(mkdir data/LES-AV ' \
       '&& mv data/LES_AV/LES-AV/images data/LES-AV/images ' \
       '&& mv data/LES_AV/LES-AV/masks data/LES-AV/mask ' \
       '&& mv data/LES_AV/LES-AV/vessel-segmentations data/LES-AV/manual' \
       '&& mv data/LES_AV/LES-AV/arteries-and-veins data/LES-AV/manual_av ' \
       '&& rm -r data/LES_AV)'
os.system(call)

path_ims = 'data/LES-AV/images'
path_masks = 'data/LES-AV/mask'
path_gts = 'data/LES-AV/manual'

all_im_names = sorted(os.listdir(path_ims))
all_mask_names = sorted(os.listdir(path_masks))
all_gt_names = sorted(os.listdir(path_gts))

all_im_names = [osp.join(path_ims, n) for n in all_im_names]
all_mask_names = [osp.join(path_masks, n) for n in all_mask_names]
all_gt_names = [osp.join(path_gts, n) for n in all_gt_names]

df_lesav_all = pd.DataFrame({'im_paths': all_im_names,
                             'gt_paths': all_gt_names,
                             'mask_paths': all_mask_names})
df_lesav_all.to_csv('data/LES-AV/test_all.csv', index=False)

# create data/LES_AV/test_av.csv:
df_lesav_all.gt_paths = [n.replace('manual', 'manual_av') for n in df_lesav_all.gt_paths]
df_lesav_all.to_csv('data/LES-AV/test_all_av.csv', index=None)
print('LES-AV prepared')

