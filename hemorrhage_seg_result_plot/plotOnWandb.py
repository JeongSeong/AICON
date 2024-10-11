#%%
import os
import nibabel as nib
import numpy as np
from tqdm import tqdm
import wandb
import subprocess
import pandas as pd


subdir = ["external/pre", "external/post", "internal/pre", "internal/post"]

image_directory = "/ssd_data1/syjeong/hematoma_copy/data/stripped"
seg_directory = "/ssd_data1/syjeong/deepbleed/segMask"

window = 4

#%%
with wandb.init(project="HE", name="deepbleed", config={"data":"stripped", "mask generation":"deepbleed", "server":"170"}) as run:
    for sub in subdir:
        img_dir = os.path.join(image_directory, sub)
        seg_dir = os.path.join(seg_directory, sub)
        df = pd.DataFrame()
        for data in tqdm(os.listdir(img_dir)):

            img = nib.load(os.path.join(img_dir, data))
            img = img.get_fdata() # H W N
            img_shape = img.shape
            # img = (img-np.min(img))/(np.max(img)-np.min(img))
            # img = img*255
            img = np.repeat(img[..., np.newaxis], 3, -1) # H W N 3
            img = np.transpose(img, (2, 3, 0, 1)) # (N 3 H W)
            img = list(img) # (3 H W) list
            img = np.concatenate(img, axis=-1) # concat all images along W axis
            img = np.transpose(img, (1, 2, 0)).astype(np.uint8) # (H W 3)

            seg = os.path.join(seg_dir, f'{data.split(".")[0]}_prediction.nii.gz')
            volume = subprocess.run(['fslstats', f'{seg}', '-V'], capture_output=True).stdout
            volume = float(volume.decode().split(' ')[1])/1000
            df = pd.concat([df, pd.DataFrame({"pid":f'{data.split(".")[0]}', "volume(ml)":volume}, index=[0])], ignore_index=True)
            seg = nib.load(seg)
            seg = seg.get_fdata() # H W N
            seg = np.transpose(seg, (2, 0, 1)) # (N H W)
            seg = list(seg) # (H W) list
            seg = np.concatenate(seg, axis=-1).astype(np.uint8) # concat all images along W axis

            wandb.log({f'deepbleed/{sub}/{data}_{img_shape}': wandb.Image(img, masks={f'predictions_vol_{volume}': {"mask_data": seg, "class_labels": {1:"hemorrhage"}}})})

        name = sub.split("/")
        df.to_csv(f'/ssd_data1/syjeong/deepbleed/hemorrhage_{name[0]}_{name[1]}.csv')

