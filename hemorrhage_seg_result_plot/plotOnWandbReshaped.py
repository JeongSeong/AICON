'''
conda activate deepbleed

cd /ssd_data1/syjeong/deepbleed

python predict.py --indir /ssd_data1/syjeong/hematoma/data/stripped/internal/pre --outdir segMask/internal/pre --weights weights --gpus 1 --brain 


python predict.py --indir /ssd_data1/syjeong/hematoma/data/stripped/internal/post --outdir segMask/internal/post --weights weights --gpus 1 --brain 


python predict.py --indir /ssd_data1/syjeong/hematoma/data/stripped/external/post --outdir segMask/external/post --weights weights --gpus 1 --brain 


python predict.py --indir /ssd_data1/syjeong/hematoma/data/stripped/external/pre --outdir segMask/external/pre --weights weights --gpus 1 --brain 


conda deactivate

conda activate syj_total
'''


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

            slice_num = len(img)
            batch = int(slice_num/window)
            remainder = slice_num % window
            if remainder != 0:
                batch = batch+1
                pad = np.zeros_like(img[0])
                for _ in range(window - remainder): img.append(pad)
            tmp=[]
            for b in range(batch):
                tmp.append(np.concatenate(img[b*window:(b+1)*window], axis=-1)) # concat images along W axis
                # if b == 0:
                #     tmp.append(np.concatenate(img[:window], axis=-1))
                # elif b == (batch-1):
                #     tmp.append(np.concatenate(img[b*window:], axis=-1))
                # else:
                #     tmp.append(np.concatenate(img[b*window:(b+1)*window], axis=-1))
            tmp = np.concatenate(tmp, axis=-2) # concat images along H axis
            img = np.transpose(tmp, (1, 2, 0)).astype(np.uint8) # (H W 3)

            seg = os.path.join(seg_dir, f'{data.split(".")[0]}_prediction.nii.gz')
            volume = subprocess.run(['fslstats', f'{seg}', '-V'], capture_output=True).stdout
            volume = float(volume.decode().split(' ')[1])/1000
            df = pd.concat([df, pd.DataFrame({"pid":f'{data.split(".")[0]}', "volume(ml)":volume}, index=[0])], ignore_index=True)
            seg = nib.load(seg)
            seg = seg.get_fdata() # H W N
            seg = np.transpose(seg, (2, 0, 1)) # (N H W)
            seg = list(seg) # (H W) list

            if remainder != 0:
                pad = np.zeros_like(seg[0])
                for _ in range(window - remainder): seg.append(pad)
            tmp=[]
            for b in range(batch):
                tmp.append(np.concatenate(seg[b*window:(b+1)*window], axis=-1)) # concat images along W axis
            tmp = np.concatenate(tmp, axis=-2)
            tmp[tmp > 0.5] = 1
            seg = tmp

            wandb.log({f'deepbleed/{sub}/{data}_{img_shape}': wandb.Image(img, masks={f'predictions_vol_{volume}': {"mask_data": seg, "class_labels": {1:"hemorrhage"}}})})

        name = sub.split("/")
        df.to_csv(f'/ssd_data1/syjeong/deepbleed/hemorrhage_{name[0]}_{name[1]}.csv')

