import wandb
# import argparse
import os
from tqdm import tqdm
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
# parser = argparse.ArgumentParser()
# parser.add_argument('--data_dir', type=str)
# parser.add_argument('--project_name', type=str)
# parser.add_argument('--n_clustering', type=int)
# args = parser.parse_args()
# palette  = {key:value for key, value in enumerate(plt.get_cmap("tab10").colors)}
# bright_factor = {'cbv':5, 'ktrans':100, 'adc':2, 't1ce':2} 
# alpha = 0.5 # for masking
# modal_list = ['ktrans_to_t1ce_orientedFlirt.nii.gz', 'cbv_to_t1ce_orientedFlirt.nii.gz', 'adc_to_t1ce_orientedFlirt.nii.gz', 't1ce.nii.gz']

def draw_mapped_mri(
        data_dir,
        project_name,
        total, # cupy with mask index column name is 'ne_ind', clustering class columns name is f'{n_clustering}_cuml_kmeans'
        modal_list = ['ktrans_to_t1ce_orientedFlirt.nii.gz', 'cbv_to_t1ce_orientedFlirt.nii.gz', 'adc_to_t1ce_orientedFlirt.nii.gz', 't1ce.nii.gz'],
        bright_factor = {'cbv':5, 'ktrans':100, 'adc':2, 't1ce':2},
        alpha = 0.5,
        palette  = {key:value for key, value in enumerate(plt.get_cmap("tab10").colors)},
        n_clustering = 6
):
    wandb.init(project=project_name) # args.project_name
    for subj in tqdm(sorted(os.listdir(data_dir))): # args.data_dir
        
        directory = os.path.join(data_dir, subj) # args.data_dir
        if not os.path.isdir(directory): continue
        os.chdir(directory)
        # print(os.getcwd())
        df = total[total['subj']==subj] 
        classes = df[f'{n_clustering}_cuml_kmeans'].unique() # args.n_clustering
        total_modal = {}
        shape = None
        for modal in modal_list:
            array = nib.load(modal)
            array = array.get_fdata() # H W N
            array = np.clip(array, a_min=0, a_max=None)
            array = (array-np.min(array))/(np.max(array)-np.min(array))
            if shape is None: shape = array.shape
            array = np.repeat(array[...,np.newaxis], 3, -1)  # H W N 3
            tag = modal.split('_')[0].split('.')[0]
            array = np.clip(array*bright_factor[tag], 0, 1)
            total_modal[tag] = array
        mask = None
        # for cla in classes: # pandas df 쓸 경우
        for cla in classes.values.get().tolist(): # cupy 쓸 경우 
            mask_df = df[df[f'{n_clustering}_cuml_kmeans']==cla] # args.n_clustering
            array = np.zeros(shape)
            array = array.flatten()
            # array[mask_df.ne_ind] = 1 # pandas df 쓸 경우
            array[mask_df.ne_ind.values.get()] = 1 # cupy 쓸 경우 
            array = array.reshape(shape)
            array = np.repeat(array[...,np.newaxis], 3, -1)  # H W N 3
            array = array*palette[cla]
            if mask is None:
                mask = array
            else:
                mask = mask + array

        del mask_df

        tag=''
        for key, array in total_modal.items():
            masked = mask*alpha + array*(1-alpha)
            masked = np.transpose(masked, (2, 3, 0, 1)) # (N 3 H W) 
            total_modal[key] = masked
            tag = tag+key+'_'

        total_modal = np.concatenate(list(total_modal.values()), axis=2)# (N, 3, H, W)
        zero_slices = np.where(np.all(total_modal == 0, axis=(1, 2, 3)))[0] 
        total_modal = np.delete(total_modal, zero_slices, axis=0) 
        show = wandb.Image(np.transpose(np.concatenate(list(total_modal), axis=-1), (1, 2, 0)), caption=f'{subj}/images')

        wandb.log({f'{subj}/{tag}': wandb.Video(total_modal*255, fps=50), f'{subj}/images': show})

    wandb.finish()