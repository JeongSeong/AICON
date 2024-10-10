from monai import transforms as mt
import numpy as np
from typing import Tuple
import torch
import torch.nn.functional as F
import random

def create_transforms(
    img_size: Tuple[int, int] = (224, 224),
    # spacing: Tuple[float, float, float] = (0.5, 0.5, 5.0),
    masks: bool = True,
    mode: str = "train",
) -> mt.Compose:
    """
    Create transformations for training or validation.

    Args:
        img_size (Tuple[int, int, int]): Target image size.
        masks (bool): Whether masks are included.
        mode (str): Mode for transformations, either 'train' or 'valid'.

    Returns:
        Compose: Composed transformations.
    """
    if img_size is not None:
        assert mode in ["train", "valid"], "mode should be either 'train' or 'valid'"

        kith = ["image", "mask"] if masks else ["image"]
        binear = ("bilinear", "nearest") if masks else "bilinear"

        base_transforms = [
            mt.LoadImaged(keys=kith),
            # Spacingd(keys=kith, pixdim=spacing, mode=binear),
            mt.EnsureChannelFirstd(keys=kith),
            mt.Orientationd(keys=kith, axcodes="RAS"),
            # ResizeWithPadOrCropd(keys=kith, spatial_size=img_size),
            # SelectMaxMaskSlice(keys=kith),
        ]

        if mode == "train":
            augmentations = [
                mt.RandAffined(
                    keys=kith,
                    prob=0.80,
                    rotate_range=((-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1)),
                    scale_range=((-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1)),
                    # translate_range=((-10, 10), (-10, 10), (-0.1, 0.1)),
                    shear_range=((-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1)),
                    mode=binear,
                    padding_mode="zeros",
                ),
                # mt.RandStdShiftIntensityd(keys=["image"], factors=(0, 2), prob=0.50, nonzero=True), # maybe change factors to (0, 10)  
                mt.RandGaussianNoised(keys=["image"], prob=0.40, mean=0, std=0.2),
                # RandCoarseShuffled(
                #     keys=["image"], spatial_size=(3, 3, 3), prob=0.30, holes=3
                # ),
            ]
            base_transforms.extend(augmentations)

        # Add ScaleIntensityRanged and ToTensord at the end
        base_transforms.extend(
            [
                mt.ScaleIntensityRanged(
                    keys=["image"], a_min=0, a_max=100, b_min=0, b_max=1, clip=True
                ),
                mt.ToTensord(keys=kith),
            ]
        )

        
        return mt.Compose(base_transforms)
    else: # if img_size is None:
        return mt.Compose([])

def maxSeg_repeat3(batch, img_size=224):
    clinics_array = np.stack([np.expand_dims(item['clinic'], 0) for item in batch])

    # 이미지 데이터들을 PyTorch 텐서로 결합
    images_tensor = []
    for item in batch:
        mask = item['mask'] # (batch=1, H, W, D)
        slice_sums = torch.sum(mask, dim=(1, 2))
        max_slice_idx = torch.argmax(slice_sums[0]).item()
        img = item['image'][:, :, :, max_slice_idx] # (batch=1, H, W)
        img = img.unsqueeze(1).repeat(1, 3, 1, 1)
        img = F.interpolate(img, size=img_size) # input size
        img = img.squeeze(0)
        images_tensor.append(img)
    images_tensor = torch.stack(images_tensor)
    # images_tensor = torch.stack([item['image'] for item in batch])

    # 레이블 데이터들을 numpy 배열로 결합
    labels_array = np.array([item['label'] for item in batch])
    binary = 1-labels_array
    # label_names=['HematomaExpansion', 'notHE'] # labels_array[0] is the true expansion label
    labels_array = np.stack((labels_array, binary), axis=1) 

    # 배치 크기만큼의 1로 구성된 리스트
    ones_list = [1] * len(batch)

    # 배치 크기만큼의 True로 구성된 리스트
    Trues_list = [True] * len(batch)

    return [clinics_array, images_tensor, labels_array, ones_list, Trues_list]

def maxSeg_around3(batch, img_size, test=False):
    # clinics_array = np.stack([np.expand_dims(item['clinic'], 0) for item in batch])# when ehr is time series and suitable for transformer
    clinics_array = np.stack([item['clinic'] for item in batch])

    if img_size is None:
        images_tensor = None
    else:
        # 이미지 데이터들을 PyTorch 텐서로 결합
        images_tensor = []
        for item in batch:
            mask = item['mask'] # (batch=1, H, W, D)
            slice_sums = torch.sum(mask, dim=(1, 2))
            max_slice_idx = torch.argmax(slice_sums[0]).item()
            if max_slice_idx != 0:
                if max_slice_idx != (mask.shape[3]-1): # only this condition would be used, but I made the others just in case. 
                    img = item['image'][:, :, :, (max_slice_idx-1):(max_slice_idx+2)]
                else: # elif max_slice_idx == (mask.shape[3]-1):
                    img = item['image'][:, :, :, max_slice_idx].unsqueeze(-1)
                    img = torch.cat([item['image'][:, :, :, (max_slice_idx-1):(max_slice_idx+1)], img], dim=3)
            else: # elif max_slice_idx == 0:
                if max_slice_idx != (mask.shape[3]-1):
                    img = item['image'][:, :, :, max_slice_idx].unsqueeze(-1)
                    img = torch.cat([img, item['image'][:, :, :, max_slice_idx:(max_slice_idx+2)]], dim=3)
                else: # the case when the first slice is the last slice. If then, the data should be checked
                    img = item['image'][:, :, :, max_slice_idx].unsqueeze(-1)
                    img = torch.cat([img, img, img], dim=3)
            img = img.permute(0, 3, 1, 2)
            img = F.interpolate(img, size=img_size) # input size
            img = img.squeeze(0)
            images_tensor.append(img)
        images_tensor = torch.stack(images_tensor)

    # 레이블 데이터들을 numpy 배열로 결합
    labels_array = np.array([item['label'] for item in batch])
    binary = 1-labels_array
    # label_names=['HematomaExpansion', 'notHE'] # labels_array[0] is the true expansion label
    labels_array = np.stack((labels_array, binary), axis=1) 

    # 배치 크기만큼의 1로 구성된 리스트
    ones_list = [1] * len(batch)

    # 배치 크기만큼의 True로 구성된 리스트
    Trues_list = [True] * len(batch)
    if test:
        pid_array = np.array([item['id'] for item in batch])
        return [pid_array, clinics_array, images_tensor, labels_array, ones_list, Trues_list]

    return [clinics_array, images_tensor, labels_array, ones_list, Trues_list]

def FTT(batch, cat_ind=[], cont_ind=[]):
    # item['clinic'] is a 1 dimension numpy array
    cat_array = np.stack([item['clinic'][cat_ind][0] for item in batch])
    cont_array = np.stack([item['clinic'][cont_ind][0] for item in batch])
    # 레이블 데이터들을 numpy 배열로 결합
    labels_array = np.array([item['label'] for item in batch])
    binary = 1-labels_array
    # label_names=['HematomaExpansion', 'notHE'] # labels_array[0] is the true expansion label
    labels_array = np.stack((labels_array, binary), axis=1) 
    pid_array = np.array([item['id'] for item in batch])
    return [pid_array, cont_array, cat_array, labels_array]


def img_mask(batch, img_size):
    clinics_array = np.stack([item['clinic'] for item in batch])
    # 이미지 데이터들을 PyTorch 텐서로 결합
    images_tensor = []
    masks_tensor = []
    for item in batch:
        mask = item['mask'] # (batch=1, H, W, D)
        slice_sums = torch.sum(mask, dim=(1, 2))
        max_slice_idx = torch.argmax(slice_sums[0]).item()
        # if max_slice_idx != 0:
        #     if max_slice_idx != (mask.shape[3]-1): # only this condition would be used, but I made the others just in case. 
        #         # img = item['image'][:, :, :, (max_slice_idx-1):(max_slice_idx+2)]
        #         idx = random.choice([max_slice_idx-1, max_slice_idx, max_slice_idx+1])
        #     else: # elif max_slice_idx == (mask.shape[3]-1):
        #         # img = item['image'][:, :, :, max_slice_idx].unsqueeze(-1)
        #         # img = torch.cat([item['image'][:, :, :, (max_slice_idx-1):(max_slice_idx+1)], img], dim=3)
        #         idx = random.choice([max_slice_idx-1, max_slice_idx])
        # else: # elif max_slice_idx == 0:
        #     if max_slice_idx != (mask.shape[3]-1):
        #         # img = item['image'][:, :, :, max_slice_idx].unsqueeze(-1)
        #         # img = torch.cat([img, item['image'][:, :, :, max_slice_idx:(max_slice_idx+2)]], dim=3)
        #         idx = random.choice([max_slice_idx, max_slice_idx+1])
        #     else: # the case when the first slice is the last slice. If then, the data should be checked
        #         # img = item['image'][:, :, :, max_slice_idx].unsqueeze(-1)
        #         # img = torch.cat([img, img, img], dim=3)
        #         idx = max_slice_idx
        idx = random.choice(list(range(max(0, max_slice_idx-1), max((mask.shape[3]-1), max_slice_idx+1)+1)))
        img = item['image'][:, :, :, idx].unsqueeze(-1).repeat(1, 1, 1, 3)
        img = img.permute(0, 3, 1, 2)
        img = F.interpolate(img, size=img_size) # input size
        img = img.squeeze(0)
        images_tensor.append(img)
        mask = mask[:, :, :, idx].unsqueeze(-1).repeat(1, 1, 1, 3)
        mask = mask.permute(0, 3, 1, 2)
        mask = F.interpolate(mask, size=img_size)
        mask = mask.squeeze(0)
        masks_tensor.append(mask)
    images_tensor = torch.stack(images_tensor)
    masks_tensor = torch.stack(masks_tensor)

    # 레이블 데이터들을 numpy 배열로 결합
    labels_array = np.array([item['label'] for item in batch])
    binary = 1-labels_array
    # label_names=['HematomaExpansion', 'notHE'] # labels_array[0] is the true expansion label
    labels_array = np.stack((labels_array, binary), axis=1) 

    pid_array = np.array([item['id'] for item in batch])
    return [pid_array, clinics_array, images_tensor, masks_tensor, labels_array]

def maxSeg_around3_sample(batch, img_size, test=False):
    # clinics_array = np.stack([np.expand_dims(item['clinic'], 0) for item in batch])# when ehr is time series and suitable for transformer
    clinics_array = np.stack([item['clinic'] for item in batch])

    if img_size is None:
        images_tensor = None
    else:
        # 이미지 데이터들을 PyTorch 텐서로 결합
        images_tensor = []
        for item in batch:
            mask = item['mask'] # (batch=1, H, W, D)
            slice_sums = torch.sum(mask, dim=(1, 2))
            max_slice_idx = torch.argmax(slice_sums[0]).item()
            # if max_slice_idx-1 != 0:
            #     if max_slice_idx != (mask.shape[3]-1): # only this condition would be used, but I made the others just in case. 
            #         # img = item['image'][:, :, :, (max_slice_idx-1):(max_slice_idx+2)]
            #         idx = random.choice([max_slice_idx-1, max_slice_idx, max_slice_idx+1])
            #         indices = 
            #     else: # elif max_slice_idx == (mask.shape[3]-1):
            #         # img = item['image'][:, :, :, max_slice_idx].unsqueeze(-1)
            #         # img = torch.cat([item['image'][:, :, :, (max_slice_idx-1):(max_slice_idx+1)], img], dim=3)
            #         idx = random.choice([max_slice_idx-1, max_slice_idx])
            #         indices = 
            # else: # elif max_slice_idx-1 == 0:
            #     if max_slice_idx != (mask.shape[3]-1):
            #         # img = item['image'][:, :, :, max_slice_idx].unsqueeze(-1)
            #         # img = torch.cat([img, item['image'][:, :, :, max_slice_idx:(max_slice_idx+2)]], dim=3)
            #         idx = random.choice([max_slice_idx, max_slice_idx+1])
            #         indices = 
            #     else: # the case when the first slice is the last slice. If then, the data should be checked
            #         # img = item['image'][:, :, :, max_slice_idx].unsqueeze(-1)
            #         # img = torch.cat([img, img, img], dim=3)
            #         idx = max_slice_idx
            #         indices = 
            # 가능한 인덱스 리스트 만들기 (5개?)
            # 리스트 내에서 연속되는 인덱스 세개 뽑기 (맨 처음 세개 랜덤 추출 + 4인덱스까지)
            # 안되는 경우 일단 생각 안하고 일단 코드 짜자
            indices = random.choice(list(range(max(max_slice_idx-1, max(max_slice_idx-2, 0)), max_slice_idx+1)))
            if indices+1 >= item['image'].shape[-1]:
                indices = [indices, indices, indices]
            elif indices+2 >= item['image'].shape[-1]:
                indices = [indices, indices+1, indices+1]
            else:
                indices = list(range(indices, indices+3))
            # if len(set(indices).difference(set(range(item['image'].shape[-1])))) != 0:
            #     print(item['image'].shape[-1], indices, max_slice_idx, (max_slice_idx in indices))
            img = item['image'][:, :, :, indices]#.unsqueeze(-1).repeat(1, 1, 1, 3)
            img = img.permute(0, 3, 1, 2)
            img = F.interpolate(img, size=img_size) # input size
            img = img.squeeze(0)
            images_tensor.append(img)
        images_tensor = torch.stack(images_tensor)

    # 레이블 데이터들을 numpy 배열로 결합
    labels_array = np.array([item['label'] for item in batch])
    binary = 1-labels_array
    # label_names=['HematomaExpansion', 'notHE'] # labels_array[0] is the true expansion label
    labels_array = np.stack((labels_array, binary), axis=1) 

    # 배치 크기만큼의 1로 구성된 리스트
    ones_list = [1] * len(batch)

    # 배치 크기만큼의 True로 구성된 리스트
    Trues_list = [True] * len(batch)
    if test:
        pid_array = np.array([item['id'] for item in batch])
        return [pid_array, clinics_array, images_tensor, labels_array, ones_list, Trues_list]

    return [clinics_array, images_tensor, labels_array, ones_list, Trues_list]