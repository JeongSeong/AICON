import os
import yaml
import argparse
from copy import deepcopy
import wandb
import shap
import matplotlib.pyplot as plt

from pathlib import Path
from argparse import Namespace

import torch
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
# from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, WeightedRandomSampler
from monai.data import SmartCacheDataset

from tqdm import tqdm
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# from models import DrFuseTrainer
from models import ehrOnlyTrainer
from collections import OrderedDict
# from utils import EHRDiscretizer, EHRNormalizer, get_ehr_datasets, load_cxr_ehr, load_discretized_header
# import rootutils
# rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
import warnings
# warnings.filterwarnings("ignore")

# from src.transform import create_transforms
# from src.trainer import Trainer
from functools import partial
from utils import (
    snuh_prep, brm_prep, impute_data, scale_data, dict_list, 
    create_transforms, maxSeg_repeat3, maxSeg_around3, # maxSeg_around3_test, collate_like_drfuse,
    save_auc_plot, save_calibration_plot#, shapModelWrapper
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hematoma Expansion")

    # parser.add_argument('--data_dir', type=str, default='./dataset/mimiciv_ehr_cxr')
    # parser.add_argument('--cxr_resized_data_dir', type=str, default='/hdd2/mimic_cxr_resized')
    # parser.add_argument('--data_ratio', type=float, default=1.0)
    # parser.add_argument('--timestep', type=float, default=1.0)
    # parser.add_argument('--normalizer_state', type=str, default=None)

    parser.add_argument('--data_pair', type=str, default='paired', choices=['partial', 'paired'])
    parser.add_argument('--lambda_disentangle_shared', type=float, default=1)
    parser.add_argument('--lambda_disentangle_ehr', type=float, default=1)
    parser.add_argument('--lambda_disentangle_cxr', type=float, default=1)
    parser.add_argument('--lambda_pred_ehr', type=float, default=1)
    parser.add_argument('--lambda_pred_cxr', type=float, default=1)
    parser.add_argument('--lambda_pred_shared', type=float, default=1)
    parser.add_argument('--aug_missing_ratio', type=float, default=0)
    parser.add_argument('--lambda_attn_aux', type=float, default=1)
    parser.add_argument('--ehr_n_layers', type=int, default=1)
    parser.add_argument('--ehr_n_head', type=int, default=4)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=1e-01)
    parser.add_argument('--wd', type=float, default=0, help='weight decay')
    parser.add_argument('--batch_size', type=int, default=97)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=100)
    
    parser.add_argument('--train_img_dir', type=str, default='/ssd_data1/syjeong/hematoma/data/stripped/internal/pre')
    parser.add_argument('--test_img_dir', type=str, default='/ssd_data1/syjeong/hematoma/data/stripped/external/pre')
    parser.add_argument('--train_mask_dir', type=str, default='/ssd_data1/syjeong/hematoma/data/segMask/internal/pre')
    parser.add_argument('--test_mask_dir', type=str, default='/ssd_data1/syjeong/hematoma/data/segMask/external/pre')
    parser.add_argument('--train_ehr', type=str, default='/ssd_data1/syjeong/hematoma/data/EHR/snuh_ehr.csv')
    parser.add_argument('--test_ehr', type=str, default='/ssd_data1/syjeong/hematoma/data/EHR/ehr_brm.csv')
    parser.add_argument('--impute_methods', type=str, default='knn')
    parser.add_argument('--experiment_dir', type=str, default=f'{os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments")}', help='directory path for storing imputer, scaler, weights')
    parser.add_argument('--img_backbone', type=str, default='Attention', choices=['CNN', 'Attention'])
    parser.add_argument('--img_size', type=int, default=None)
    
    args = parser.parse_args()

    args.exp_name = input("Specify the note for this run: ") # must be int maybe... in my case, I put date and time

    save_path = os.path.join(args.experiment_dir, args.exp_name)
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(vars(args), f)

    # set number of threads allowed
    torch.set_num_threads(5)

    # # set seed
    # L.seed_everything(0)

    # # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # in this study, train set was snuh, and test set was brm. please make your own ehr preparation function.
    train_data = snuh_prep(args.train_ehr, args.train_img_dir, args.train_mask_dir)

    # split train data into train and validation set
    train_data, valid_data = train_test_split(train_data, test_size=0.2, stratify=train_data["label"])
    # impute ehr's not available clinical variables, and scale continuous variables
    train_data = train_data.reset_index(drop=True)
    train_data, imputer = impute_data(train_data, mode='train', impute_methods=args.impute_methods, save_path=save_path)
    train_data, scaler = scale_data(train_data, mode='train', save_path=save_path)
    feature = train_data.drop(columns=["id", "image", "label", "mask"])
    shapClinics_train = torch.from_numpy(feature.values).float().requires_grad_()
    train_data = dict_list(train_data, exclude_cols=["id", "image", "label", "mask"])
    print(f"train pos / total = {sum([d['label'] for d in train_data])}/{len(train_data)}")
    
    valid_data = valid_data.reset_index(drop=True)
    valid_data = impute_data(valid_data, mode="test", imputer=imputer) # don't need to specity impute_methods in test mode
    valid_data = scale_data(valid_data, mode="test", scaler=scaler)
    valid_data = dict_list(valid_data)
    print(f"validation pos / total = {sum([d['label'] for d in valid_data])}/{len(valid_data)}")

    # input_size = (224, 224)

    train_dataset = SmartCacheDataset(
        data=train_data,
        # cache_rate=1.0,
        replace_rate=0.2,
        transform=create_transforms(
            img_size=args.img_size, 
            masks=True, mode="train"
        ),
        num_init_workers=args.num_workers,
        shuffle=True,
    )
    valid_dataset = SmartCacheDataset(
        data=valid_data,
        # cache_rate=1.0,
        transform=create_transforms(
            img_size=args.img_size, 
            masks=True, mode="valid"
        ),
        num_init_workers=args.num_workers,
        shuffle=False,
    )

    labels = [sample['label'] for sample in train_data]
    class_counts = torch.bincount(torch.tensor(labels, dtype=torch.long))
    class_weights = 1. / class_counts.float()
    weights = class_weights[torch.tensor(labels, dtype=torch.long)]

    # 3. WeightedRandomSampler 생성하기
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

    train_dataset.start()
    valid_dataset.start()
    train_dl = DataLoader(train_dataset, args.batch_size, shuffle=False, collate_fn=partial(maxSeg_around3, img_size=args.img_size), 
                          pin_memory=True, num_workers=args.num_workers, drop_last=True, sampler=sampler)
    val_dl = DataLoader(valid_dataset, args.batch_size, shuffle=False, collate_fn=partial(maxSeg_around3, img_size=args.img_size), 
                        pin_memory=True, num_workers=args.num_workers, drop_last=False)
    
    model = ehrOnlyTrainer(args=args, label_names=['HematomaExpansion', 'nonHE'], train_dataset=train_dataset, valid_dataset=valid_dataset) # label_names=['HematomaExpansion', 'nonHE']



    early_stop_callback = EarlyStopping(monitor='val_loss/total',
                                        min_delta=0.00,
                                        patience=args.patience,
                                        verbose=False,
                                        mode="min")

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss/total',  # val_loss 기반으로 모델 저장
        dirpath=save_path,
        filename='{epoch}-{val_PRAUC_avg_over_dxs/final:.2f}-{val_loss/total:.2f}', # 'checkpoint',
        auto_insert_metric_name=False,    
        save_top_k=1,  # 최고 성능 모델 1개만 저장
        mode='min'  # val_loss가 최소일 때 저장
    )

    # logger = TensorBoardLogger(save_dir=os.getcwd(), version=args.exp_name, name="lightning_logs")
    logger = WandbLogger(save_dir=os.getcwd(), version=args.exp_name, name=f"{args.exp_name}", project="Hematoma Expansion")
    trainer = L.Trainer(devices=[0],
                        accelerator='gpu',
                        max_epochs=args.epochs,
                        min_epochs=min(args.epochs, 10),
                        callbacks=[early_stop_callback, checkpoint_callback],
                        log_every_n_steps=int(len(train_dataset)/args.batch_size),
                        logger=logger)

    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    train_dataset.shutdown()
    valid_dataset.shutdown()
    # # # comment below if you don't use wandb
    # # run = wandb.init(project="Hematoma Expansion", config=vars(args), notes = args.exp_name)
    # # cfg = run.cfg
    # with wandb.init(project="Hematoma Expansion", config=vars(args), notes = args.exp_name) as run:

    # do testing
    results = {
        'best_val_prauc': trainer.callback_metrics['val_PRAUC_avg_over_dxs/final'].item(),
        'best_val_roauc': trainer.callback_metrics['val_AUROC_avg_over_dxs/final'].item()
    }

    # in this study, train set was snuh, and test set was brm. please make your own ehr preparation function.
    test_data = brm_prep(args.test_ehr, args.test_img_dir, args.test_mask_dir)
    test_data = test_data.reset_index(drop=True)
    test_data = impute_data(test_data, mode="test", save_path=os.path.join(save_path, "imputer.joblib"))
    test_data = scale_data(test_data, mode="test", save_path=os.path.join(save_path, "scaler.joblib"))
    feature = test_data.drop(columns=["id", "image", "label", "mask"])
    shapClinics = torch.from_numpy(feature.values).float().requires_grad_()
    test_data = dict_list(test_data)
    print(f"test pos / total = {sum([d['label'] for d in test_data])}/{len(test_data)}")

    dataset = SmartCacheDataset(
        data=test_data,
        cache_rate=1.0,
        transform=create_transforms(
            img_size=args.img_size, 
            masks=True, mode="valid"
        ),
        num_init_workers=args.num_workers,
        shuffle=True,
    )
    test_dl_paired = DataLoader(dataset, batch_size=5, shuffle=False, num_workers=12, 
                                collate_fn=partial(maxSeg_around3, img_size=args.img_size, test=True)
                                )

    # trainer.loggers = None
    # trainer.test(model=model, dataloaders=test_dl_partial)
    # results['partial_test_results'] = deepcopy(model.test_results)
    model = ehrOnlyTrainer.load_from_checkpoint(checkpoint_callback.best_model_path, args=args, label_names=['HematomaExpansion', 'nonHE'], train_dataset=train_dataset, valid_dataset=valid_dataset)

    trainer.test(model=model, dataloaders=test_dl_paired)
    results['paired_test_results'] = deepcopy(model.test_results)
    print('best_val_prauc: ', results['best_val_prauc'])
    print('best_val_roauc: ', results['best_val_roauc'])
    print('mlaps: ', results['paired_test_results']['mlaps'])
    print('mlroc: ', results['paired_test_results']['mlroc'])
    print('prauc: ', results['paired_test_results']['prauc'])
    print('auroc: ', results['paired_test_results']['auroc'])

    save_auc_plot(model.test_results["y_gt"][:, 0], model.test_results["preds"][:, 0], save_path)
    save_calibration_plot(model.test_results["y_gt"][:, 0], model.test_results["preds"][:, 0], save_path)
    torch.set_grad_enabled(True)
    model.model.train()
    model = model.cpu()  # Move model to CPU if not already done
    explainer = shap.DeepExplainer(model.model, shapClinics_train) # (shapModelWrapper(self.model.ehr_model.cpu()), x)
    # explainer = shap.GradientExplainer(shapModelWrapper(model.model.ehr_model.cpu()), shapClinics) # (shapModelWrapper(self.model.ehr_model.cpu()), x)
    # print(feature.columns, shapClinics_train.shape, shapClinics.shape)
    shap_values = explainer.shap_values(shapClinics)[:, :, 0]
    # print(shap_values.shape)
    plt.figure()  # fig, ax = plt.subplots()
    shap.summary_plot(shap_values, shapClinics.cpu().detach().numpy(), show=False, feature_names=feature.columns)
    plt.savefig(os.path.join(save_path, 'shap_plot.png'))
    print(f"All results saved in {save_path}")
