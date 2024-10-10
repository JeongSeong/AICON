import math
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torchmetrics.functional.classification import multilabel_average_precision, multilabel_auroc, binary_average_precision, binary_auroc

import lightning.pytorch as pl

from .drfuse import DrFuseModel
from monai.visualize.utils import blend_images
import wandb
import copy
import shap
import matplotlib.pyplot as plt
import sys
import os
# # /ssd_data1/syjeong/hematoma/code/HE_fuse0901/models/ehrOnly_trainer.py
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# from FTransformer import FTTransformer
from tab_transformer_pytorch import FTTransformer
class FTT_trainer(pl.LightningModule):
    def __init__(self, args, label_names, train_dataset, valid_dataset):
        super().__init__()
        # print(FTTransformer.get_default_kwargs())
        # self.model = FTTransformer(
        #             n_cont_features=len(args.cont_ind),
        #             cat_cardinalities=args.cat_cardinalities,
        #             d_out=len(label_names),
        #             **FTTransformer.get_default_kwargs(),
        #         )
        self.model = FTTransformer(
            categories = tuple(args.cat_cardinalities), # tuple containing the number of unique values within each category
            num_continuous = len(args.cont_ind),        # number of continuous values
            dim = args.hidden_size,             # dimension, paper set at 32 # 64~192로 바꿔보기
            dim_out = 1,                        # binary prediction, but could be anything
            depth = args.ehr_n_layers,          # depth, paper recommended 6 # 1~4 로 바꿔보기
            heads = args.ehr_n_head,            # heads, paper recommends 8
            attn_dropout = 0.1,                 # post-attention dropout
            ff_dropout = 0.1                    # feed forward dropout
        )

        self.save_hyperparameters(args)  # args goes to self.hparams
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        # self.pred_criterion = nn.BCELoss(reduction='none')
        self.pred_criterion = nn.BCEWithLogitsLoss(reduction='mean') # mean

        # self.val_preds = []
        self.val_preds = {k: [] for k in ['final', 'ehr', 'cxr']}
        self.val_labels = []

        self.test_preds = []
        self.test_labels = []
        self.test_feats = {k: [] for k in ['feat_ehr_shared', 'feat_ehr_distinct',
                                           'feat_cxr_shared', 'feat_cxr_distinct']}
        self.test_attns = []

        self.label_names = label_names

        self.interp_mode = ['linear', 'bilinear', 'trilinear']

    def _compute_masked_pred_loss(self, input, target, mask):
        return (self.pred_criterion(input, target).mean(dim=1) * mask).sum() / max(mask.sum(), 1e-6)

    def _compute_and_log_loss(self, model_output, y_gt, pairs, log=True, mode='train'):

        # loss_total = self.pred_criterion(model_output[:, 0], y_gt[:, 0])#.mean(dim=1)
        loss_total = self.pred_criterion(model_output, y_gt[:, 0])#.mean(dim=1)
        epoch_log = {}

        if log:
            epoch_log.update({
                f'{mode}_loss/total': loss_total.detach(),
                # f'{mode}_loss/prediction': loss_prediction.detach(),
                'step': float(self.current_epoch)
            })
            self.log_dict(epoch_log, on_epoch=True, on_step=False, batch_size=y_gt.shape[0])

        return loss_total

    def _get_batch_data(self, batch, test=False):
        pid, cont, cat, labels = batch
        labels = torch.from_numpy(labels).float().to(self.device)
        cat = torch.from_numpy(cat).long().to(self.device)
        cont = torch.from_numpy(cont).long().to(self.device)
        return pid, cont, cat, labels


    def training_step(self, batch, batch_idx):
        pid, cont, cat, labels = self._get_batch_data(batch)
        # out = self.model(cont, cat)
        out = self.model(cat, cont)
        # print(out.shape) # (batch, 1)
        out = out.squeeze(-1)
        # print(out.shape) # (batch)
        return self._compute_and_log_loss(out, y_gt=labels, pairs=None)

    def validation_step(self, batch, batch_idx):
        pid, cont, cat, labels = self._get_batch_data(batch)
        # out = self.model(cont, cat)
        out = self.model(cat, cont)
        # print(out.shape) # (batch, 1)
        out = out.squeeze(-1)
        # print(out.shape) # (batch)
        loss = self._compute_and_log_loss(out, y_gt=labels, pairs=None, mode='val')
        pred_final =  out#['pred_final']
        self.val_preds['final'].append(pred_final)
        self.val_labels.append(labels)
        return loss # self._compute_masked_pred_loss(pred_final, y, torch.ones_like(y[:, 0]))

    def on_validation_epoch_end(self):
        for name in ['final']: # , 'ehr', 'cxr']:
            y_gt = torch.concat(self.val_labels, dim=0)
            preds = torch.concat(self.val_preds[name], dim=0)
            # mlaps = binary_average_precision(preds[:, 0], y_gt[:, 0].long())
            # mlroc = binary_auroc(preds[:, 0], y_gt[:, 0].long())
            mlaps = binary_average_precision(preds, y_gt[:, 0].long())
            mlroc = binary_auroc(preds, y_gt[:, 0].long())

            if name == 'final':
                self.log('Val_PRAUC', mlaps.mean(), logger=False, prog_bar=True)
                self.log('Val_AUROC', mlroc.mean(), logger=False, prog_bar=True)

            log_dict = {
                'step': float(self.current_epoch),
                f'val_PRAUC_avg_over_dxs/{name}': mlaps.mean(),
                f'val_AUROC_avg_over_dxs/{name}': mlroc.mean(),
            }
            log_dict[f'val_PRAUC_per_dx_{name}/{self.label_names[0]}'] = mlaps.item()
            log_dict[f'val_AUROC_per_dx_{name}/{self.label_names[0]}'] = mlroc.item()

            self.log_dict(log_dict)

        for k in self.val_preds:
            self.val_preds[k].clear()
        # self.val_pairs.clear()
        self.val_labels.clear()

    def test_step(self, batch, batch_idx):
        
        pid, cont, cat, labels = self._get_batch_data(batch, True)
        # pred_final = self.model(cont, cat) 
        pred_final = self.model(cat, cont) 
        # print(pred_final.shape) # (batch, 1)
        pred_final = pred_final.squeeze(-1)
        # print(pred_final.shape) # (batch)
        # If preds has values outside [0,1] range binary_average_precision 
        # consider the input to be logits and will auto apply sigmoid per element.
        pred_final = F.sigmoid(pred_final)
        # pred_final = torch.where(F.sigmoid(pred_final) > 0.5, 1, 0)
        self.test_preds.append(pred_final)
        self.test_labels.append(labels)

    def on_test_epoch_end(self):
        y_gt = torch.concat(self.test_labels, dim=0)
        preds = torch.concat(self.test_preds, dim=0)
        # print(preds.shape, y_gt.shape) # (batch), (batch, num_classes=2)
        # mlaps = binary_average_precision(preds[:, 0], y_gt[:, 0].long())
        # mlroc = binary_auroc(preds[:, 0], y_gt[:, 0].long())
        mlaps = binary_average_precision(preds, y_gt[:, 0].long())
        mlroc = binary_auroc(preds, y_gt[:, 0].long())
        self.test_results = {
            'y_gt': y_gt.cpu(),
            'preds': preds.cpu(),
            'mlaps': mlaps.cpu(),
            'mlroc': mlroc.cpu(),
            'prauc': mlaps.mean().item(),
            'auroc': mlroc.mean().item(),
        }
        self.test_labels.clear()
        self.test_preds.clear()


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        return optimizer

    def on_train_epoch_start(self):
        self.train_dataset.update_cache()

    def on_validation_epoch_start(self):
        self.valid_dataset.update_cache()
