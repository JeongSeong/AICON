import math
import sys
import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import functional as F
from torchmetrics.functional.classification import multilabel_average_precision, multilabel_auroc, binary_average_precision, binary_auroc
import timm
import lightning.pytorch as pl

# from .drfuse import DrFuseModel
import monai.networks.nets as nets
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights, resnet34, ResNet34_Weights
from monai.visualize.utils import blend_images
from monai.visualize import GradCAM

# from utils import ViTAttentionMap
# import wandb
# import copy
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class ViTAttentionMap:
    def __init__(self, model):
        self.model = model
        self.attention_maps = []
        # Hook을 걸어 attention weights를 저장합니다.
        self._register_hooks()

    def _register_hooks(self):
        def hook_fn(module, input, output):
            with torch.no_grad():
                # timm/models/vision_transformer.py
                input = input[0]
                B, N, C = input.shape
                # print(module.num_heads) # 6개
                qkv = module.qkv(input).reshape(B, N, 3, module.num_heads, module.head_dim).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)
                q, k = module.q_norm(q), module.k_norm(k)
                # print(q.shape, k.shape, v.shape) # torch.Size([1, 6, 577, 64]) torch.Size([1, 6, 577, 64]) torch.Size([1, 6, 577, 64])
                if module.fused_attn:
                    # x = F.scaled_dot_product_attention(
                    #     q, k, v,
                    #     dropout_p=0 # training=False
                    # )
                    # L, S = q.size(-2), k.size(-2)
                    # scale_factor = 1 / math.sqrt(q.size(-1))
                    # attn_bias = torch.zeros(L, S, dtype=q.dtype, device=q.device)
                    # attn = q @ k.transpose(-2, -1) * scale_factor
                    # attn += attn_bias
                    # attn = torch.softmax(attn, dim=-1)
                    # # print(attn.shape)
                    # function = attn[:]
                    #################################################################################
                    attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
                    attn = torch.softmax(attn, dim=-1)
                    # print(torch.equal(function, attn))
                else:
                    q = q * module.scale
                    attn = q @ k.transpose(-2, -1)
                    attn = attn.softmax(dim=-1)
                    # attn = self.model.custom.attn_drop(attn)
                    # x = attn @ v
                # print(input.shape, attn.shape) # torch.Size([1, 577, 384]) torch.Size([1, 6, 577, 577])
                self.attention_maps.append(attn.cpu()) 
                del qkv, q, k, v, attn

        # ViT의 각 Transformer block에 hook을 설정합니다.
        for blk in self.model.blocks:
            blk.attn.register_forward_hook(hook_fn)

    def get_attention_maps(self, input_tensor):
        # https://github.com/jacobgil/vit-explain/blob/main/vit_rollout.py
        self.attention_maps = []
        with torch.no_grad():
            # Hook에서 attention maps를 추출합니다.
            _ = self.model(input_tensor)
        return self.attention_maps


def fetch_model(
    model_name, spatial_dims=2, in_channels=3, out_channels=1
):
    model_name = model_name.lower()
    # print(model_name)
    if model_name == "densenet121":
        model =  nets.DenseNet121(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
        )
    elif model_name == "densenet169": 
        model = nets.DenseNet169(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
        )
    elif model_name == "densenet264":
        model = nets.DenseNet264(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
        )
        # "swinunetr": nets.SwinUNETR(
        #     img_size=img_size, # segmentation
        #     spatial_dims=spatial_dims,
        #     in_channels=in_channels,
        #     out_channels=out_channels,
        # ),
    elif model_name == "seresnet50": 
        model = nets.SEResNet50(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            dropout_prob=0.4,
            num_classes=2,
            pretrained=True,
        )
    elif model_name == "seresnet101": 
        model = nets.SEResNet101(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            dropout_prob=0.4,
            num_classes=2,
            pretrained=True,
        )
    elif model_name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(512 * 4, 2)
    elif model_name == "resnet34":
        model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(512, 2)
    elif model_name == "resnet18":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(512, 2)
    elif model_name == "seresnext50": 
        model = nets.SEResNext50(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            dropout_prob=0.3,
            num_classes=2,
            pretrained=True,
        )
    elif model_name == "seresnext50_timm": 
        model = timm.create_model(
            model_name='seresnext50_32x4d', 
            pretrained=True,
            num_classes=2,
        )

    elif model_name == "seresnext101": 
        model = nets.SEResNext101(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            dropout_prob=0.3,
            num_classes=2,
            pretrained=True,
        )
    elif model_name == "vit_small_patch16_384": 
        model = timm.create_model(
            model_name=model_name,
            pretrained=True, # True # False
            in_chans=in_channels,
            img_size=(384, 384),
            num_classes=2,
        )

    return model

class imgOnlyTrainer(pl.LightningModule):
    def __init__(self, args, label_names, train_dataset, valid_dataset):
        super().__init__()
        self.model = fetch_model(
            args.model_name,
            spatial_dims=2,
            in_channels=3,
            out_channels=1,
        )

        self.save_hyperparameters(args)  # args goes to self.hparams
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.pred_criterion = nn.BCEWithLogitsLoss(reduction='mean') # mean
        self.val_preds = []
        self.val_preds = {'final':[]}
        self.val_labels = []
        self.val_pairs = []

        self.test_preds = []
        self.test_labels = []
        self.test_pairs = []
        self.test_attns = []

        self.see_val_img = False

        self.label_names = label_names

        self.interp_mode = ['linear', 'bilinear', 'trilinear']

    def _compute_and_log_loss(self, model_output, y_gt, pairs, log=True, mode='train'):

        loss_total = self.pred_criterion(model_output[:, 0], y_gt[:, 0])
        epoch_log = {}

        if log:
            epoch_log.update({
                f'{mode}_loss/total': loss_total.detach(),
                'step': float(self.current_epoch)
            })
            self.log_dict(epoch_log, on_epoch=True, on_step=False, batch_size=y_gt.shape[0])

        return loss_total

    def _get_batch_data(self, batch, test=True):
        if test:
            pid, x, img, y_ehr, seq_lengths, pairs = batch
        else:
            x, img, y_ehr, seq_lengths, pairs = batch
        y = torch.from_numpy(y_ehr).float().to(self.device)
        x = torch.from_numpy(x).float().to(self.device)
        img = img.to(self.device)
        pairs = torch.FloatTensor(pairs).to(self.device)
        seq_lengths = torch.FloatTensor(seq_lengths).to(self.device) # added to the original
        if test:
            return pid, x, img, y, seq_lengths, pairs
        else:
            return x, img, y, seq_lengths, pairs


    def training_step(self, batch, batch_idx):
        pid, x, img, y, seq_lengths, pairs = self._get_batch_data(batch)
        out = self.model(img)
        # with torch.no_grad():
        #     plot = img[0].cpu()*255 # (3, H, W)
        #     plot = torch.cat(list(plot), dim=-1).unsqueeze(0).repeat(3, 1, 1)
        #     if isinstance(self.logger, pl.loggers.WandbLogger):
        #         # self.logger.log_table(key=f"test_label{int(label[0])}/{patient_id}/clinical", data=[[i.item() for i in ehr[0]]], columns=self.columns)
        #         self.logger.log_image(key="training_sample", 
        #                               images=[torch.permute(plot, (1, 2, 0)).numpy()],
        #                               caption=[f'patient number: {pid[0]}'])
        return self._compute_and_log_loss(out, y_gt=y, pairs=pairs)

    def validation_step(self, batch, batch_idx):
        pid, x, img, y, seq_lengths, pairs = self._get_batch_data(batch)
        out = self.model(img) # torch.Size([batch, num_classes])
        loss = self._compute_and_log_loss(out, y_gt=y, pairs=pairs, mode='val')
        pred_final =  out#['pred_final']
        self.val_preds['final'].append(pred_final)
        self.val_pairs.append(pairs)
        self.val_labels.append(y)
        if not self.see_val_img:
            with torch.no_grad():
                plot = img[0].cpu()*255 # (3, H, W)
                plot = torch.cat(list(plot), dim=-1).unsqueeze(0).repeat(3, 1, 1)
                if isinstance(self.logger, pl.loggers.WandbLogger):
                    # self.logger.log_table(key=f"test_label{int(label[0])}/{patient_id}/clinical", data=[[i.item() for i in ehr[0]]], columns=self.columns)
                    self.logger.log_image(key="training_sample", 
                                        images=[torch.permute(plot, (1, 2, 0)).numpy()],
                                        caption=[f'patient number: {pid[0]}'])
            self.see_val_img = True
        return loss

    def on_validation_epoch_end(self):
        for name in ['final']:#, 'ehr', 'cxr']:
            y_gt = torch.concat(self.val_labels, dim=0)
            preds = torch.concat(self.val_preds[name], dim=0)
            mlaps = binary_average_precision(preds[:, 0], y_gt[:, 0].long())
            mlroc = binary_auroc(preds[:, 0], y_gt[:, 0].long())

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
        self.val_pairs.clear()
        self.val_labels.clear()

    def on_test_epoch_start(self):

        if self.hparams.img_backbone == 'CNN':
            self.columns = ["sex", "age", "ivh", "sbp", "dbp", "e", "v", "m", "inr", "fibrinogen", "platelet", "aptt", "bt", "anti_p_anti_coag"]

        elif self.hparams.img_backbone == 'Attention':
            self.attention_map_module = ViTAttentionMap(self.model)

    def test_step(self, batch, batch_idx):
        if self.hparams.img_backbone == 'CNN':
            pid, x, img, y, seq_lengths, pairs = self._get_batch_data(batch, True)
            out = self.model(img) 
            # print(out)
            pred_final =  F.sigmoid(out)#['pred_final']
            self.test_preds.append(pred_final)
            self.test_labels.append(y)
            self.test_pairs.append(pairs)
        else: # if self.hparams.img_backbone == 'Attention':
            pid, x, img, y, seq_lengths, pairs = self._get_batch_data(batch, True)
            out = self.model(img) 
            # print(out)
            pred_final =  F.sigmoid(out)#['pred_final']
            
            self.test_preds.append(pred_final)
            self.test_labels.append(y)
            self.test_pairs.append(pairs)

            with torch.no_grad():
                attention_maps = self.attention_map_module.get_attention_maps(img) # [x, img, seq_lengths, pairs]
               # 마지막 block attention map 시각화
                # https://hongl.tistory.com/234 # rollout
                discard_ratio = 0
                result = torch.eye(attention_maps[0].size(-1), device=attention_maps[0].device)
                for attention in attention_maps:
                    # print(attention.shape) # torch.Size([12, 6, 577, 577])
                    attention_heads_fused = attention.mean(dim=1) 
                    # attention_heads_fused = attention.max(dim=1)[0]
                    # attention_heads_fused = attention.min(dim=1)[0]
                    # print(attention.shape) # torch.Size([12, 577, 577])
                    flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
                    k = int(flat.size(-1)*discard_ratio)
                    for batch in range(flat.shape[0]):
                        _, indices = flat[batch].topk(k, dim=-1, largest=False)
                        indices = indices[indices!=0]
                        # print(flat.shape) # torch.Size([12, 332929])
                        flat[batch, indices] = 0
                    I = torch.eye(attention_heads_fused.size(-1), device=attention_heads_fused.device)
                    a = (attention_heads_fused + 1.0*I)/2
                    # print(I.shape, a.shape) # torch.Size([577, 577]) torch.Size([12, 577, 577])
                    a = a/a.sum(dim=-1, keepdim=True)
                    # print(a.shape, result.shape) # torch.Size([12, 577, 577]) torch.Size([577, 577])
                    result = torch.matmul(a, result)
                    # print(result.shape) # torch.Size([12, 577, 577])
            del I, a, attention_maps, attention, attention_heads_fused, flat
            self.plot_attention_map(result, pid, x, img, y, pred_final)
    
    def plot_attention_map(self, attention_map, pid, x, img, y, pred_final):
        # CLS 토큰을 제거하고 남은 패치에 대한 attention map을 시각화합니다.
        # print(attention_map.shape)
        attention_map = attention_map[:, 0, 1:]  # 첫번째 CLS 토큰에 대한, 토큰이 아닌 feature만 추출 # []
        attention_map = attention_map.numpy()  #.cpu().numpy()  
        # print(attention_map.shape) # torch.Size([batch, 576])
        num_patches = int(math.sqrt(attention_map.shape[-1]))
        attention_map = attention_map.reshape(attention_map.shape[0], num_patches, num_patches)
        interp_mode = self.interp_mode[img[0].ndim-1] # interp_mode = ['linear', 'bilinear', 'trilinear']
        # print(attention_map.shape, img.shape) # (batch, 24, 24) torch.Size([batch, 3, 384, 384])
        attention_map = torch.from_numpy(attention_map).unsqueeze(1).unsqueeze(0)
        # print(attention_map.shape,img.shape, interp_mode) # torch.Size([1, 12, 1, 24, 24]) torch.Size([12, 3, 384, 384]) trilinear
        attention_map = F.interpolate(attention_map, size=img.shape[-3:], mode=interp_mode, align_corners=False)
        # print(attention_map.shape) # torch.Size([1, batch, 3, 384, 384])
        attention_map = attention_map.squeeze(0).numpy()#.cpu().numpy()
        # print(attention_map.shape) # (batch, 3, 384, 384)
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
        img = img * 255
        # print(img.shape, attention_map.shape)
        for patient_id, ehr, ct, label, cam, pred in zip(pid, x, img, y, attention_map, pred_final):
            ct = torch.cat(list(ct), dim=-1).unsqueeze(0).repeat(3, 1, 1).cpu()
            cam = torch.cat(list(torch.from_numpy(cam)), dim=-1).unsqueeze(0) * 255 #.cpu() * 255
            blended = blend_images(ct, cam, alpha=0.5, cmap='jet')*255 # torch.permute(blend_images(ct, cam, alpha=0.5, cmap='hsv'), (1, 2, 0))
            cam = cam.repeat(3, 1, 1)
            # print(torch.min(ct), torch.min(cam), torch.min(blended), torch.max(ct), torch.max(cam), torch.max(blended))
            if isinstance(self.logger, pl.loggers.WandbLogger):
                # self.logger.log_table(key=f"test_label{int(label[0])}/{patient_id}/clinical", data=[[i.item() for i in ehr[0]]], columns=self.columns)
                self.logger.log_image(key=f"test_label{int(label[0])}/{patient_id}", 
                                      images=[torch.permute(torch.cat([ct, cam, blended], dim=1), (1, 2, 0)).numpy()],
                                      caption=[f'prediction_label: {pred}']
                                      )

    def on_test_epoch_end(self):
        y_gt = torch.concat(self.test_labels, dim=0)
        preds = torch.concat(self.test_preds, dim=0)
        pairs = torch.concat(self.test_pairs, dim=0)
        # print(preds.shape, y_gt.shape)
        mlaps = binary_average_precision(preds[:, 0], y_gt[:, 0].long())
        mlroc = binary_auroc(preds[:, 0], y_gt[:, 0].long())
        self.test_results = {
            'y_gt': y_gt.cpu(),
            'preds': preds.cpu(),
            'pairs': pairs.cpu(),
            'mlaps': mlaps.cpu(),
            'mlroc': mlroc.cpu(),
            'prauc': mlaps.mean().item(),
            'auroc': mlroc.mean().item(),
        }
        self.test_labels.clear()
        self.test_preds.clear()
        self.test_pairs.clear()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        # if self.hparams.img_backbone == 'Attention':
        #     scheduler = ReduceLROnPlateau(optimizer) # , mode='min', factor=0.5, patience=10, verbose=True
        #     return {
        #         'optimizer': optimizer,
        #         'lr_scheduler': {
        #             'scheduler': scheduler,
        #             'monitor': 'val_loss/total',  # 또는 다른 validation metric을 사용
        #             'interval': 'epoch',    # 또는 'step'을 사용
        #             'frequency': 1,         # 호출 빈도
        #         },
        #     }
        return optimizer

    def on_train_epoch_start(self):
        self.train_dataset.update_cache()

    def on_validation_epoch_start(self):
        self.valid_dataset.update_cache()
