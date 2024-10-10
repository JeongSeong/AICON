import math
import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import functional as F
from torchmetrics.functional.classification import multilabel_average_precision, multilabel_auroc

import lightning.pytorch as pl

from .drfuse import DrFuseModel
from monai.visualize.utils import blend_images
import wandb
import copy
import shap
import matplotlib.pyplot as plt

class ViTAttentionMap:
    def __init__(self, model):
        self.model = model
        self.attention_maps = []
        # self.hooks = []

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
                    # function = attn
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
        for blk in self.model.cxr_model_spec[0]:
            # handle = blk.attn.register_forward_hook(hook_fn)
            # self.hooks.append(handle)
            blk.attn.register_forward_hook(hook_fn)

    def get_attention_maps(self, input_tensor):
        # https://github.com/jacobgil/vit-explain/blob/main/vit_rollout.py
        self.attention_maps = []
        with torch.no_grad():
            # Hook에서 attention maps를 추출합니다.
            _ = self.model(input_tensor)
        return self.attention_maps

    # def _remove_hooks(self):
    #     for handle in self.hooks:
    #         handle.remove()
    #     self.hooks.clear()

    # def __del__(self):
    #     self._remove_hooks()    

class JSD(nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction='none', log_target=True)

    def forward(self, p: torch.tensor, q: torch.tensor, masks):
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log()
        return 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log())).sum() / max(1e-6, masks.sum())


class DrFuseTrainer(pl.LightningModule):
    def __init__(self, args, label_names, train_dataset, valid_dataset):
        super().__init__()
        self.model = DrFuseModel(hidden_size=args.hidden_size,
                                 num_classes=len(label_names),
                                 ehr_dropout=args.dropout,
                                 ehr_n_head=args.ehr_n_head,
                                 ehr_n_layers=args.ehr_n_layers)

        self.save_hyperparameters(args)  # args goes to self.hparams
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.pred_criterion = nn.BCELoss(reduction='none')
        self.alignment_cos_sim = nn.CosineSimilarity(dim=1)
        self.triplet_loss = nn.TripletMarginLoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')
        self.jsd = JSD()

        # self.val_preds = []
        self.val_preds = {k: [] for k in ['final', 'ehr', 'cxr']}
        self.val_labels = []
        self.val_pairs = []

        self.test_preds = []
        self.test_labels = []
        self.test_pairs = []
        self.test_feats = {k: [] for k in ['feat_ehr_shared', 'feat_ehr_distinct',
                                           'feat_cxr_shared', 'feat_cxr_distinct']}
        self.test_attns = []

        self.label_names = label_names

        self.interp_mode = ['linear', 'bilinear', 'trilinear']

        if args.img_backbone == 'CNN':
            # target_layer = self.model.cxr_model_spec[3][2].conv3
            target_layer = self.model.cxr_model_spec[0][11]
            target_layer.register_forward_hook(self.save_activation)
            target_layer.register_full_backward_hook(self.save_gradient)
            self.columns = ["sex", "age", "ivh", "sbp", "dbp", "e", "v", "m", "inr", "fibrinogen", "platelet", "aptt", "bt", "anti_p_anti_coag"]
            self.activations = None
            self.gradients = None
        else: # if args.img_backbone == 'Attention':
            print('see attention map in test')
            # self.attention_map_module = ViTAttentionMap(self.model)
            # # self.attention_maps = []
            # # def hook_fn(module, input, output):
            # #     # Attention weights는 softmax로 normalize되기 전의 값을 사용합니다.
            # #     attention_weights = output[1]
            # #     self.attention_maps.append(attention_weights)
            # # # ViT의 각 Transformer block에 hook을 설정합니다.
            # # for blk in self.model.model.blocks:
            # #     blk.attn.register_forward_hook(hook_fn)

    def save_activation(self, module, input, output):
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def _compute_masked_pred_loss(self, input, target, mask):
        return (self.pred_criterion(input, target).mean(dim=1) * mask).sum() / max(mask.sum(), 1e-6)

    def _masked_abs_cos_sim(self, x, y, mask):
        return (self.alignment_cos_sim(x, y).abs() * mask).sum() / max(mask.sum(), 1e-6)

    def _masked_cos_sim(self, x, y, mask):
        return (self.alignment_cos_sim(x, y) * mask).sum() / max(mask.sum(), 1e-6)

    def _masked_mse(self, x, y, mask):
        return (self.mse_loss(x, y).mean(dim=1) * mask).sum() / max(mask.sum(), 1e-6)

    def _disentangle_loss_jsd(self, model_output, pairs, log=True, mode='train'):
        ehr_mask = torch.ones_like(pairs)
        loss_sim_cxr = self._masked_abs_cos_sim(model_output['feat_cxr_shared'],
                                                model_output['feat_cxr_distinct'], pairs)
        loss_sim_ehr = self._masked_abs_cos_sim(model_output['feat_ehr_shared'],
                                                model_output['feat_ehr_distinct'], ehr_mask)

        jsd = self.jsd(model_output['feat_ehr_shared'].sigmoid(),
                       model_output['feat_cxr_shared'].sigmoid(), pairs)

        loss_disentanglement = (self.hparams.lambda_disentangle_shared * jsd +
                                self.hparams.lambda_disentangle_ehr * loss_sim_ehr +
                                self.hparams.lambda_disentangle_cxr * loss_sim_cxr)
        if log:
            self.log_dict({
                f'disentangle_{mode}/EHR_disinct': loss_sim_ehr.detach(),
                f'disentangle_{mode}/CXR_disinct': loss_sim_cxr.detach(),
                f'disentangle_{mode}/shared_jsd': jsd.detach(),
                'step': float(self.current_epoch)
            }, on_epoch=True, on_step=False, batch_size=pairs.shape[0])

        return loss_disentanglement

    def _compute_prediction_losses(self, model_output, y_gt, pairs, log=True, mode='train'):
        ehr_mask = torch.ones_like(model_output['pred_final'][:, 0])
        loss_pred_final = self._compute_masked_pred_loss(model_output['pred_final'], y_gt, ehr_mask)
        loss_pred_ehr = self._compute_masked_pred_loss(model_output['pred_ehr'], y_gt, ehr_mask)
        loss_pred_cxr = self._compute_masked_pred_loss(model_output['pred_cxr'], y_gt, pairs)
        loss_pred_shared = self._compute_masked_pred_loss(model_output['pred_shared'], y_gt, ehr_mask)

        if log:
            self.log_dict({
                f'{mode}_loss/pred_final': loss_pred_final.detach(),
                f'{mode}_loss/pred_shared': loss_pred_shared.detach(),
                f'{mode}_loss/pred_ehr': loss_pred_ehr.detach(),
                f'{mode}_loss/pred_cxr': loss_pred_cxr.detach(),
                'step': float(self.current_epoch)
            }, on_epoch=True, on_step=False, batch_size=y_gt.shape[0])

        return loss_pred_final, loss_pred_ehr, loss_pred_cxr, loss_pred_shared

    def _compute_and_log_loss(self, model_output, y_gt, pairs, log=True, mode='train'):
        prediction_losses = self._compute_prediction_losses(model_output, y_gt, pairs, log, mode)
        loss_pred_final, loss_pred_ehr, loss_pred_cxr, loss_pred_shared = prediction_losses

        loss_prediction = (self.hparams.lambda_pred_shared * loss_pred_shared +
                           self.hparams.lambda_pred_ehr * loss_pred_ehr +
                           self.hparams.lambda_pred_cxr * loss_pred_cxr)

        loss_prediction = loss_pred_final + loss_prediction

        loss_disentanglement = self._disentangle_loss_jsd(model_output, pairs, log, mode)

        loss_total = loss_prediction + loss_disentanglement
        epoch_log = {}

        # aux loss for attention ranking
        raw_pred_loss_ehr = F.binary_cross_entropy(model_output['pred_ehr'].data, y_gt, reduction='none')
        raw_pred_loss_cxr = F.binary_cross_entropy(model_output['pred_cxr'].data, y_gt, reduction='none')
        raw_pred_loss_shared = F.binary_cross_entropy(model_output['pred_shared'].data, y_gt, reduction='none')

        pairs = pairs.unsqueeze(1)
        attn_weights = model_output['attn_weights']
        attn_ehr, attn_shared, attn_cxr = attn_weights[:, :, 0], attn_weights[:, :, 1], attn_weights[:, :, 2]

        cxr_overweights_ehr = 2 * (raw_pred_loss_cxr < raw_pred_loss_ehr).float() - 1
        loss_attn1 = pairs * F.margin_ranking_loss(attn_cxr, attn_ehr, cxr_overweights_ehr, reduction='none')
        loss_attn1 = loss_attn1.sum() / max(1e-6, loss_attn1[loss_attn1>0].numel())

        shared_overweights_ehr = 2 * (raw_pred_loss_shared < raw_pred_loss_ehr).float() - 1
        loss_attn2 = pairs * F.margin_ranking_loss(attn_shared, attn_ehr, shared_overweights_ehr, reduction='none')
        loss_attn2 = loss_attn2.sum() / max(1e-6, loss_attn2[loss_attn2>0].numel())

        shared_overweights_cxr = 2 * (raw_pred_loss_shared < raw_pred_loss_cxr).float() - 1
        loss_attn3 = pairs * F.margin_ranking_loss(attn_shared, attn_cxr, shared_overweights_cxr, reduction='none')
        loss_attn3 = loss_attn3.sum() / max(1e-6, loss_attn3[loss_attn3>0].numel())

        loss_attn_ranking = (loss_attn1 + loss_attn2 + loss_attn3) / 3

        loss_total = loss_total + self.hparams.lambda_attn_aux * loss_attn_ranking
        epoch_log[f'{mode}_loss/attn_aux'] = loss_attn_ranking.detach()

        if log:
            epoch_log.update({
                f'{mode}_loss/total': loss_total.detach(),
                f'{mode}_loss/prediction': loss_prediction.detach(),
                'step': float(self.current_epoch)
            })
            self.log_dict(epoch_log, on_epoch=True, on_step=False, batch_size=y_gt.shape[0])

        return loss_total

    def _get_batch_data(self, batch, test=False):
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
        x, img, y, seq_lengths, pairs = self._get_batch_data(batch)
        if self.hparams.data_pair == 'paired' and self.hparams.aug_missing_ratio > 0:
            perm = torch.randperm(pairs.shape[0])
            idx = perm[:int(self.hparams.aug_missing_ratio * pairs.shape[0])]
            pairs[idx] = 0
        out = self.model([x, img, seq_lengths, pairs])
        return self._compute_and_log_loss(out, y_gt=y, pairs=pairs)

    def validation_step(self, batch, batch_idx):
        x, img, y, seq_lengths, pairs = self._get_batch_data(batch)
        out = self.model([x, img, seq_lengths, pairs])
        loss = self._compute_and_log_loss(out, y_gt=y, pairs=pairs, mode='val')
        pred_final =  out['pred_final']
        # print(pred_final.shape) # (batch, num_classes)
        # self.val_preds.append(out['pred_final'])
        self.val_preds['final'].append(pred_final)
        self.val_preds['ehr'].append(out['pred_ehr'])
        self.val_preds['cxr'].append(out['pred_cxr'])
        self.val_pairs.append(pairs)
        self.val_labels.append(y)

        # return self._compute_masked_pred_loss(out['pred_final'], y, torch.ones_like(y[:, 0]))
        return self._compute_masked_pred_loss(pred_final, y, torch.ones_like(y[:, 0]))

    def on_validation_epoch_end(self):
        for name in ['final', 'ehr', 'cxr']:
            y_gt = torch.concat(self.val_labels, dim=0)
            preds = torch.concat(self.val_preds[name], dim=0)
            if name == 'cxr':
                pairs = torch.concat(self.val_pairs, dim=0)
                y_gt = y_gt[pairs==1, :]
                preds = preds[pairs==1, :]

            mlaps = multilabel_average_precision(preds, y_gt.long(), num_labels=y_gt.shape[1], average=None)
            mlroc = multilabel_auroc(preds, y_gt.long(), num_labels=y_gt.shape[1], average=None)

            if name == 'final':
                self.log('Val_PRAUC', mlaps.mean(), logger=False, prog_bar=True)
                self.log('Val_AUROC', mlroc.mean(), logger=False, prog_bar=True)

            log_dict = {
                'step': float(self.current_epoch),
                f'val_PRAUC_avg_over_dxs/{name}': mlaps.mean(),
                f'val_AUROC_avg_over_dxs/{name}': mlroc.mean(),
            }
            for i in range(mlaps.shape[0]):
                log_dict[f'val_PRAUC_per_dx_{name}/{self.label_names[i]}'] = mlaps[i]
                log_dict[f'val_AUROC_per_dx_{name}/{self.label_names[i]}'] = mlroc[i]

            self.log_dict(log_dict)

        for k in self.val_preds:
            self.val_preds[k].clear()
        self.val_pairs.clear()
        self.val_labels.clear()

    def on_test_epoch_start(self):
        self.attention_map_module = ViTAttentionMap(self.model)

    def test_step(self, batch, batch_idx):
        self.automatic_optimization = False
        torch.set_grad_enabled(True)
        
        pid, x, img, y, seq_lengths, pairs = self._get_batch_data(batch, True)
        
        out = self.model([x, img, seq_lengths, pairs]) 
        pred_final =  out['pred_final']
        
        self.test_preds.append(pred_final)
        self.test_labels.append(y)
        self.test_pairs.append(pairs)
        self.test_attns.append(out['attn_weights'])

        for k in self.test_feats:
            self.test_feats[k].append(out[k].cpu())

        if self.hparams.img_backbone == 'CNN':
            self.plot_gradCAM(self, self.activations, self.gradients, out, pid, x, img, y, seq_lengths, pairs)
        else: # if self.hparams.img_backbone == 'Attention':
            with torch.no_grad():
                attention_maps = self.attention_map_module.get_attention_maps([x, img, seq_lengths, pairs])
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
        # x = x.cpu()# .numpy()
        # explainer = shap.DeepExplainer(self.model.ehr_model.cpu(), x) # (shapModelWrapper(self.model.ehr_model.cpu()), x)
        # explainer = shap.GradientExplainer(self.model.ehr_model.cpu(), x) # (shapModelWrapper(self.model.ehr_model.cpu()), x)
        # CLS 토큰을 제거하고 남은 패치에 대한 attention map을 시각화합니다.
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
        # print(attention_map.shape) # torch.Size([1, 12, 3, 384, 384])
        attention_map = attention_map.squeeze().numpy()#.cpu().numpy()
        # print(attention_map.shape) # (12, 3, 384, 384)
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
        img = img * 255
        for patient_id, ehr, ct, label, cam, pred in zip(pid, x, img, y, attention_map, pred_final):
            # shap_values = explainer.shap_values(ehr)
            # # print(shap_values)
            # plt.figure()  # fig, ax = plt.subplots()
            # shap.summary_plot(shap_values, ehr, show=False)

            # # 기존 ax에 수동으로 그림 복사
            # for item in plt.gca().get_children():
            #     item.set_clip_on(False)
            # ax.imshow(plt.gca().get_images()[0].get_array())

            ct = torch.cat(list(ct), dim=-1).unsqueeze(0).repeat(3, 1, 1).cpu()
            cam = torch.cat(list(torch.from_numpy(cam)), dim=-1).unsqueeze(0) * 255 #.cpu() * 255
            blended = blend_images(ct, cam, alpha=0.5, cmap='jet')*255 # torch.permute(blend_images(ct, cam, alpha=0.5, cmap='hsv'), (1, 2, 0))
            cam = cam.repeat(3, 1, 1)
            # print(torch.min(ct), torch.min(cam), torch.min(blended), torch.max(ct), torch.max(cam), torch.max(blended))
            if isinstance(self.logger, pl.loggers.WandbLogger):
                # self.logger.log_table(key=f"test_label{int(label[0])}/{patient_id}/clinical", data=[[i.item() for i in ehr[0]]], columns=self.columns)
                # self.logger.log_image(key=f"test_label{int(label[0])}/{patient_id}", 
                #                       images=[torch.permute(torch.cat([ct, cam, blended], dim=1), (1, 2, 0)).numpy(), plt.gcf()], # fig
                #                       caption=[f'prediction_label: {pred}', 'ehr']
                #                       )
                self.logger.log_image(key=f"test_label{int(label[0])}/{patient_id}", 
                                      images=[torch.permute(torch.cat([ct, cam, blended], dim=1), (1, 2, 0)).numpy()], # fig
                                      caption=[f'prediction_label: {pred}']
                                      )
        self.model.to(self.device)

    def plot_gradCAM(self, activations, gradients, out, pid, x, img, y, seq_lengths, pairs):
        loss = self._compute_and_log_loss(out, y_gt=y, pairs=pairs, mode='val')
        print(loss)
        loss.requires_grad_()
        self.manual_backward(loss)
        # for name, param in self.model.named_parameters():
        #     if param.grad is None:
        #         print(f"Parameter {name} has no gradient.")
        grad_cam = self.compute_grad_cam(self.activations, self.gradients)
        # print(grad_cam.shape, grad_cam[0].shape, img[0].shape)
        interp_mode = self.interp_mode[img[0].ndim-1] # interp_mode = ['linear', 'bilinear', 'trilinear']
        # print(grad_cam.shape, img.shape) # torch.Size([48, 24, 24]) torch.Size([1, 3, 384, 384])
        grad_cam = F.interpolate(grad_cam.unsqueeze(1).unsqueeze(0), size=img.shape[-3:], mode=interp_mode, align_corners=False)
        # print(grad_cam.shape) # torch.Size([1, 48, 3, 384, 384])
        grad_cam = grad_cam.squeeze().cpu().numpy()
        # print(grad_cam.shape) # (48, 3, 384, 384)
        # print(torch.min(grad_cam), torch.max(grad_cam))
        grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min())
        img = img * 255
        # print(torch.min(grad_cam), torch.max(grad_cam))
        # self.columns = ["sex", "age", "ivh", "sbp", "dbp", "e", "v", "m", "inr", "fibrinogen", "platelet", "aptt", "bt", "anti_p_anti_coag"]
        for patient_id, ehr, ct, label, cam in zip(pid, x, img, y, grad_cam):
            # id, label이랑 image를 tabel에 넣는것도 생각해보기
            # columns = ["id", "label", "CT", "gradCAM", "sex", "age", "ivh", "sbp", "dbp", "e", "v", "m", "inr", "fibrinogen", "platelet", "aptt", "bt", "anti_p_anti_coag"]
            # ehr 데이터 형식 확인
            # data = [[patient_id, label, wandb.Image(ct), wandb.Image(cam)].extend(list(ehr)), [], [], ...]
            # clinical variable이 normalize된 채로 get됨
            # print(patient_id, label[0], ct.shape, cam.shape, ehr)
            # [(224, 224), (224, 224), (224, 224)] -> (224, 224 * 3) -> (3, 224, 224*3)
            ct = torch.cat(list(ct), dim=-1).unsqueeze(0).repeat(3, 1, 1).cpu()
            cam = torch.cat(list(torch.from_numpy(cam)), dim=-1).unsqueeze(0).cpu() * 255
            blended = blend_images(ct, cam, alpha=0.5, cmap='jet') # torch.permute(blend_images(ct, cam, alpha=0.5, cmap='hsv'), (1, 2, 0))
            cam = cam.repeat(3, 1, 1)
            # print(torch.max(ct), torch.max(cam), torch.max(blended))
            # (3, 224*2, 224*3) -> (224*2, 224*3, 3)
            if isinstance(self.logger, pl.loggers.WandbLogger):
                # self.logger.log_table(key=f"test_label{int(label[0])}/{patient_id}/clinical", data=[[i.item() for i in ehr[0]]], columns=self.columns)
                self.logger.log_image(key=f"test_label{int(label[0])}/{patient_id}", 
                                      images=[torch.permute(torch.cat([ct, cam, blended], dim=1), (1, 2, 0)).numpy()]
                                      )
            
    def compute_grad_cam(self, activations, gradients):
        # Grad-CAM 계산 코드
        # print('gradients', gradients.shape) # [batch, 577, 384] # vit hidden_feature 
        # print('activations', activations.shape)# [1, 577, 384] # vit hidden_feature 

        weights = torch.mean(gradients, dim=[2, 3], keepdim=True) # CNN (batch, channel, 1, 1)
        grad_cam = torch.relu(torch.sum(weights * activations, dim=1))
        
        # gradients = torch.permute(gradients[:, 1:, :], (0, 2, 1))  # (batch, 384, 576)
        # activations = torch.permute(activations[:, 1:, :], (0, 2, 1))  # (1, 384, 576)
        
        # size = int(math.sqrt(gradients.size(2)))
        # gradients = gradients.view(gradients.size(0), gradients.size(1), size, size)
        # activations = activations.view(activations.size(0), activations.size(1), size, size) #(batch, hidden_size, 24, 24)

        # weights = torch.mean(gradients, dim=[2, 3], keepdim=True) #(batch, hidden_size, 1, 1)
        # grad_cam = torch.relu(torch.sum(weights * activations, dim=1))
        # # print(gradients.shape, weights.shape, activations.shape, grad_cam.shape)
        # # # torch.Size([48, 384, 24, 24]) torch.Size([48, 384, 1, 1]) torch.Size([1, 384, 24, 24]) torch.Size([48, 24, 24])


        # # weights = torch.mean(gradients, dim=-1, keepdim=False) # (batch, 576) # sum으로 바꾸는 거 생각해보기
        # # # activations = torch.mean(activations, dim=-1, keepdim=False) # (1, 576) # sum으로 바꾸는 거 생각해보기
        # # weights = weights.view(weights.size(0), 24, 24)  # (batch, 24, 24) # 24 = root(576)
        # # activations = activations.view(activations.size(0), 24, 24) # (1, 24, 24) # 24 = root(576)
        # # weights = weights.unsqueeze(1)
        # # activations = activations.unsqueeze(1)
        # # weights = torch.mean(weights, dim=[2, 3], keepdim=True)
        # # grad_cam = torch.relu(torch.sum(weights * activations, dim=1))  # (batch,)
        
        return grad_cam

    def on_test_epoch_end(self):
        y_gt = torch.concat(self.test_labels, dim=0)
        preds = torch.concat(self.test_preds, dim=0)
        pairs = torch.concat(self.test_pairs, dim=0)
        attn_weights = torch.concat(self.test_attns, dim=0)
        mlaps = multilabel_average_precision(preds, y_gt.long(), num_labels=y_gt.shape[1], average=None)
        mlroc = multilabel_auroc(preds, y_gt.long(), num_labels=y_gt.shape[1], average=None)
        self.test_results = {
            'y_gt': y_gt.cpu(),
            'preds': preds.cpu(),
            'pairs': pairs.cpu(),
            'mlaps': mlaps.cpu(),
            'mlroc': mlroc.cpu(),
            'prauc': mlaps.mean().item(),
            'auroc': mlroc.mean().item(),
            'attn_weight': attn_weights.cpu()
        }
        for k in self.test_feats:
            self.test_results[k] = torch.concat(self.test_feats[k], dim=0)
            self.test_feats[k].clear()
        self.test_labels.clear()
        self.test_preds.clear()
        self.test_pairs.clear()
        # if isinstance(self.logger, pl.loggers.WandbLogger):
        #     self.logger.watch(copy.copy(self.model))


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        # scheduler = ReduceLROnPlateau(optimizer) # , mode='min', factor=0.5, patience=10, verbose=True

        # return {
        #     'optimizer': optimizer,
        #     'lr_scheduler': {
        #         'scheduler': scheduler,
        #         'monitor': 'val_loss/total',  # 또는 다른 validation metric을 사용
        #         'interval': 'epoch',    # 또는 'step'을 사용
        #         'frequency': 1,         # 호출 빈도
        #     },
        # }
        return optimizer

    def on_train_epoch_start(self):
        self.train_dataset.update_cache()

    def on_validation_epoch_start(self):
        self.valid_dataset.update_cache()
