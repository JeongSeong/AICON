import math
import timm
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50, ResNet50_Weights

from .ehr_transformer import EHRTransformer
from einops import rearrange, repeat
# VisionTransformer 클래스 상속
class CustomVisionTransformer(nn.Module):
    def __init__(self, model_name='vit_small_patch16_384', pretrained=True, in_chans=3, img_size=384, num_classes=1000, device='cuda'):
        super(CustomVisionTransformer, self).__init__()
        # timm 라이브러리를 사용해 pretrained 모델 불러오기
        self.model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            img_size=img_size,
            num_classes=num_classes
        ).to(device)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # 원래의 forward_features를 사용하되, self.norm_pre까지만 사용
        x = self.model.patch_embed(x)
        x = self.model._pos_embed(x)
        x = self.model.patch_drop(x)
        x = self.model.norm_pre(x)
        # # x를 self.norm_pre에서 추출한 후 반환
        # if self.model.grad_checkpointing and not torch.jit.is_scripting():
        #     x = checkpoint_seq(self.model.blocks, x)
        # else:
        #     x = self.model.blocks(x)
        # x = self.model.norm(x)
        return x
    
    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.model.pool(x)
        x = self.model.fc_norm(x)
        x = self.model.head_drop(x)
        return x if pre_logits else self.model.head(x)

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     # forward_features 메서드 호출
    #     x = self.forward_features(x)
    #     x = self.forward_head(x)
    #     # x를 그대로 반환
    #     return x

class DrFuseModel(nn.Module):
    def __init__(self, hidden_size, num_classes, ehr_dropout, ehr_n_layers, ehr_n_head, device='cuda',
                 cxr_model='swin_s', logit_average=False):
        super().__init__()
        self.num_classes = num_classes
        self.logit_average = logit_average
        self.ehr_model = EHRTransformer(input_size=14, # the number of clinical variables
                                        max_len=1, # maximum unmber of time events in ehr for an image
                                        num_classes=num_classes,
                                        d_model=hidden_size, n_head=ehr_n_head,
                                        n_layers_feat=1, n_layers_shared=ehr_n_layers,
                                        n_layers_distinct=ehr_n_layers,
                                        dropout=ehr_dropout)

        # feature_extractor_1 = timm.create_model(
        #     model_name='vit_small_patch16_384', # 'vit_small_patch16_384.augreg_in1k'
        #     pretrained=True,
        #     in_chans=3,
        #     img_size=384,
        #     num_classes=hidden_size
        # )
        custom1 = CustomVisionTransformer(
            model_name='vit_small_patch16_384', # 'vit_small_patch16_384.augreg_in1k'
            pretrained=True,
            in_chans=3,
            img_size=384,
            num_classes=hidden_size,
            device=device
        )
        # # # for name, mod in feature_extractor_1.named_modules():
        # # #     print(name)
        # # resnet = resnet50()
        # # self.cxr_model_feat = nn.Sequential(
        # #     resnet.conv1,
        # #     resnet.bn1,
        # #     resnet.relu, 
        # #     resnet.maxpool, 
        # # )

        # self.cxr_model_feat = nn.Sequential(
        #     feature_extractor_1.patch_embed,
        #     feature_extractor_1.pos_drop, # _pos_embed # pos_drop
        #     feature_extractor_1.patch_drop,
        #     feature_extractor_1.norm_pre,
        # )
        self.cxr_model_feat = custom1.forward_features

        # # # # resnet = resnet50()
        # # # # self.cxr_model_shared = nn.Sequential(
        # # # #     resnet.layer1,
        # # # #     resnet.layer2,
        # # # #     resnet.layer3,
        # # # #     resnet.layer4,
        # # # #     resnet.avgpool, 
        # # # #     nn.Flatten(), 
        # # # # )
        # # # # self.cxr_model_shared.fc = nn.Linear(in_features=resnet.fc.in_features, out_features=hidden_size)
        # # # self.cxr_model_shared = nn.Sequential(
        # # #     feature_extractor_1.blocks,
        # # #     feature_extractor_1.norm,
        # # #     #################################
        # # #     feature_extractor_1.fc_norm,
        # # #     feature_extractor_1.head_drop,
        # # #     feature_extractor_1.head, 
        # # #     )
        # # self.cxr_model_shared = custom.model.forward_head
        # self.cxr_model_shared = nn.Sequential(
        #     custom.model.blocks,
        #     custom.model.norm,
        #     #################################
        #     custom.model.fc_norm,
        #     custom.model.head_drop,
        #     custom.model.head
        # )
        custom2 = CustomVisionTransformer(
            model_name='vit_small_patch16_384', # 'vit_small_patch16_384.augreg_in1k'
            pretrained=True,
            in_chans=3,
            img_size=384,
            num_classes=hidden_size,
            device=device
        )
        self.cxr_model_shared = nn.Sequential(
            custom2.model.blocks,
            custom2.model.norm,
        )
        self.cxr_model_shared2 = custom2.forward_head

        # # # # resnet = resnet50()
        # # # # self.cxr_model_spec = nn.Sequential(
        # # # #     resnet.layer1,
        # # # #     resnet.layer2,
        # # # #     resnet.layer3,
        # # # #     resnet.layer4,
        # # # #     resnet.avgpool, 
        # # # #     nn.Flatten(), 
        # # # # )
        # # # # self.cxr_model_spec.fc = nn.Linear(in_features=resnet.fc.in_features, out_features=hidden_size)
        # # # self.cxr_model_spec = nn.Sequential(
        # # #     feature_extractor_1.blocks,
        # # #     feature_extractor_1.norm,
        # # #     #################################
        # # #     feature_extractor_1.fc_norm,
        # # #     feature_extractor_1.head_drop,
        # # #     feature_extractor_1.head,
        # # #     )
        # # self.cxr_model_spec = custom.model.forward_head
        # self.cxr_model_spec = nn.Sequential(
        #     custom.model.blocks,
        #     custom.model.norm,
        #     #################################
        #     custom.model.fc_norm,
        #     custom.model.head_drop,
        #     custom.model.head
        # )
        custom3 = CustomVisionTransformer(
            model_name='vit_small_patch16_384', # 'vit_small_patch16_384.augreg_in1k'
            pretrained=True,
            in_chans=3,
            img_size=384,
            num_classes=hidden_size,
            device=device
        )
        self.cxr_model_spec = nn.Sequential(
            custom3.model.blocks,
            custom3.model.norm,
        )
        self.cxr_model_spec2 = custom3.forward_head

        self.shared_project = nn.Sequential(
            nn.Linear(hidden_size, hidden_size*2),
            nn.ReLU(),
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.ehr_model_linear = nn.Linear(in_features=hidden_size, out_features=num_classes)
        self.cxr_model_linear = nn.Linear(in_features=hidden_size, out_features=num_classes)
        self.fuse_model_shared = nn.Linear(in_features=hidden_size, out_features=num_classes)

        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, 1)
        )
        self.attn_proj = nn.Linear(hidden_size, (2+num_classes)*hidden_size)
        self.final_pred_fc = nn.Linear(hidden_size, num_classes)

    def forward(self, data): 
        x, img, seq_lengths, pairs = data
        feat_ehr_shared, feat_ehr_distinct, pred_ehr = self.ehr_model(x, seq_lengths)
        feat_cxr = self.cxr_model_feat(img)
        # print('feat_cxr', feat_cxr.shape) # (batch, 577, 384)
        feat_cxr_shared = self.cxr_model_shared(feat_cxr)
        feat_cxr_shared = self.cxr_model_shared2(feat_cxr_shared)
        feat_cxr_distinct = self.cxr_model_spec(feat_cxr)
        feat_cxr_distinct = self.cxr_model_spec2(feat_cxr_distinct)
        # print('feat_cxr_distinct', feat_cxr_distinct.shape) # (batch, hidden_size) # (batch, 576, hidden_size)
        # get shared feature
        pred_cxr = self.cxr_model_linear(feat_cxr_distinct).sigmoid()
        # print('pred_cxr', pred_cxr.shape) # (batch, num_classes)

        
        feat_cxr_shared = self.shared_project(feat_cxr_shared)
        feat_ehr_shared = self.shared_project(feat_ehr_shared)
        # feat_ehr_shared = repeat(self.shared_project(feat_ehr_shared).unsqueeze(1), 'b 1 d -> b f d', f=feat_cxr_shared.shape[1])
        # print('h1:', feat_ehr_shared.shape, ', h2:', feat_cxr_shared.shape) # h1: (batch, hidden_size) , h2: (batch, hidden_size)
        pairs = pairs.unsqueeze(1)
        # pairs = pairs.unsqueeze(1).unsqueeze(1)
        
        h1 = feat_ehr_shared
        h2 = feat_cxr_shared
        term1 = torch.stack([h1+h2, h1+h2, h1, h2], dim=2)
        term2 = torch.stack([torch.zeros_like(h1), torch.zeros_like(h1), h1, h2], dim=2)
        feat_avg_shared = torch.logsumexp(term1, dim=2) - torch.logsumexp(term2, dim=2)
        # print(feat_ehr_shared.shape, feat_cxr_shared.shape, feat_avg_shared.shape, pairs.shape) # (batch, hidden_size) x 3, (batch, 1)
        feat_avg_shared = pairs * feat_avg_shared + (1 - pairs) * feat_ehr_shared 
        pred_shared = self.fuse_model_shared(feat_avg_shared).sigmoid() # (batch, 2)

        # Disease-wise Attention은 binary classification에서는 필요하지 않은 것 같다
        # feat_ehr_distinct = repeat(feat_ehr_distinct.unsqueeze(1), 'b 1 d -> b f d', f = feat_avg_shared.shape[1])
        # print(pred_shared.shape, feat_ehr_distinct.shape, feat_avg_shared.shape, feat_cxr_distinct.shape) # (batch, hidden_size)
        # Disease-wise Attention
        attn_input = torch.stack([feat_ehr_distinct, feat_avg_shared, feat_cxr_distinct], dim=1)
        # print('attn_input', attn_input.shape) # (batch, 3, hidden_size)
        qkvs = self.attn_proj(attn_input)
        q, v, *k = qkvs.chunk(2+self.num_classes, dim=-1)

        # compute query vector
        q_mean = pairs * q.mean(dim=1) + (1-pairs) * q[:, :-1].mean(dim=1)

        # compute attention weighting
        ks = torch.stack(k, dim=1)
        attn_logits = torch.einsum('bd,bnkd->bnk', q_mean, ks)
        attn_logits = attn_logits / math.sqrt(q.shape[-1])

        # filter out non-paired
        attn_mask = torch.ones_like(attn_logits)
        attn_mask[pairs.squeeze()==0, :, -1] = 0
        attn_logits = attn_logits.masked_fill(attn_mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_logits, dim=-1)

        # get final class-specific representation and prediction
        feat_final = torch.matmul(attn_weights, v)
        pred_final = self.final_pred_fc(feat_final)
        pred_final = torch.diagonal(pred_final, dim1=1, dim2=2).sigmoid()
        # print('pred_final', pred_final.shape) # (batch, num_classes)
        outputs = {
            'feat_cxr': feat_cxr,
            'feat_ehr_shared': feat_ehr_shared,
            'feat_cxr_shared': feat_cxr_shared,
            'feat_ehr_distinct': feat_ehr_distinct,
            'feat_cxr_distinct': feat_cxr_distinct,
            'feat_final': feat_final,
            'pred_final': pred_final,
            'pred_shared': pred_shared,
            'pred_ehr': pred_ehr,
            'pred_cxr': pred_cxr,
            'attn_weights': attn_weights,
        }

        return outputs
