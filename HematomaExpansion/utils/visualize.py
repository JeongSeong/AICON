import os
import torch
import matplotlib.pyplot as plt
from monai.visualize import GradCAM
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import roc_curve, auc
from monai.visualize.utils import blend_images

class shapModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # self.model.eval()
        
    def __call__(self, x):
        # with torch.no_grad():
            # output = self.model(x)[-1] # torch.from_numpy(x)
            # output.requires_grad = True
        return self.model(x)[-1].detach()  # output	# .cpu().numpy() 
# class gradCAM:
#     def __init__(self, model, layer):
#         self.model = model
#         self.layer = layer
#         self.attentions = []
#         self.gradients = []

#         self._register_hooks()

#     def _register_hooks(self):
#         def hook_fn(module, input, output):
#             output.register
#             with torch.no_grad():
# def save_attention_maps(model, dataloader, savedir, device):
#     # print the model architecture
#     print([k for k, v in model.named_parameters()])
#     # input the layer name for visualization
#     layername = input("Enter the layer name for visualization: ")
#     gradcam = GradCAM(nn_module=model, target_layers=layername)
#     for idx, d in enumerate(dataloader):
#         (patient_id, inputs, clinics, labels) = (
#             d["id"],
#             d["image"],
#             d["clinic"],
#             d["label"],
#         )
#         inputs = inputs.to(device)
#         attention_map = gradcam(x=inputs)
#         attention_map = attention_map[0, 0].cpu().numpy()
#         inputs = inputs[0, :, :, :, :].cpu().numpy()
#         attention_map = attention_map[np.newaxis, :, :, :]
#         blended_image = blend_images(
#             torch.tensor(inputs), torch.tensor(attention_map), alpha=0.5
#         )
#         print(
#             f"inputs shape: {inputs.shape} | attention shape: {attention_map.shape} | blended shape: {blended_image.shape}"
#         )
#         for i in range(10, 54, 3):  # Iterate through the slices
#             # Use MONAI's blend_images to overlay attention map
#             patient_dir = os.path.join(savedir, str(patient_id[0]))
#             os.makedirs(patient_dir, exist_ok=True)

#             plt.imshow(torch.moveaxis(blended_image[:, :, :, i], 0, -1))
#             plt.show()
#             plt.savefig(os.path.join(patient_dir, f"attention_map_slice_{i}.png"))
#             plt.close()
class ViTAttentionMap: # vit_small_patch16_384
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
            blk.attn.register_forward_hook(hook_fn)

    def get_attention_maps(self, input_tensor):
        # https://github.com/jacobgil/vit-explain/blob/main/vit_rollout.py
        self.attention_maps = []
        with torch.no_grad():
            # Hook에서 attention maps를 추출합니다.
            _ = self.model(input_tensor)
        return self.attention_maps

def save_auc_plot(y_true, y_pred, savedir):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:0.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(savedir, "auc_plot.png"))
    plt.close()


def save_calibration_plot(y_true, y_pred, savedir):
    # prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10)
    CalibrationDisplay.from_predictions(y_true, y_pred, n_bins=10)
    # disp.plot()
    plt.savefig(os.path.join(savedir, "calibration_plot.png"))
    plt.close()
