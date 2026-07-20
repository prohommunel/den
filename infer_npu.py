import os
import glob
import argparse
import numpy as np
import math
import json
from PIL import Image, ImageFile
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

# ==========================================
# 🌟 NPU 与组装加载核心库
# ==========================================
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except ImportError:
    pass

from transformers import AutoModel
from peft import PeftModel

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ==========================================
# 1. 模型架构定义（严格对齐 GAPL 训练代码）
# ==========================================
class Norm(nn.Module):
    def __init__(self, mode='clip'):
        super().__init__()
        self.mode = mode
        if mode == 'clip':
            self.register_buffer('mean', torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))
        else:
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std

class DINO(nn.Module):
    def __init__(self, dinov3_path):
        super(DINO, self).__init__()
        print(f"Loading Backbone from: {dinov3_path}")
        self.dino = AutoModel.from_pretrained(dinov3_path, weights_only=False, trust_remote_code=True)
        self.dino.requires_grad_(False)

    def forward(self, x):
        outputs = self.dino(pixel_values=x)
        last_hidden_state = outputs[0]
        # 只需取出全局 CLS token
        feat_cls = last_hidden_state[:, 0]
        return feat_cls

class GAPL_CrossAttentionHead(nn.Module):
    def __init__(self, feature_dim=1280, proj_dim=256, num_prototypes=64):
        super().__init__()
        self.proj_dim = proj_dim
        
        # 降维投影
        self.f_proj = nn.Linear(feature_dim, proj_dim)

        # 交叉注意力 Query, Key, Value
        self.W_q = nn.Linear(proj_dim, proj_dim, bias=False)
        self.W_k = nn.Linear(proj_dim, proj_dim, bias=False)
        self.W_v = nn.Linear(proj_dim, proj_dim, bias=False)

        # 最终分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(proj_dim, 2)
        )

        # 注册 Prototype Buffer
        self.register_buffer('prototypes', torch.zeros(num_prototypes, proj_dim))

    def forward(self, feat_cls):
        f = self.f_proj(feat_cls)
        Q = self.W_q(f).unsqueeze(1)               # [B, 1, D']
        K = self.W_k(self.prototypes)              # [N_proto, D']
        V = self.W_v(self.prototypes)              # [N_proto, D']

        attn_scores = torch.matmul(Q, K.transpose(0, 1)) / math.sqrt(self.proj_dim)  # [B, 1, N_proto]
        attn_weights = torch.softmax(attn_scores, dim=-1)
        f_bar = torch.matmul(attn_weights, V).squeeze(1)                             # [B, D']

        logits = self.classifier(f_bar)
        return logits

class GAPL_Detector(nn.Module):
    def __init__(self, dino_path, feature_dim=1280, proj_dim=256, num_prototypes=64):
        super().__init__()
        self.norm = Norm(mode='imagenet')
        self.backbone = DINO(dino_path) 
        self.detector = GAPL_CrossAttentionHead(feature_dim, proj_dim, num_prototypes)

    def forward(self, x):
        x = self.norm(x)
        feat_cls = self.backbone(x)
        logits = self.detector(feat_cls)
        return logits

# ==========================================
# 2. 预处理与全路径扫描数据集
# ==========================================
class DynamicScaleCenterCrop:
    def __init__(self, crop_size=224):
        self.crop_size = crop_size
    def __call__(self, img):
        short_edge = min(img.size)
        if short_edge < 256:
            img = TF.resize(img, 256, interpolation=TF.InterpolationMode.BICUBIC)
        elif short_edge > 512:
            img = TF.resize(img, 512, interpolation=TF.InterpolationMode.BICUBIC)
        img = TF.center_crop(img, [self.crop_size, self.crop_size])
        return TF.to_tensor(img)

class InferenceDataset(Dataset):
    def __init__(self, folder_path, transform):
        self.transform = transform
        self.image_paths = []
        
        valid_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff'}
        all_files = glob.glob(os.path.join(folder_path, "**", "*.*"), recursive=True)
        
        for p in all_files:
            if os.path.splitext(p)[-1].lower() in valid_exts:
                self.image_paths.append(p)
                
        print(f"🔍 扫描完成！在目标文件夹中共找到 {len(self.image_paths)} 张待预测图片。")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert('RGB')
            return self.transform(img), path
        except Exception as e:
            print(f"⚠️ 警告: 无法读取图片 {path}, 错误信息: {e}")
            return torch.zeros(3, 224, 224), f"ERROR_LOADING:{path}"

# ==========================================
# 3. 主推理逻辑
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="NPU Inference Script for GAPL Enhanced Detector")
    parser.add_argument('--input_dir', type=str, required=True, help='待检测的图片文件夹路径')
    parser.add_argument('--backbone_path', type=str, default='/root/autodl-tmp/cache/facebook/dinov3-vith16plus-pretrain-lvd1689m')
    parser.add_argument('--weight_dir', type=str, default='./weight52', help='保存权重的文件夹')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8, help="Dataloader的线程数")
    parser.add_argument('--num_prototypes', type=int, default=64, help="鉴伪标尺的数量")
    parser.add_argument('--proj_dim', type=int, default=256, help="交叉注意力投影维度")
    args = parser.parse_args()

    # 🌟 NPU 设备指定
    device = torch.device('npu' if hasattr(torch, 'npu') and torch.npu.is_available() else 'cpu')
    print(f"🚀 初始化 GAPL 网络模型，运行设备: {device}")
    
    # ---------------------------
    # A. 恢复 GAPL 模型和交叉注意力头
    # ---------------------------
    model = GAPL_Detector(
        dino_path=args.backbone_path, 
        feature_dim=1280, 
        proj_dim=args.proj_dim, 
        num_prototypes=args.num_prototypes
    )
    
    lora_path = os.path.join(args.weight_dir, "lora_adapter")
    head_path = os.path.join(args.weight_dir, "custom_head.pth")
    
    if not (os.path.exists(lora_path) and os.path.exists(head_path)):
        raise FileNotFoundError(f"❌ 未在 {args.weight_dir} 下找到完整的适配器参数，请确保训练已成功完成。")
        
    print(f"📦 正在载入增量权重: {args.weight_dir}")
    model.backbone.dino = PeftModel.from_pretrained(model.backbone.dino, lora_path)
    
    # 恢复分类头与原型 (Prototypes)
    custom_state = torch.load(head_path, map_location='cpu')
    model.norm.load_state_dict(custom_state['norm'])
    model.detector.load_state_dict(custom_state['detector'])
    
    model = model.to(device)
    model.eval()

    # ---------------------------
    # B. 构建数据加载器
    # ---------------------------
    transform = DynamicScaleCenterCrop(224)
    dataset = InferenceDataset(args.input_dir, transform)
    
    if len(dataset) == 0:
        print("⚠️ 该目录下没有找到有效的图片。")
        return
        
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # 标签映射对照
    id_to_class = {1: "Real", 0: "Fake"}
    
    count_real = 0
    count_fake = 0
    
    print("\n🏃 开始执行 GAPL NPU 原型前向推理...\n")
    print("-" * 80)
    print(f"{'图片路径':<50} | {'预测标签':<10} | {'置信度 (Confidence)'}")
    print("-" * 80)

    # 🌟 新增：将结果以 JSON Lines 格式写入文件
    output_json_path = 'qiankongjian_result.json'
    with open(output_json_path, 'w', encoding='utf-8') as f_out:
        with torch.no_grad():
            for imgs, paths in tqdm(loader, desc="[Inference]", leave=False):
                valid_mask = [not p.startswith("ERROR_LOADING") for p in paths]
                if not any(valid_mask): continue
                    
                # 🌟 数据推向 NPU 并开启 non_blocking 加速
                imgs = imgs[valid_mask].to(device, non_blocking=True)
                filtered_paths = [paths[i] for i, v in enumerate(valid_mask) if v]
                
                # 🌟 开启 NPU 的混合精度自动铸造
                with torch.amp.autocast('npu'):
                    logits = model(imgs)
                    
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                
                for idx, path in enumerate(filtered_paths):
                    pred_id = preds[idx]
                    pred_label = id_to_class[pred_id]
                    confidence = probs[idx][pred_id] 
                    
                    if pred_id == 1:
                        count_real += 1
                        is_generated = "0"  # Real 代表真实，不是生成的 (0)
                    else:
                        count_fake += 1
                        is_generated = "1"  # Fake 代表伪造，是生成的 (1)
                        
                    display_path = os.path.relpath(path, args.input_dir)
                    print(f"{display_path:<50} | {pred_label:<10} | {confidence * 100:.2f}%")
                    
                    # 提取文件名并保存为指定格式
                    image_name = os.path.basename(path)
                    result_dict = {"image_name": image_name, "is_generated": is_generated}
                    f_out.write(json.dumps(result_dict) + '\n')

    # ---------------------------
    # C. 输出总体占比统计
    # ---------------------------
    total_predicted = count_real + count_fake
    print("-" * 80)
    print("\n" + "="*50)
    print(f"📊 目标检测文件夹: {args.input_dir}")
    print(f"总有效图片预测数: {total_predicted} 张")
    print("-" * 50)
    if total_predicted > 0:
        print(f"🟢 真实图片 (Real) 数量: {count_real:<5} 张  |  占比: {(count_real/total_predicted)*100:.2f}%")
        print(f"🔴 伪造图片 (Fake) 数量: {count_fake:<5} 张  |  占比: {(count_fake/total_predicted)*100:.2f}%")
    else:
        print("未成功预测任何图片。")
    print("="*50 + "\n")

if __name__ == '__main__':
    main()