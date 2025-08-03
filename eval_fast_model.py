#!/usr/bin/env python3
"""
评估快速训练模型的脚本
"""

import argparse
import sys
import os
import torch
import numpy as np
from scipy import io

sys.path.append(os.getcwd())
from utils import create_logger, seed_set
from utils.script import *
from config import Config, update_config
from utils.evaluation import compute_stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='h36m_fast', help='使用与训练时相同的配置')
    parser.add_argument('--mode', default='eval', help='eval mode')
    parser.add_argument('--ckpt', required=True, help='checkpoint path')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str,
                        default=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    parser.add_argument('--multimodal_threshold', type=float, default=0.5)
    parser.add_argument('--multimodal_th_high', type=float, default=0.1)
    args = parser.parse_args()

    """setup"""
    seed_set(args.seed)

    # 使用与训练时相同的配置
    cfg = Config(f'{args.cfg}', test=True)
    cfg = update_config(cfg, vars(args))

    dataset, dataset_multi_test = dataset_split(cfg)

    # 加载condition数据（与训练时保持一致）
    print("🔄 Loading condition data for evaluation...")
    
    data_path = 'data/train'
    mat_files = [os.path.join(data_path, file) for file in os.listdir(data_path) if file.endswith('.mat')]
    all_data = []

    for mat_file in mat_files:
        data = io.loadmat(mat_file)['x']
        all_data.append(data)

    condition = np.concatenate(all_data, axis=0)

    # 加载保存的标准化参数
    try:
        condition_stats = np.load('condition_stats.npy', allow_pickle=True).item()
        condition_mean = condition_stats['mean']
        condition_std = condition_stats['std']
        print("✅ Loaded saved condition statistics")
    except:
        print("⚠️ No saved condition statistics found, computing new ones...")
        condition_mean = np.mean(condition, axis=0, keepdims=True)
        condition_std = np.std(condition, axis=0, keepdims=True)
        condition_std = np.where(condition_std == 0, 1.0, condition_std)

    # 标准化
    condition_normalized = (condition - condition_mean) / condition_std
    condition = torch.tensor(condition_normalized, dtype=torch.float32)

    print(f"📊 Condition shape: {condition.shape}")

    """logger"""
    logger = create_logger('eval_log.txt')
    
    """model"""
    print("🏗️ Creating model with fast configuration...")
    model, diffusion = create_model_and_diffusion(cfg, condition)

    logger.info(">>> total params: {:.2f}M".format(
        sum(p.numel() for p in list(model.parameters())) / 1000000.0))

    # 加载checkpoint
    print(f"📂 Loading checkpoint: {args.ckpt}")
    try:
        ckpt = torch.load(args.ckpt, map_location='cpu')
        
        # 如果checkpoint包含多个键，尝试找到模型状态
        if isinstance(ckpt, dict):
            if 'model_state_dict' in ckpt:
                model_state = ckpt['model_state_dict']
            elif 'state_dict' in ckpt:
                model_state = ckpt['state_dict']
            else:
                model_state = ckpt
        else:
            model_state = ckpt
            
        model.load_state_dict(model_state)
        print("✅ Model loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        print("🔍 Checkpoint keys:", list(ckpt.keys()) if isinstance(ckpt, dict) else "Not a dict")
        return

    # 移动到GPU
    condition = condition.clone().detach().to(cfg.device)
    
    # 准备评估数据
    print("📊 Preparing evaluation dataset...")
    multimodal_dict = get_multimodal_gt_full(logger, dataset_multi_test, args, cfg)
    
    # 开始评估
    print("🚀 Starting evaluation...")
    logger.info("=== Evaluation Results ===")
    logger.info(f"Model: {args.ckpt}")
    logger.info(f"Configuration: {args.cfg}")
    logger.info(f"Model layers: {cfg.num_layers}")
    logger.info(f"Latent dims: {cfg.latent_dims}")
    logger.info(f"Condition shape: {condition.shape}")
    
    compute_stats(diffusion, multimodal_dict, model, logger, cfg, condition=condition)
    
    print("✅ Evaluation completed! Check eval_log.txt for detailed results.")

if __name__ == '__main__':
    main()
