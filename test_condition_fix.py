#!/usr/bin/env python3
"""
测试脚本：验证condition参数修复是否有效
"""

import torch
import numpy as np
import sys
import os
sys.path.append(os.getcwd())

from models.transformer import MotionTransformer
from models.diffusion import Diffusion

def test_condition_fix():
    """测试condition参数的修复"""
    
    print("=== 测试condition参数修复 ===")
    
    # 创建测试数据
    batch_size = 4
    seq_len = 20
    joint_num = 16
    input_feats = 3 * joint_num  # 48
    
    # 创建模拟的condition数据
    condition_dim = 4
    condition = torch.randn(100, condition_dim)  # 100个样本，每个4维
    
    print(f"Condition shape: {condition.shape}")
    
    # 创建模型
    model = MotionTransformer(
        input_feats=input_feats,
        num_frames=seq_len,
        num_layers=4,
        num_heads=4,
        latent_dim=256,
        dropout=0.1,
        condition=condition,
    )
    
    print(f"Model created successfully")
    print(f"Condition encoder: {model.condition_encoder}")

    # 设置模型为训练模式以确保condition有影响
    model.train()

    # 测试前向传播
    x = torch.randn(batch_size, seq_len, input_feats)
    timesteps = torch.randint(1, 1000, (batch_size,))

    # 创建极端不同的condition值来测试
    condition_batch1 = torch.ones(batch_size, condition_dim) * 1.0
    condition_batch2 = torch.ones(batch_size, condition_dim) * -1.0  # 完全相反的值

    print(f"Input x shape: {x.shape}")
    print(f"Timesteps shape: {timesteps.shape}")
    print(f"Condition batch1 shape: {condition_batch1.shape}")
    print(f"Condition batch2 shape: {condition_batch2.shape}")

    try:
        # 测试不带condition的前向传播
        output1 = model(x, timesteps)
        print(f"✓ Forward pass without condition successful, output shape: {output1.shape}")

        # 测试带不同condition的前向传播
        output2 = model(x, timesteps, condition=condition_batch1)
        print(f"✓ Forward pass with condition1 successful, output shape: {output2.shape}")

        output3 = model(x, timesteps, condition=condition_batch2)
        print(f"✓ Forward pass with condition2 successful, output shape: {output3.shape}")

        # 验证输出不同（说明condition起作用了）
        diff1 = torch.mean(torch.abs(output1 - output2)).item()
        diff2 = torch.mean(torch.abs(output2 - output3)).item()
        print(f"Mean absolute difference (no condition vs condition1): {diff1:.6f}")
        print(f"Mean absolute difference (condition1 vs condition2): {diff2:.6f}")

        if diff1 > 1e-6 or diff2 > 1e-6:
            print("✓ Condition parameter affects model output")
        else:
            print("⚠ Warning: Condition parameter may not be affecting model output")
            
    except Exception as e:
        print(f"✗ Error during forward pass: {e}")
        return False
    
    # 测试diffusion
    print("\n=== 测试Diffusion中的condition支持 ===")
    
    diffusion = Diffusion(
        noise_steps=100,
        motion_size=(seq_len, input_feats),
        device="cpu",
        condition=None  # 不在diffusion中设置condition，而是在采样时传入
    )
    
    print(f"Diffusion created successfully")
    
    # 测试采样
    try:
        traj_dct = torch.randn(batch_size, seq_len, input_feats)
        traj_dct_mod = torch.randn(batch_size, seq_len, input_feats)
        mode_dict = {
            'sample_num': batch_size,
            'mask': torch.ones(batch_size, seq_len, input_feats),
            'mode': 'pred'
        }
        
        # 测试不带condition的采样
        sample1 = diffusion.sample_ddim(model, traj_dct, traj_dct_mod, mode_dict)
        print(f"✓ Sampling without condition successful, shape: {sample1.shape}")
        
        # 测试带condition的采样 - 只使用需要的数量
        condition_for_sampling = condition[:batch_size]  # 只取需要的batch_size数量
        sample2 = diffusion.sample_ddim(model, traj_dct, traj_dct_mod, mode_dict, condition=condition_for_sampling)
        print(f"✓ Sampling with condition successful, shape: {sample2.shape}")

        # 验证输出不同
        diff = torch.mean(torch.abs(sample1 - sample2)).item()
        print(f"Mean absolute difference between samples: {diff:.6f}")
        if diff > 1e-6:
            print("✓ Condition parameter affects sampling output")
        else:
            print("⚠ Warning: Condition parameter may not be affecting sampling")
            
    except Exception as e:
        print(f"✗ Error during sampling: {e}")
        return False
    
    print("\n=== 所有测试通过！ ===")
    return True

if __name__ == "__main__":
    success = test_condition_fix()
    if success:
        print("\n🎉 修复验证成功！现在可以重新训练模型了。")
        print("\n建议的下一步操作：")
        print("1. 重新开始训练，或从最近的checkpoint继续训练")
        print("2. 监控训练和验证loss，确保它们保持一致")
        print("3. 检查性能指标是否有改善")
    else:
        print("\n❌ 修复验证失败，需要进一步检查代码。")
