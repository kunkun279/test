#!/usr/bin/env python3
"""
调试脚本：专门测试condition是否真的在影响模型
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
sys.path.append(os.getcwd())

from models.transformer import MotionTransformer

def test_condition_step_by_step():
    """逐步测试condition的影响"""
    
    print("=== 逐步调试condition影响 ===")
    
    # 创建简单的测试数据
    batch_size = 2
    seq_len = 10
    input_feats = 48
    condition_dim = 4
    
    # 创建极端不同的condition
    condition_data = torch.randn(100, condition_dim)
    condition1 = torch.ones(batch_size, condition_dim) * 5.0   # 大正值
    condition2 = torch.ones(batch_size, condition_dim) * -5.0  # 大负值
    
    print(f"Condition1: {condition1[0]}")
    print(f"Condition2: {condition2[0]}")
    
    # 创建模型
    model = MotionTransformer(
        input_feats=input_feats,
        num_frames=seq_len,
        num_layers=2,  # 减少层数，简化测试
        num_heads=2,
        latent_dim=128,  # 减少维度
        dropout=0.0,  # 关闭dropout
        condition=condition_data,
    )
    
    print(f"模型创建成功")
    print(f"Condition encoder: {model.condition_encoder}")
    
    # 设置为训练模式
    model.train()
    
    # 创建相同的输入
    x = torch.randn(batch_size, seq_len, input_feats)
    timesteps = torch.ones(batch_size, dtype=torch.long) * 500  # 固定timestep
    
    print(f"输入x: {x.shape}")
    print(f"Timesteps: {timesteps}")
    
    # 测试condition encoder单独的输出
    print("\n=== 测试condition encoder ===")
    with torch.no_grad():
        cond_proj1 = model.condition_encoder(condition1)
        cond_proj2 = model.condition_encoder(condition2)
        
        print(f"Condition projection 1 mean: {torch.mean(cond_proj1):.6f}")
        print(f"Condition projection 2 mean: {torch.mean(cond_proj2):.6f}")
        print(f"Condition projection difference: {torch.mean(torch.abs(cond_proj1 - cond_proj2)):.6f}")
    
    # 测试完整的前向传播
    print("\n=== 测试完整前向传播 ===")
    
    # 启用调试信息
    original_forward = model.forward
    
    def debug_forward(self, x, timesteps, mod=None, condition=None):
        B, T = x.shape[0], x.shape[1]
        
        # Time embedding
        from models.transformer import timestep_embedding
        emb = self.time_embed(timestep_embedding(timesteps, self.latent_dim))
        print(f"Time embedding mean: {torch.mean(emb):.6f}")
        
        # Mod embedding
        if mod is not None:
            mod_proj = self.cond_embed(mod.reshape(B, -1))
            emb = emb + mod_proj
            print(f"After mod, embedding mean: {torch.mean(emb):.6f}")
        
        # Condition embedding
        if condition is not None and self.condition_encoder is not None:
            cond_proj = self.condition_encoder(condition)
            print(f"Condition projection mean: {torch.mean(cond_proj):.6f}")
            condition_weight = 1.0
            emb_before = emb.clone()
            emb = emb + condition_weight * cond_proj
            print(f"Embedding change due to condition: {torch.mean(torch.abs(emb - emb_before)):.6f}")
            print(f"After condition, embedding mean: {torch.mean(emb):.6f}")
        
        # Joint embedding
        h = self.joint_embed(x)
        h = h + self.sequence_embedding.unsqueeze(0)[:, :T, :]
        print(f"Joint embedding mean: {torch.mean(h):.6f}")
        
        # Transformer blocks
        i = 0
        prelist = []
        for module in self.temporal_decoder_blocks:
            if i < (self.num_layers // 2):
                prelist.append(h)
                h = module(h, emb)
            elif i >= (self.num_layers // 2):
                h = module(h, emb)
                h += prelist[-1]
                prelist.pop()
            i += 1
        
        print(f"After transformer blocks mean: {torch.mean(h):.6f}")
        
        # Output
        output = self.out(h).view(B, T, -1).contiguous()
        print(f"Final output mean: {torch.mean(output):.6f}")
        
        return output
    
    # 替换forward方法进行调试
    import types
    model.forward = types.MethodType(debug_forward, model)
    
    print("\n--- 测试condition1 ---")
    with torch.no_grad():
        output1 = model(x, timesteps, condition=condition1)
    
    print("\n--- 测试condition2 ---")
    with torch.no_grad():
        output2 = model(x, timesteps, condition=condition2)
    
    print("\n=== 最终结果比较 ===")
    diff = torch.mean(torch.abs(output1 - output2)).item()
    print(f"输出差异: {diff:.6f}")
    
    if diff > 1e-4:
        print("✓ Condition成功影响了模型输出！")
        return True
    else:
        print("✗ Condition没有明显影响模型输出")
        return False

def test_simple_condition_network():
    """测试一个简化的condition网络"""
    print("\n=== 测试简化的condition网络 ===")
    
    # 创建一个简单的测试网络
    condition_dim = 4
    embed_dim = 128
    
    condition_encoder = nn.Sequential(
        nn.Linear(condition_dim, embed_dim // 2),
        nn.ReLU(),
        nn.Linear(embed_dim // 2, embed_dim),
    )
    
    # 测试数据
    condition1 = torch.ones(1, condition_dim) * 2.0
    condition2 = torch.ones(1, condition_dim) * -2.0
    
    with torch.no_grad():
        out1 = condition_encoder(condition1)
        out2 = condition_encoder(condition2)
        
        print(f"简单网络输出1 mean: {torch.mean(out1):.6f}")
        print(f"简单网络输出2 mean: {torch.mean(out2):.6f}")
        print(f"简单网络输出差异: {torch.mean(torch.abs(out1 - out2)):.6f}")
    
    return torch.mean(torch.abs(out1 - out2)).item() > 1e-4

if __name__ == "__main__":
    print("开始调试condition影响...")
    
    # 测试简单网络
    simple_works = test_simple_condition_network()
    print(f"简单网络测试: {'✓ 通过' if simple_works else '✗ 失败'}")
    
    # 测试完整模型
    full_works = test_condition_step_by_step()
    print(f"完整模型测试: {'✓ 通过' if full_works else '✗ 失败'}")
    
    if simple_works and full_works:
        print("\n🎉 Condition功能正常工作！")
    elif simple_works and not full_works:
        print("\n⚠ 简单网络工作，但完整模型有问题，需要进一步调试")
    else:
        print("\n❌ 基础网络就有问题，需要检查实现")
