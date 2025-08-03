#!/usr/bin/env python3
"""
检查checkpoint文件的内容和结构
"""

import torch
import argparse

def check_checkpoint(ckpt_path):
    """检查checkpoint文件的详细信息"""
    
    print(f"🔍 Checking checkpoint: {ckpt_path}")
    print("=" * 60)
    
    try:
        # 加载checkpoint
        ckpt = torch.load(ckpt_path, map_location='cpu')
        
        print(f"📁 Checkpoint type: {type(ckpt)}")
        
        if isinstance(ckpt, dict):
            print(f"📋 Checkpoint keys: {list(ckpt.keys())}")
            
            # 检查每个键的内容
            for key, value in ckpt.items():
                if isinstance(value, dict):
                    print(f"\n🔑 Key '{key}' contains {len(value)} items:")
                    # 显示前几个参数名称
                    param_names = list(value.keys())[:10]
                    for name in param_names:
                        if hasattr(value[name], 'shape'):
                            print(f"  - {name}: {value[name].shape}")
                        else:
                            print(f"  - {name}: {type(value[name])}")
                    if len(value) > 10:
                        print(f"  ... and {len(value) - 10} more parameters")
                        
                elif hasattr(value, 'shape'):
                    print(f"\n🔑 Key '{key}': tensor with shape {value.shape}")
                else:
                    print(f"\n🔑 Key '{key}': {type(value)} - {value}")
        
        else:
            # 如果直接是state_dict
            print(f"📋 Direct state_dict with {len(ckpt)} parameters")
            param_names = list(ckpt.keys())[:20]
            for name in param_names:
                if hasattr(ckpt[name], 'shape'):
                    print(f"  - {name}: {ckpt[name].shape}")
            if len(ckpt) > 20:
                print(f"  ... and {len(ckpt) - 20} more parameters")
        
        # 检查模型层数
        if isinstance(ckpt, dict):
            state_dict = None
            if 'model_state_dict' in ckpt:
                state_dict = ckpt['model_state_dict']
            elif 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt
            
        if state_dict:
            # 查找temporal_decoder_blocks的最大索引
            max_layer = -1
            for key in state_dict.keys():
                if 'temporal_decoder_blocks.' in key:
                    try:
                        layer_idx = int(key.split('temporal_decoder_blocks.')[1].split('.')[0])
                        max_layer = max(max_layer, layer_idx)
                    except:
                        pass
            
            if max_layer >= 0:
                print(f"\n🏗️ Model architecture info:")
                print(f"  - Maximum layer index: {max_layer}")
                print(f"  - Total layers: {max_layer + 1}")
            
            # 检查condition encoder
            has_condition = any('condition_encoder' in key for key in state_dict.keys())
            print(f"  - Has condition encoder: {has_condition}")
            
            if has_condition:
                condition_keys = [key for key in state_dict.keys() if 'condition_encoder' in key]
                print(f"  - Condition encoder parameters: {len(condition_keys)}")
                for key in condition_keys[:5]:
                    print(f"    - {key}: {state_dict[key].shape}")
        
        print("\n✅ Checkpoint analysis completed!")
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True, help='Path to checkpoint file')
    args = parser.parse_args()
    
    check_checkpoint(args.ckpt)

if __name__ == '__main__':
    main()
