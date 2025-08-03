#!/usr/bin/env python3
"""
Complete Checkpoint Analysis Tool

This script provides comprehensive analysis of checkpoint architecture
to determine all necessary parameters for model compatibility.
"""

import torch
import os
import sys
import argparse

sys.path.append(os.getcwd())


def analyze_checkpoint_complete(ckpt_path):
    """Complete analysis of checkpoint architecture"""
    
    print(f"🔍 Complete Checkpoint Analysis: {ckpt_path}")
    print("=" * 70)
    
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        
        # Get state dict
        state_dict = None
        if isinstance(ckpt, dict):
            for key in ['model_state_dict', 'state_dict', 'model', 'net']:
                if key in ckpt:
                    state_dict = ckpt[key]
                    break
            if state_dict is None:
                state_dict = ckpt
        else:
            state_dict = ckpt
        
        print(f"📊 Checkpoint contains {len(state_dict)} parameters")
        
        # Analyze architecture
        architecture = {}
        
        # 1. Sequence embedding (determines n_pre/num_frames)
        if 'sequence_embedding' in state_dict:
            seq_shape = state_dict['sequence_embedding'].shape
            architecture['num_frames'] = seq_shape[0]
            architecture['latent_dim_from_seq'] = seq_shape[1]
            print(f"✅ Sequence embedding: {seq_shape} -> num_frames={seq_shape[0]}")
        
        # 2. Joint embedding (determines input_feats)
        if 'joint_embed.weight' in state_dict:
            joint_shape = state_dict['joint_embed.weight'].shape
            architecture['input_feats'] = joint_shape[1]
            architecture['latent_dim_from_joint'] = joint_shape[0]
            print(f"✅ Joint embedding: {joint_shape} -> input_feats={joint_shape[1]}")
        
        # 3. Output layer (confirms input_feats and latent_dim)
        if 'out.weight' in state_dict:
            out_shape = state_dict['out.weight'].shape
            architecture['output_feats'] = out_shape[0]
            architecture['latent_dim_from_out'] = out_shape[1]
            print(f"✅ Output layer: {out_shape} -> output_feats={out_shape[0]}")
        
        # 4. Time embedding (determines time_embed_dim)
        time_embed_keys = [k for k in state_dict.keys() if 'time_embed' in k and 'weight' in k]
        if time_embed_keys:
            first_time_key = time_embed_keys[0]
            time_shape = state_dict[first_time_key].shape
            if 'time_embed.0.weight' in state_dict:
                # First layer of time embedding
                time_input = state_dict['time_embed.0.weight'].shape[1]
                time_output = state_dict['time_embed.0.weight'].shape[0]
                architecture['time_embed_input'] = time_input
                architecture['time_embed_dim'] = time_output
                print(f"✅ Time embedding: input={time_input}, output={time_output}")
        
        # 5. Condition encoder analysis
        condition_keys = [k for k in state_dict.keys() if 'condition_encoder' in k]
        if condition_keys:
            print(f"✅ Condition encoder found with {len(condition_keys)} parameters")
            
            # Analyze condition encoder layers
            condition_layers = {}
            for key in condition_keys:
                parts = key.split('.')
                if len(parts) >= 3:
                    layer_idx = parts[1]
                    param_type = parts[2]
                    
                    if layer_idx not in condition_layers:
                        condition_layers[layer_idx] = {}
                    
                    condition_layers[layer_idx][param_type] = state_dict[key].shape
            
            print("   Condition encoder layers:")
            for layer_idx, params in sorted(condition_layers.items()):
                print(f"     Layer {layer_idx}:")
                for param_type, shape in params.items():
                    print(f"       {param_type}: {shape}")
            
            # Extract key dimensions
            if '0' in condition_layers and 'weight' in condition_layers['0']:
                cond_input = condition_layers['0']['weight'][1]
                cond_hidden = condition_layers['0']['weight'][0]
                architecture['condition_input_dim'] = cond_input
                architecture['condition_first_hidden'] = cond_hidden
                print(f"   -> Condition input: {cond_input}, first hidden: {cond_hidden}")
        
        # 6. Transformer layers analysis
        decoder_keys = [k for k in state_dict.keys() if 'temporal_decoder_blocks' in k]
        if decoder_keys:
            # Count layers
            layer_indices = set()
            for key in decoder_keys:
                parts = key.split('.')
                if len(parts) >= 2:
                    try:
                        layer_idx = int(parts[1])
                        layer_indices.add(layer_idx)
                    except ValueError:
                        pass
            
            architecture['num_layers'] = len(layer_indices)
            print(f"✅ Transformer layers: {len(layer_indices)} layers")
            
            # Analyze attention heads
            attn_keys = [k for k in decoder_keys if 'self_attn' in k and 'to_qkv.weight' in k]
            if attn_keys:
                qkv_shape = state_dict[attn_keys[0]].shape
                # QKV weight shape: [3 * latent_dim, latent_dim] for multi-head attention
                latent_from_qkv = qkv_shape[1]
                qkv_total = qkv_shape[0]
                
                # Typically num_heads = latent_dim / head_dim, where head_dim is usually 64
                if latent_from_qkv % 64 == 0:
                    estimated_heads = latent_from_qkv // 64
                elif latent_from_qkv % 32 == 0:
                    estimated_heads = latent_from_qkv // 32
                else:
                    estimated_heads = 8  # default
                
                architecture['num_heads_estimated'] = estimated_heads
                architecture['latent_dim_from_qkv'] = latent_from_qkv
                print(f"✅ Attention: QKV shape {qkv_shape} -> estimated {estimated_heads} heads")
        
        # 7. Consistency check
        print(f"\n📋 Architecture Summary:")
        latent_dims = []
        if 'latent_dim_from_seq' in architecture:
            latent_dims.append(architecture['latent_dim_from_seq'])
        if 'latent_dim_from_joint' in architecture:
            latent_dims.append(architecture['latent_dim_from_joint'])
        if 'latent_dim_from_out' in architecture:
            latent_dims.append(architecture['latent_dim_from_out'])
        if 'latent_dim_from_qkv' in architecture:
            latent_dims.append(architecture['latent_dim_from_qkv'])
        
        if len(set(latent_dims)) == 1:
            architecture['latent_dim'] = latent_dims[0]
            print(f"   ✅ Consistent latent_dim: {latent_dims[0]}")
        else:
            print(f"   ⚠️  Inconsistent latent_dims: {latent_dims}")
        
        # Print final architecture
        print(f"\n🎯 Final Architecture Parameters:")
        key_params = [
            'num_frames', 'input_feats', 'output_feats', 'latent_dim', 
            'num_layers', 'num_heads_estimated', 'condition_input_dim', 
            'condition_first_hidden'
        ]
        
        for param in key_params:
            if param in architecture:
                print(f"   {param}: {architecture[param]}")
        
        return architecture
        
    except Exception as e:
        print(f"❌ Error analyzing checkpoint: {e}")
        return None


def generate_compatible_model_code(architecture):
    """Generate compatible model code based on architecture analysis"""
    
    print(f"\n🔧 Generated Compatible Model Parameters:")
    print("=" * 50)
    
    code = f"""
# Compatible model parameters for this checkpoint:
model = CompatibleMotionTransformer(
    input_feats={architecture.get('input_feats', 48)},
    num_frames={architecture.get('num_frames', 20)},
    latent_dim={architecture.get('latent_dim', 384)},
    num_layers={architecture.get('num_layers', 6)},
    num_heads={architecture.get('num_heads_estimated', 8)},
    dropout=0.2,
    condition=condition
)

# Condition encoder architecture:
# Input dim: {architecture.get('condition_input_dim', 4)}
# First hidden: {architecture.get('condition_first_hidden', 192)}
# Output dim: {architecture.get('latent_dim', 384)}
"""
    
    print(code)
    return code


def main():
    parser = argparse.ArgumentParser(description='Complete checkpoint analysis')
    parser.add_argument('--ckpt', required=True, help='Path to checkpoint file')
    parser.add_argument('--generate-code', action='store_true', 
                       help='Generate compatible model code')
    
    args = parser.parse_args()
    
    print("🔧 HumanMAC Complete Checkpoint Analyzer")
    print("=" * 70)
    
    # Analyze checkpoint
    architecture = analyze_checkpoint_complete(args.ckpt)
    if not architecture:
        return 1
    
    # Generate code if requested
    if args.generate_code:
        generate_compatible_model_code(architecture)
    
    return 0


if __name__ == '__main__':
    exit(main())
