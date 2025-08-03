#!/usr/bin/env python3
"""
Fix Condition Encoder Configuration

This script analyzes checkpoint condition encoder dimensions and creates
a compatible configuration or suggests fixes.
"""

import torch
import os
import sys
import argparse
import yaml

sys.path.append(os.getcwd())


def analyze_condition_encoder(ckpt_path):
    """Analyze condition encoder in checkpoint"""
    
    print(f"🔍 Analyzing condition encoder in: {ckpt_path}")
    print("=" * 60)
    
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
        
        # Analyze condition encoder
        condition_info = {}
        
        # Check if condition encoder exists
        has_condition = any('condition_encoder' in key for key in state_dict.keys())
        condition_info['has_condition_encoder'] = has_condition
        
        if not has_condition:
            print("❌ No condition encoder found in checkpoint")
            return None
        
        print("✅ Condition encoder found")
        
        # Find condition encoder layers and dimensions
        condition_layers = {}
        for key, tensor in state_dict.items():
            if 'condition_encoder' in key:
                layer_info = key.replace('condition_encoder.', '').split('.')
                layer_idx = layer_info[0]
                param_type = layer_info[1] if len(layer_info) > 1 else 'unknown'
                
                if layer_idx not in condition_layers:
                    condition_layers[layer_idx] = {}
                
                condition_layers[layer_idx][param_type] = tensor.shape
        
        print("📊 Condition Encoder Architecture:")
        for layer_idx, params in sorted(condition_layers.items()):
            print(f"  Layer {layer_idx}:")
            for param_type, shape in params.items():
                print(f"    {param_type}: {shape}")
        
        # Extract key dimensions
        if '0' in condition_layers and 'weight' in condition_layers['0']:
            # First layer: [hidden_dim, input_dim]
            first_layer_shape = condition_layers['0']['weight']
            condition_info['input_dim'] = first_layer_shape[1]
            condition_info['first_hidden_dim'] = first_layer_shape[0]
        
        # Find the relationship with main model
        main_latent_dims = None
        for key, tensor in state_dict.items():
            if 'out.weight' in key:
                main_latent_dims = tensor.shape[1]
                break
        
        condition_info['main_latent_dims'] = main_latent_dims
        
        print(f"\n📋 Key Dimensions:")
        print(f"  Condition input dim: {condition_info.get('input_dim', 'Unknown')}")
        print(f"  Condition first hidden: {condition_info.get('first_hidden_dim', 'Unknown')}")
        print(f"  Main model latent dims: {condition_info.get('main_latent_dims', 'Unknown')}")
        
        return condition_info
        
    except Exception as e:
        print(f"❌ Error analyzing checkpoint: {e}")
        return None


def create_condition_compatible_config(condition_info, base_config='h36m_fast'):
    """Create a configuration compatible with the condition encoder"""
    
    print(f"\n🔧 Creating condition-compatible configuration...")
    
    base_config_path = f'cfg/{base_config}.yml'
    if not os.path.exists(base_config_path):
        print(f"❌ Base config not found: {base_config_path}")
        return None
    
    try:
        with open(base_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update configuration based on condition encoder analysis
        if condition_info.get('main_latent_dims'):
            config['latent_dims'] = condition_info['main_latent_dims']
        
        # Add condition encoder specific settings if needed
        # This might require model code changes, but we can document the requirements
        
        # Create new config file name
        new_config_name = f"{base_config}_condition_compatible"
        new_config_path = f"cfg/{new_config_name}.yml"
        
        # Add comment about condition encoder
        config_comment = f"""# This configuration is compatible with checkpoint condition encoder
# Condition input dim: {condition_info.get('input_dim', 'Unknown')}
# Condition first hidden: {condition_info.get('first_hidden_dim', 'Unknown')}
# Main latent dims: {condition_info.get('main_latent_dims', 'Unknown')}
"""
        
        # Save new config
        with open(new_config_path, 'w') as f:
            f.write(config_comment)
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"✅ Created condition-compatible config: {new_config_path}")
        print("Updated parameters:")
        if condition_info.get('main_latent_dims'):
            print(f"  latent_dims: {condition_info['main_latent_dims']}")
        
        return new_config_name
        
    except Exception as e:
        print(f"❌ Error creating compatible config: {e}")
        return None


def suggest_model_fixes(condition_info):
    """Suggest model code fixes for condition encoder compatibility"""
    
    print(f"\n💡 Suggested Model Fixes:")
    print("=" * 40)
    
    if condition_info.get('first_hidden_dim') and condition_info.get('main_latent_dims'):
        first_hidden = condition_info['first_hidden_dim']
        main_latent = condition_info['main_latent_dims']
        
        if first_hidden != main_latent // 2:  # Common pattern
            print(f"⚠️  Condition encoder first hidden dim ({first_hidden}) doesn't match expected pattern")
            print(f"   Expected: {main_latent // 2} (main_latent_dims // 2)")
            print(f"   Actual: {first_hidden}")
            
            print(f"\n🔧 Possible fixes:")
            print(f"1. Modify model code to use condition_hidden_dim = {first_hidden}")
            print(f"2. Or retrain with condition_hidden_dim = {main_latent // 2}")
            
    print(f"\n📝 Model code locations to check:")
    print(f"  - Look for condition_encoder initialization")
    print(f"  - Check if condition dimensions are hardcoded")
    print(f"  - Verify condition encoder architecture matches checkpoint")


def main():
    parser = argparse.ArgumentParser(description='Fix condition encoder configuration')
    parser.add_argument('--ckpt', required=True, help='Path to checkpoint file')
    parser.add_argument('--create-config', action='store_true', 
                       help='Create a compatible configuration file')
    parser.add_argument('--base-config', default='h36m_fast', 
                       help='Base configuration to modify (default: h36m_fast)')
    
    args = parser.parse_args()
    
    print("🔧 HumanMAC Condition Encoder Fixer")
    print("=" * 60)
    
    # Analyze condition encoder
    condition_info = analyze_condition_encoder(args.ckpt)
    if not condition_info:
        return 1
    
    # Suggest fixes
    suggest_model_fixes(condition_info)
    
    # Create compatible config if requested
    if args.create_config:
        new_config = create_condition_compatible_config(condition_info, args.base_config)
        if new_config:
            print(f"\n🎉 Try using: python main.py --cfg {new_config} --mode eval --ckpt {args.ckpt}")
        else:
            print(f"\n❌ Failed to create compatible configuration")
    else:
        print(f"\n💡 Use --create-config to create a compatible configuration file")
    
    return 0


if __name__ == '__main__':
    exit(main())
