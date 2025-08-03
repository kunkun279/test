#!/usr/bin/env python3
"""
Fix condition data loading issues

This script helps diagnose and fix common issues with condition data loading
in the HumanMAC project.
"""

import os
import sys
import numpy as np
from scipy import io
import torch


def check_condition_data():
    """Check if condition data exists and is properly formatted"""
    
    print("🔍 Checking condition data...")
    print("=" * 50)
    
    data_path = 'data/train'
    
    # Check if data directory exists
    if not os.path.exists(data_path):
        print(f"❌ Data directory not found: {data_path}")
        print("Please ensure the data directory exists and contains .mat files")
        return False
    
    # Get all .mat files
    mat_files = [os.path.join(data_path, file) for file in os.listdir(data_path) if file.endswith('.mat')]
    
    if not mat_files:
        print(f"❌ No .mat files found in {data_path}")
        print("Please ensure the data directory contains .mat files with condition data")
        return False
    
    print(f"✅ Found {len(mat_files)} .mat files")
    
    # Try to load and process condition data
    try:
        all_data = []
        
        for i, mat_file in enumerate(mat_files):
            print(f"  Loading {os.path.basename(mat_file)}...")
            
            try:
                data = io.loadmat(mat_file)
                
                # Check if 'x' key exists
                if 'x' not in data:
                    print(f"    ⚠️  Warning: 'x' key not found in {mat_file}")
                    print(f"    Available keys: {list(data.keys())}")
                    continue
                
                x_data = data['x']
                print(f"    Shape: {x_data.shape}, Type: {x_data.dtype}")
                all_data.append(x_data)
                
            except Exception as e:
                print(f"    ❌ Error loading {mat_file}: {e}")
                continue
        
        if not all_data:
            print("❌ No valid condition data found")
            return False
        
        # Concatenate all data
        condition = np.concatenate(all_data, axis=0)
        print(f"\n✅ Combined condition data shape: {condition.shape}")
        
        # Check for NaN or infinite values
        if np.any(np.isnan(condition)):
            print("⚠️  Warning: NaN values found in condition data")
        
        if np.any(np.isinf(condition)):
            print("⚠️  Warning: Infinite values found in condition data")
        
        # Normalize condition data
        condition_mean = np.mean(condition, axis=0, keepdims=True)
        condition_std = np.std(condition, axis=0, keepdims=True)
        condition_std = np.where(condition_std == 0, 1.0, condition_std)
        condition_normalized = (condition - condition_mean) / condition_std
        
        print(f"✅ Normalization completed")
        print(f"  Original range: [{np.min(condition):.4f}, {np.max(condition):.4f}]")
        print(f"  Normalized range: [{np.min(condition_normalized):.4f}, {np.max(condition_normalized):.4f}]")
        
        # Save normalized condition and stats
        condition_tensor = torch.tensor(condition_normalized, dtype=torch.float32)
        condition_stats = {
            'mean': condition_mean,
            'std': condition_std,
            'original_shape': condition.shape,
            'normalized_shape': condition_normalized.shape
        }
        
        # Save for later use
        torch.save(condition_tensor, 'condition_normalized.pt')
        np.save('condition_stats.npy', condition_stats)

        # Also save enhanced condition stats for compatibility
        enhanced_stats = {
            'mean': condition_mean,
            'std': condition_std,
            'original_shape': condition.shape,
            'normalized_shape': condition_normalized.shape
        }
        np.save('enhanced_condition_stats.npy', enhanced_stats)
        
        print(f"✅ Saved normalized condition data to 'condition_normalized.pt'")
        print(f"✅ Saved condition stats to 'condition_stats.npy'")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing condition data: {e}")
        return False


def create_dummy_condition_data():
    """Create dummy condition data for testing purposes"""
    
    print("\n🔧 Creating dummy condition data for testing...")
    
    # Create dummy condition data (leg angular velocity and acceleration)
    # Assuming 6 dimensions: 3 for angular velocity, 3 for angular acceleration
    num_samples = 10000
    condition_dim = 6
    
    # Generate realistic-looking dummy data
    np.random.seed(42)
    condition = np.random.randn(num_samples, condition_dim) * 0.5
    
    # Add some structure to make it more realistic
    t = np.linspace(0, 100, num_samples)
    for i in range(condition_dim):
        condition[:, i] += 0.1 * np.sin(0.1 * t + i)
    
    print(f"✅ Created dummy condition data: {condition.shape}")
    
    # Normalize
    condition_mean = np.mean(condition, axis=0, keepdims=True)
    condition_std = np.std(condition, axis=0, keepdims=True)
    condition_std = np.where(condition_std == 0, 1.0, condition_std)
    condition_normalized = (condition - condition_mean) / condition_std
    
    # Save as .mat file for compatibility
    os.makedirs('data/train', exist_ok=True)
    io.savemat('data/train/dummy_condition.mat', {'x': condition})
    
    # Also save normalized version
    condition_tensor = torch.tensor(condition_normalized, dtype=torch.float32)
    condition_stats = {
        'mean': condition_mean,
        'std': condition_std,
        'original_shape': condition.shape,
        'normalized_shape': condition_normalized.shape
    }
    
    torch.save(condition_tensor, 'condition_normalized.pt')
    np.save('condition_stats.npy', condition_stats)
    
    print(f"✅ Saved dummy condition data to 'data/train/dummy_condition.mat'")
    print(f"✅ Saved normalized condition data to 'condition_normalized.pt'")
    print(f"✅ Saved condition stats to 'condition_stats.npy'")
    
    return True


def main():
    print("🔧 HumanMAC Condition Data Fix Tool")
    print("=" * 50)
    
    # First, try to check existing condition data
    success = check_condition_data()
    
    if not success:
        print("\n❌ Condition data check failed.")
        
        response = input("\n🤔 Would you like to create dummy condition data for testing? (y/n): ")
        
        if response.lower() in ['y', 'yes']:
            dummy_success = create_dummy_condition_data()
            
            if dummy_success:
                print("\n✅ Dummy condition data created successfully!")
                print("You can now test the model with:")
                print("  python test_checkpoint_loading.py")
                print("  python main.py --cfg h36m_fast --mode eval --ckpt ./checkpoints/ckpt_ema_150.pt")
            else:
                print("\n❌ Failed to create dummy condition data")
                return 1
        else:
            print("\n📝 Please ensure proper condition data is available in 'data/train/' directory")
            return 1
    
    else:
        print("\n✅ Condition data is properly configured!")
        print("You can now run:")
        print("  python test_checkpoint_loading.py")
        print("  python main.py --cfg h36m_fast --mode eval --ckpt ./checkpoints/ckpt_ema_150.pt")
    
    return 0


if __name__ == '__main__':
    exit(main())
