"""
Simplified Evaluation Script for HumanMAC on Human3.6M Dataset
Quick evaluation of MPJPE metrics at different time horizons.
"""

import os
import sys
import torch
import numpy as np
import argparse
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from data_loader.dataset_h36m import DatasetH36M


def compute_mpjpe_simple(pred, gt):
    """
    Simple MPJPE computation.
    
    Args:
        pred: Predicted poses [batch, time, joints*3]
        gt: Ground truth poses [batch, time, joints*3]
        
    Returns:
        MPJPE values [batch, time]
    """
    # Reshape to [batch, time, joints, 3]
    pred = pred.reshape(pred.shape[0], pred.shape[1], -1, 3)
    gt = gt.reshape(gt.shape[0], gt.shape[1], -1, 3)
    
    # Compute L2 distance per joint
    diff = pred - gt
    joint_errors = torch.norm(diff, dim=-1)  # [batch, time, joints]
    
    # Mean across joints
    mpjpe = joint_errors.mean(dim=-1)  # [batch, time]
    
    return mpjpe


def evaluate_baseline(dataset, time_horizons_frames=[4, 8, 16, 20, 50]):
    """
    Evaluate baseline (zero velocity) prediction.
    
    Args:
        dataset: H36M dataset
        time_horizons_frames: Frame indices for evaluation
        
    Returns:
        Dictionary with MPJPE results
    """
    print("Evaluating baseline (zero velocity) prediction...")
    
    results = {f'{ms}ms': [] for ms in [80, 160, 320, 400, 1000]}
    
    for i in tqdm(range(min(100, dataset.data_len)), desc="Processing samples"):
        data_sample = dataset.sample()  # [1, t_total, joints, 3]
        data = torch.tensor(data_sample[0])  # [t_total, joints, 3]
        
        # Split into input and target
        input_seq = data[:dataset.t_his]  # [t_his, joints, 3]
        target_seq = data[dataset.t_his:]  # [t_pred, joints, 3]
        
        # Baseline prediction: repeat last frame
        last_frame = input_seq[-1:].repeat(dataset.t_pred, 1, 1)  # [t_pred, joints, 3]
        
        # Reshape for MPJPE computation
        pred = last_frame.unsqueeze(0).flatten(start_dim=2)  # [1, t_pred, joints*3]
        gt = target_seq.unsqueeze(0).flatten(start_dim=2)    # [1, t_pred, joints*3]
        
        # Compute MPJPE
        mpjpe = compute_mpjpe_simple(pred, gt)  # [1, t_pred]
        
        # Extract values at specific time horizons
        for i, (ms, frame_idx) in enumerate(zip([80, 160, 320, 400, 1000], time_horizons_frames)):
            if frame_idx <= mpjpe.shape[1]:
                mpjpe_value = mpjpe[0, frame_idx-1].item() * 1000  # Convert to mm
                results[f'{ms}ms'].append(mpjpe_value)
    
    # Average results
    final_results = {}
    for horizon_key, values in results.items():
        if values:
            final_results[horizon_key] = np.mean(values)
        else:
            final_results[horizon_key] = None
    
    return final_results


def test_data_loading():
    """Test data loading functionality."""
    print("=== Testing Data Loading ===")
    
    try:
        # Test loading all actions
        cfg = Config('h36m', test=True)
        dataset_all = DatasetH36M(mode='test', t_his=cfg.t_his, t_pred=cfg.t_pred, actions='all')
        print(f"✓ All actions dataset loaded: {dataset_all.data_len} samples")
        
        # Test loading specific action
        dataset_walking = DatasetH36M(mode='test', t_his=cfg.t_his, t_pred=cfg.t_pred, actions=['Walking'])
        print(f"✓ Walking action dataset loaded: {dataset_walking.data_len} samples")
        
        # Test data sample
        if dataset_walking.data_len > 0:
            sample = dataset_walking.sample()
            print(f"✓ Sample shape: {sample.shape}")
            print(f"  Expected: [1, {cfg.t_his + cfg.t_pred}, 17, 3]")
            
        return True
        
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False


def test_mpjpe_computation():
    """Test MPJPE computation."""
    print("\n=== Testing MPJPE Computation ===")
    
    try:
        # Create dummy data
        batch_size, time_steps, joints = 2, 100, 17
        pred = torch.randn(batch_size, time_steps, joints * 3)
        gt = torch.randn(batch_size, time_steps, joints * 3)
        
        # Compute MPJPE
        mpjpe = compute_mpjpe_simple(pred, gt)
        print(f"✓ MPJPE computation successful")
        print(f"  Input shape: {pred.shape}")
        print(f"  Output shape: {mpjpe.shape}")
        print(f"  Sample MPJPE values: {mpjpe[0, :5].tolist()}")
        
        return True
        
    except Exception as e:
        print(f"✗ MPJPE computation failed: {e}")
        return False


def quick_evaluation():
    """Quick evaluation on a subset of data."""
    print("\n=== Quick Evaluation ===")
    
    try:
        # Load dataset
        cfg = Config('h36m', test=True)
        dataset = DatasetH36M(mode='test', t_his=cfg.t_his, t_pred=cfg.t_pred, actions=['Walking'])
        
        if dataset.data_len == 0:
            print("No data available for evaluation")
            return
        
        # Evaluate baseline
        results = evaluate_baseline(dataset)
        
        print("\nBaseline (Zero Velocity) Results:")
        for horizon, mpjpe in results.items():
            if mpjpe is not None:
                print(f"  {horizon}: {mpjpe:.2f} mm")
            else:
                print(f"  {horizon}: N/A")
        
        return results
        
    except Exception as e:
        print(f"✗ Quick evaluation failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Simple HumanMAC H36M Evaluation')
    parser.add_argument('--test_data', action='store_true', 
                       help='Test data loading only')
    parser.add_argument('--test_mpjpe', action='store_true',
                       help='Test MPJPE computation only')
    parser.add_argument('--quick_eval', action='store_true',
                       help='Run quick evaluation with baseline')
    parser.add_argument('--all_tests', action='store_true', default=True,
                       help='Run all tests (default)')
    
    args = parser.parse_args()
    
    print("=== HumanMAC H36M Simple Evaluation ===")
    
    success = True
    
    if args.test_data or args.all_tests:
        success &= test_data_loading()
    
    if args.test_mpjpe or args.all_tests:
        success &= test_mpjpe_computation()
    
    if args.quick_eval or args.all_tests:
        results = quick_evaluation()
        success &= (results is not None)
    
    if success:
        print("\n✓ All tests passed!")
        print("\nNext steps:")
        print("1. Run full evaluation: python eval_h36m_mpjpe_fixed.py --model_path <path_to_model>")
        print("2. Evaluate specific actions: python eval_h36m_mpjpe_fixed.py --actions Walking Eating")
        print("3. Adjust number of samples: python eval_h36m_mpjpe_fixed.py --num_samples 100")
    else:
        print("\n✗ Some tests failed. Please check the error messages above.")


if __name__ == "__main__":
    main()
