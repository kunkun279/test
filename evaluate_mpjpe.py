#!/usr/bin/env python3
"""
Standalone MPJPE Evaluation Script for HumanMAC

This script can be used to evaluate motion predictions using MPJPE metrics
at standard intervals (80ms, 160ms, 320ms, 400ms, 1000ms).

Usage:
    python evaluate_mpjpe.py --cfg h36m --ckpt ./checkpoints/h36m_ckpt.pt
    python evaluate_mpjpe.py --cfg humaneva --ckpt ./checkpoints/humaneva_ckpt.pt
"""

import argparse
import sys
import os
import torch
import numpy as np
from tqdm import tqdm

sys.path.append(os.getcwd())
from config import Config, update_config
from utils import create_logger, seed_set
from utils.script import *
from utils.mpjpe_evaluation import MPJPEEvaluator, print_mpjpe_results
from utils.metrics import compute_all_metrics
from utils.checkpoint_utils import load_checkpoint_flexible


def evaluate_mpjpe_only(diffusion, multimodal_dict, model, logger, cfg):
    """
    Evaluate model using only MPJPE metrics at standard intervals
    """
    
    def get_prediction(data, model_select):
        traj_np = data[..., 1:, :].transpose([0, 2, 3, 1])
        traj = torch.tensor(traj_np, device=cfg.device, dtype=torch.float32)
        traj = traj.reshape([traj.shape[0], -1, traj.shape[-1]]).transpose(1, 2)

        mode_dict, traj_dct, traj_dct_cond = sample_preprocessing(traj, cfg, mode='metrics')
        
        # Support condition parameter for enhanced models
        if hasattr(diffusion, 'sample_ddim'):
            try:
                sampled_motion = diffusion.sample_ddim(model_select,
                                                       traj_dct,
                                                       traj_dct_cond,
                                                       mode_dict,
                                                       condition=None)
            except TypeError:
                # Fallback for models without condition parameter
                sampled_motion = diffusion.sample_ddim(model_select,
                                                       traj_dct,
                                                       traj_dct_cond,
                                                       mode_dict)
        else:
            sampled_motion = diffusion.sample_ddim(model_select,
                                                   traj_dct,
                                                   traj_dct_cond,
                                                   mode_dict)

        traj_est = torch.matmul(cfg.idct_m_all[:, :cfg.n_pre], sampled_motion)
        traj_est = traj_est.cpu().numpy()
        traj_est = traj_est[None, ...]
        return traj_est

    gt_group = multimodal_dict['gt_group']
    data_group = multimodal_dict['data_group']
    traj_gt_arr = multimodal_dict['traj_gt_arr']
    num_samples = multimodal_dict['num_samples']

    # Get evaluation parameters from config
    fps = getattr(cfg, 'fps', 50)
    intervals_ms = getattr(cfg, 'mpjpe_intervals', [80, 160, 320, 400, 1000])
    
    # Initialize MPJPE evaluator
    evaluator = MPJPEEvaluator(fps=fps, intervals_ms=intervals_ms)
    
    # Initialize metric accumulators
    mpjpe_metrics = {f'MPJPE_{ms}ms': [] for ms in intervals_ms}
    mm_mpjpe_metrics = {f'MMMPJPE_{ms}ms': [] for ms in intervals_ms}

    K = 50
    logger.info(f"Starting MPJPE evaluation with {K} samples per prediction...")
    logger.info(f"Dataset FPS: {fps}, Evaluation intervals: {intervals_ms}ms")
    
    pred = []
    for i in tqdm(range(0, K), desc="Generating predictions"):
        pred_i_nd = get_prediction(data_group, model)
        pred.append(pred_i_nd)
        
        if i == K - 1:  # in last iteration, concatenate all candidate pred
            pred = np.concatenate(pred, axis=0)
            pred = pred[:, :, cfg.t_his:, :]
            
            # Convert to GPU for faster computation
            try:
                gt_group = torch.from_numpy(gt_group).to('cuda')
                pred = torch.from_numpy(pred).to('cuda')
            except:
                pass
            
            logger.info(f"Evaluating {num_samples} test samples...")
            
            for j in tqdm(range(0, num_samples), desc="Computing MPJPE"):
                pred_sample = pred[:, j, :, :]  # [K, t_pred, 3*joints]
                gt_sample = gt_group[j][np.newaxis, ...]  # [1, t_pred, 3*joints]
                gt_multi_sample = traj_gt_arr[j]  # [multi_modal, t_pred, 3*joints]
                
                # Compute MPJPE metrics
                mpjpe_results = evaluator.compute_mpjpe_single(pred_sample, gt_sample.squeeze(0))
                mm_mpjpe_results = evaluator.compute_mpjpe_multimodal(pred_sample, 
                                                                      torch.from_numpy(gt_multi_sample).to(pred_sample.device))
                
                # Accumulate results
                for metric, value in mpjpe_results.items():
                    mpjpe_metrics[metric].append(value)
                
                for metric, value in mm_mpjpe_results.items():
                    mm_mpjpe_metrics[metric].append(value)
            
            # Compute averages
            logger.info("\n=== MPJPE Evaluation Results ===")
            
            final_results = {}
            
            # Single-modal MPJPE
            logger.info("Single-modal MPJPE:")
            for metric, values in mpjpe_metrics.items():
                avg_value = np.mean(values)
                final_results[metric] = avg_value
                logger.info(f"  {metric}: {avg_value:.4f}mm")
            
            # Multi-modal MPJPE
            logger.info("Multi-modal MPJPE:")
            for metric, values in mm_mpjpe_metrics.items():
                avg_value = np.mean(values)
                final_results[metric] = avg_value
                logger.info(f"  {metric}: {avg_value:.4f}mm")
            
            # Save results to file
            results_file = os.path.join(cfg.result_dir, 'mpjpe_results.txt')
            with open(results_file, 'w') as f:
                f.write("=== MPJPE Evaluation Results ===\n")
                f.write(f"Dataset: {cfg.dataset}\n")
                f.write(f"FPS: {fps}\n")
                f.write(f"Evaluation intervals: {intervals_ms}ms\n\n")
                
                f.write("Single-modal MPJPE:\n")
                for metric, value in final_results.items():
                    if not metric.startswith('MM'):
                        f.write(f"  {metric}: {value:.4f}mm\n")
                
                f.write("\nMulti-modal MPJPE:\n")
                for metric, value in final_results.items():
                    if metric.startswith('MM'):
                        f.write(f"  {metric}: {value:.4f}mm\n")
            
            logger.info(f"Results saved to: {results_file}")
            
            pred = []


def main():
    parser = argparse.ArgumentParser(description='MPJPE Evaluation for HumanMAC')
    parser.add_argument('--cfg', default='h36m', help='h36m or humaneva')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str,
                        default=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    parser.add_argument('--multimodal_threshold', type=float, default=0.5)
    parser.add_argument('--multimodal_th_high', type=float, default=0.1)
    
    args = parser.parse_args()
    
    # Setup
    seed_set(args.seed)
    
    cfg = Config(f'{args.cfg}', test=True)
    cfg = update_config(cfg, vars(args))
    
    dataset, dataset_multi_test = dataset_split(cfg)
    
    # Logger
    logger = create_logger(os.path.join(cfg.log_dir, 'mpjpe_eval_log.txt'))
    logger.info("=== MPJPE Evaluation Started ===")
    logger.info(f"Dataset: {cfg.dataset}")
    logger.info(f"Checkpoint: {args.ckpt}")
    
    # Model
    model, diffusion = create_model_and_diffusion(cfg)
    
    logger.info(">>> total params: {:.2f}M".format(
        sum(p.numel() for p in list(model.parameters())) / 1000000.0))
    
    # Load checkpoint using flexible loader
    try:
        metadata = load_checkpoint_flexible(args.ckpt, model, cfg.device, logger)
    except (FileNotFoundError, RuntimeError) as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return
    
    # Prepare evaluation dataset
    multimodal_dict = get_multimodal_gt_full(logger, dataset_multi_test, args, cfg)
    
    # Run MPJPE evaluation
    evaluate_mpjpe_only(diffusion, multimodal_dict, model, logger, cfg)
    
    logger.info("=== MPJPE Evaluation Completed ===")


if __name__ == '__main__':
    main()
