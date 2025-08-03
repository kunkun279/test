#!/usr/bin/env python3
"""
加速训练脚本：包含多种训练优化技巧
"""

import argparse
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import time

sys.path.append(os.getcwd())
from utils import create_logger, seed_set
from utils.script import *
from config import Config, update_config
from tensorboardX import SummaryWriter
from utils.training import Trainer
from utils.evaluation import compute_stats

class AcceleratedTrainer(Trainer):
    """加速训练器，包含多种优化技巧"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 混合精度训练
        self.use_amp = True
        self.scaler = GradScaler() if self.use_amp else None

        # 累积梯度
        self.accumulation_steps = 2  # 每2步更新一次梯度
        self.accumulated_loss = 0.0
        self.step_count = 0

        # 早停机制
        self.best_val_loss = float('inf')
        self.patience = 20
        self.patience_counter = 0

        # 动态批量大小
        self.dynamic_batch_size = True
        self.max_batch_size = 256
        self.min_batch_size = 32

    def loop(self):
        """重写训练循环，添加早停机制"""
        self.before_train()
        for self.iter in range(0, self.cfg.num_epoch):
            self.before_train_step()
            self.run_train_step()
            self.after_train_step()
            self.before_val_step()
            self.run_val_step()

            # 检查是否需要早停
            should_stop = self.after_val_step()
            if should_stop:
                self.logger.info("Early stopping triggered!")
                break
        
    def before_train(self):
        """改进的训练前设置"""
        # 使用AdamW优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 使用OneCycleLR进行更激进的学习率调度
        self.lr_scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.cfg.lr * 10,  # 最大学习率是初始学习率的10倍
            total_steps=self.cfg.num_epoch,
            pct_start=0.1,  # 10%的时间用于warmup
            anneal_strategy='cos',
            div_factor=10.0,  # 初始学习率 = max_lr / div_factor
            final_div_factor=100.0  # 最终学习率 = max_lr / final_div_factor
        )
        
        self.criterion = nn.MSELoss()
        
        # 编译模型（PyTorch 2.0+）
        if hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model)
                self.logger.info("Model compiled successfully!")
            except Exception as e:
                self.logger.info(f"Model compilation failed: {e}")
    
    def run_train_step(self):
        """改进的训练步骤"""
        
        for traj_np in self.generator_train:
            with torch.no_grad():
                # 数据预处理
                traj_np = traj_np[..., 1:, :].reshape([traj_np.shape[0], self.cfg.t_his + self.cfg.t_pred, -1])
                traj = tensor(traj_np, device=self.cfg.device, dtype=self.cfg.dtype)
                traj_pad = padding_traj(traj, self.cfg.padding, self.cfg.idx_pad, self.cfg.zero_index)
                traj_dct = torch.matmul(self.cfg.dct_m_all[:self.cfg.n_pre], traj)
                traj_dct_mod = torch.matmul(self.cfg.dct_m_all[:self.cfg.n_pre], traj_pad)
                if np.random.random() > self.cfg.mod_train:
                    traj_dct_mod = None

                # condition处理
                if self.condition is not None:
                    batch_size = traj.shape[0]
                    condition_size = self.condition.shape[0]
                    
                    if condition_size >= batch_size:
                        indices = torch.randperm(condition_size)[:batch_size]
                        condition_batch = self.condition[indices]
                    else:
                        indices = torch.arange(batch_size) % condition_size
                        condition_batch = self.condition[indices]
                else:
                    condition_batch = None

            # 前向传播（使用混合精度）
            if self.use_amp:
                with autocast():
                    t = self.diffusion.sample_timesteps(traj.shape[0]).to(self.cfg.device)
                    x_t, noise = self.diffusion.noise_motion(traj_dct, t)
                    predicted_noise = self.model(x_t, t, mod=traj_dct_mod, condition=condition_batch)
                    loss = self.criterion(predicted_noise, noise)
                    loss = loss / self.accumulation_steps  # 归一化累积损失
            else:
                t = self.diffusion.sample_timesteps(traj.shape[0]).to(self.cfg.device)
                x_t, noise = self.diffusion.noise_motion(traj_dct, t)
                predicted_noise = self.model(x_t, t, mod=traj_dct_mod, condition=condition_batch)
                loss = self.criterion(predicted_noise, noise)
                loss = loss / self.accumulation_steps

            # 反向传播
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            self.accumulated_loss += loss.item()
            self.step_count += 1

            # 每accumulation_steps步更新一次参数
            if self.step_count % self.accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()

                # EMA更新
                if self.ema_setup and self.ema_setup[0]:
                    self.ema_setup[1].step_ema(self.ema_setup[2], self.model)

                # 记录损失
                self.train_losses.update(self.accumulated_loss)
                self.tb_logger.add_scalar('Loss/train', self.accumulated_loss, self.iter)
                
                self.accumulated_loss = 0.0

            del loss, traj, traj_dct, traj_dct_mod, traj_pad, traj_np

    def after_val_step(self):
        """改进的验证后处理"""
        super().after_val_step()
        
        # 早停检查
        current_val_loss = self.val_losses.avg
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.patience_counter = 0
            # 保存最佳模型
            torch.save(self.model.state_dict(), 
                      os.path.join(self.cfg.model_dir, f'best_model_epoch_{self.iter}.pt'))
            self.logger.info(f"New best validation loss: {self.best_val_loss:.6f}")
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                self.logger.info(f"Early stopping triggered after {self.patience} epochs without improvement")
                return True  # 触发早停
        
        return False

    def after_train_step(self):
        """改进的训练后处理"""
        # 学习率调度
        self.lr_scheduler.step()
        self.lrs.append(self.optimizer.param_groups[0]['lr'])
        
        self.logger.info(
            '====> Epoch: {} Time: {:.2f} Train Loss: {:.6f} lr: {:.6f}'.format(
                self.iter,
                time.time() - self.t_s,
                self.train_losses.avg,
                self.lrs[-1]))
        
        # 动态调整批量大小（可选）
        if self.dynamic_batch_size and self.iter > 0 and self.iter % 50 == 0:
            self.adjust_batch_size()
        
        # 其他原有逻辑...
        if self.iter % self.cfg.save_gif_interval == 0:
            pose_gen = pose_generator(self.dataset['train'], self.model, self.diffusion, self.cfg, condition=self.condition, mode='gif')
            render_animation(self.dataset['train'].skeleton, pose_gen, ['HumanMAC'], self.cfg.t_his, ncol=4,
                           output=os.path.join(self.cfg.gif_dir, f'training_{self.iter}.gif'))

    def adjust_batch_size(self):
        """动态调整批量大小"""
        try:
            # 根据GPU内存使用情况调整批量大小
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                if memory_used < 0.7 and self.cfg.batch_size < self.max_batch_size:
                    self.cfg.batch_size = min(self.cfg.batch_size + 16, self.max_batch_size)
                    self.logger.info(f"Increased batch size to {self.cfg.batch_size}")
                elif memory_used > 0.9 and self.cfg.batch_size > self.min_batch_size:
                    self.cfg.batch_size = max(self.cfg.batch_size - 16, self.min_batch_size)
                    self.logger.info(f"Decreased batch size to {self.cfg.batch_size}")
        except Exception as e:
            self.logger.info(f"Failed to adjust batch size: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='h36m', help='h36m or humaneva')
    parser.add_argument('--mode', default='train', help='train / eval')
    parser.add_argument('--iter', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str,
                        default=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    parser.add_argument('--multimodal_threshold', type=float, default=0.5)
    parser.add_argument('--multimodal_th_high', type=float, default=0.1)
    parser.add_argument('--milestone', type=list, default=[75, 150, 225, 275, 350, 450])
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--save_model_interval', type=int, default=10)
    parser.add_argument('--save_gif_interval', type=int, default=10)
    parser.add_argument('--save_metrics_interval', type=int, default=100)
    parser.add_argument('--ckpt', type=str, default='./checkpoints/h36m_ckpt.pt')
    parser.add_argument('--ema', type=bool, default=True)
    parser.add_argument('--vis_switch_num', type=int, default=10)
    parser.add_argument('--vis_col', type=int, default=5)
    parser.add_argument('--vis_row', type=int, default=3)
    args = parser.parse_args()

    # 设置
    seed_set(args.seed)
    cfg = Config(f'{args.cfg}', test=(args.mode != 'train'))
    cfg = update_config(cfg, vars(args))
    dataset, dataset_multi_test = dataset_split(cfg)

    # 加载condition数据（腿部角速度和角加速度）
    from scipy import io
    import numpy as np

    data_path = 'data/train'
    # 获取所有 .mat 文件路径
    mat_files = [os.path.join(data_path, file) for file in os.listdir(data_path) if file.endswith('.mat')]

    # 初始化一个空列表用于存储读取的矩阵
    all_data = []

    # 遍历文件，读取并添加到列表中
    for mat_file in mat_files:
        data = io.loadmat(mat_file)['x']  # 假设每个文件都有键 'x'
        all_data.append(data)

    # 合并所有数据，假设需要沿第一个维度拼接
    condition = np.concatenate(all_data, axis=0)

    # 获取condition的维度
    cond_dim = condition.shape
    print("Condition dimension:", cond_dim)
    print("Condition data type: 腿部角速度和角加速度")

    # 对腿部角速度和角加速度数据进行标准化处理
    condition_mean = np.mean(condition, axis=0, keepdims=True)
    condition_std = np.std(condition, axis=0, keepdims=True)
    condition_std = np.where(condition_std == 0, 1.0, condition_std)  # 避免除零
    condition_normalized = (condition - condition_mean) / condition_std

    print(f"Condition statistics:")
    print(f"  Original range: [{np.min(condition):.4f}, {np.max(condition):.4f}]")
    print(f"  Normalized range: [{np.min(condition_normalized):.4f}, {np.max(condition_normalized):.4f}]")
    print(f"  Mean: {np.mean(condition_normalized, axis=0)}")
    print(f"  Std: {np.std(condition_normalized, axis=0)}")

    condition = torch.tensor(condition_normalized, dtype=torch.float32)

    # 保存标准化参数，用于后续的推理
    condition_stats = {
        'mean': condition_mean,
        'std': condition_std
    }
    np.save('condition_stats.npy', condition_stats)

    # 日志
    tb_logger = SummaryWriter(cfg.tb_dir)
    logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'))
    display_exp_setting(logger, cfg)

    # 模型
    model, diffusion = create_model_and_diffusion(cfg, condition)
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs.")
        model = torch.nn.DataParallel(model)

    logger.info(">>> total params: {:.2f}M".format(
        sum(p.numel() for p in list(model.parameters())) / 1000000.0))

    if args.mode == 'train':
        multimodal_dict = get_multimodal_gt_full(logger, dataset_multi_test, args, cfg)
        condition = condition.clone().detach().to(cfg.device)
        
        # 使用加速训练器
        trainer = AcceleratedTrainer(
            model=model,
            diffusion=diffusion,
            dataset=dataset,
            cfg=cfg,
            multimodal_dict=multimodal_dict,
            logger=logger,
            tb_logger=tb_logger,
            condition=condition)
        
        logger.info("🚀 Starting accelerated training...")
        trainer.loop()

if __name__ == '__main__':
    main()
