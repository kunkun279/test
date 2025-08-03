#!/usr/bin/env python3
"""
高级训练策略：课程学习、对抗训练、知识蒸馏等
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils.training import Trainer

class CurriculumTrainer(Trainer):
    """课程学习训练器"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 课程学习参数
        self.curriculum_stages = [
            {'epochs': 50, 'pred_length': 25, 'noise_scale': 0.5},
            {'epochs': 100, 'pred_length': 50, 'noise_scale': 0.7},
            {'epochs': 150, 'pred_length': 75, 'noise_scale': 0.9},
            {'epochs': 200, 'pred_length': 100, 'noise_scale': 1.0},
        ]
        self.current_stage = 0
        
        # 对抗训练
        self.use_adversarial = True
        if self.use_adversarial:
            self.discriminator = self.create_discriminator()
            self.disc_optimizer = optim.AdamW(
                self.discriminator.parameters(), 
                lr=self.cfg.lr * 0.1, 
                weight_decay=1e-4
            )
        
        # 多任务学习权重
        self.task_weights = {
            'reconstruction': 1.0,
            'prediction': 1.0,
            'adversarial': 0.1,
            'consistency': 0.5
        }
        
    def create_discriminator(self):
        """创建判别器"""
        return nn.Sequential(
            nn.Linear(self.cfg.t_pred * 48, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        ).to(self.cfg.device)
    
    def get_current_curriculum(self):
        """获取当前课程设置"""
        cumulative_epochs = 0
        for i, stage in enumerate(self.curriculum_stages):
            cumulative_epochs += stage['epochs']
            if self.iter < cumulative_epochs:
                return stage, i
        return self.curriculum_stages[-1], len(self.curriculum_stages) - 1
    
    def adaptive_noise_schedule(self, t, stage_info):
        """自适应噪声调度"""
        base_noise = self.diffusion.sample_timesteps(t.shape[0])
        noise_scale = stage_info['noise_scale']
        
        # 根据课程阶段调整噪声
        adjusted_noise = (base_noise * noise_scale).long()
        adjusted_noise = torch.clamp(adjusted_noise, 1, self.cfg.noise_steps - 1)
        
        return adjusted_noise.to(self.cfg.device)
    
    def compute_consistency_loss(self, pred1, pred2):
        """计算一致性损失"""
        return nn.MSELoss()(pred1, pred2)
    
    def compute_adversarial_loss(self, fake_motion, real_motion):
        """计算对抗损失"""
        if not self.use_adversarial:
            return torch.tensor(0.0, device=self.cfg.device)
        
        # 判别器损失
        real_pred = self.discriminator(real_motion.view(real_motion.shape[0], -1))
        fake_pred = self.discriminator(fake_motion.view(fake_motion.shape[0], -1))
        
        disc_loss = nn.BCELoss()(real_pred, torch.ones_like(real_pred)) + \
                   nn.BCELoss()(fake_pred, torch.zeros_like(fake_pred))
        
        # 生成器损失
        gen_loss = nn.BCELoss()(fake_pred, torch.ones_like(fake_pred))
        
        return gen_loss, disc_loss
    
    def run_train_step(self):
        """增强的训练步骤"""
        
        # 获取当前课程
        stage_info, stage_idx = self.get_current_curriculum()
        
        if stage_idx != self.current_stage:
            self.logger.info(f"📚 Entering curriculum stage {stage_idx + 1}: {stage_info}")
            self.current_stage = stage_idx
        
        for traj_np in self.generator_train:
            with torch.no_grad():
                # 数据预处理
                traj_np = traj_np[..., 1:, :].reshape([traj_np.shape[0], self.cfg.t_his + self.cfg.t_pred, -1])
                traj = tensor(traj_np, device=self.cfg.device, dtype=self.cfg.dtype)
                
                # 课程学习：动态调整预测长度
                pred_length = min(stage_info['pred_length'], self.cfg.t_pred)
                traj_truncated = traj[:, :self.cfg.t_his + pred_length]
                
                traj_pad = padding_traj(traj_truncated, self.cfg.padding, self.cfg.idx_pad, self.cfg.zero_index)
                traj_dct = torch.matmul(self.cfg.dct_m_all[:self.cfg.n_pre], traj_truncated)
                traj_dct_mod = torch.matmul(self.cfg.dct_m_all[:self.cfg.n_pre], traj_pad)
                
                if np.random.random() > self.cfg.mod_train:
                    traj_dct_mod = None

                # Condition处理
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

            # 自适应噪声调度
            t = self.adaptive_noise_schedule(traj_dct, stage_info)
            
            # 处理DataParallel
            diffusion_module = self.diffusion.module if hasattr(self.diffusion, 'module') else self.diffusion
            x_t, noise = diffusion_module.noise_motion(traj_dct, t)
            
            # 前向传播
            predicted_noise = self.model(x_t, t, mod=traj_dct_mod, condition=condition_batch)
            
            # 多任务损失
            losses = {}
            
            # 主要重建损失
            losses['reconstruction'] = self.criterion(predicted_noise, noise)
            
            # 一致性损失（使用不同的噪声水平）
            if np.random.random() < 0.3:  # 30%的时间计算一致性损失
                t2 = self.adaptive_noise_schedule(traj_dct, stage_info)
                x_t2, noise2 = diffusion_module.noise_motion(traj_dct, t2)
                predicted_noise2 = self.model(x_t2, t2, mod=traj_dct_mod, condition=condition_batch)
                losses['consistency'] = self.compute_consistency_loss(predicted_noise, predicted_noise2)
            else:
                losses['consistency'] = torch.tensor(0.0, device=self.cfg.device)
            
            # 对抗损失
            if self.use_adversarial and self.iter > 50:  # 在训练稳定后引入对抗损失
                with torch.no_grad():
                    # 生成假样本
                    fake_motion = diffusion_module.sample_ddim(
                        self.model, traj_dct[:4], traj_dct_mod[:4] if traj_dct_mod is not None else None,
                        {'sample_num': 4, 'mask': None, 'mode': 'pred'},
                        condition=condition_batch[:4] if condition_batch is not None else None
                    )
                    real_motion = traj_dct[:4]
                
                gen_loss, disc_loss = self.compute_adversarial_loss(fake_motion, real_motion)
                losses['adversarial'] = gen_loss
                
                # 更新判别器
                self.disc_optimizer.zero_grad()
                disc_loss.backward()
                self.disc_optimizer.step()
            else:
                losses['adversarial'] = torch.tensor(0.0, device=self.cfg.device)
            
            # 总损失
            total_loss = sum(self.task_weights[key] * losses[key] for key in losses)
            
            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # EMA更新
            if self.ema_setup and self.ema_setup[0]:
                self.ema_setup[1].step_ema(self.ema_setup[2], self.model)

            # 记录损失
            self.train_losses.update(total_loss.item())
            self.tb_logger.add_scalar('Loss/train_total', total_loss.item(), self.iter)
            for key, loss in losses.items():
                self.tb_logger.add_scalar(f'Loss/{key}', loss.item(), self.iter)

            del total_loss, traj, traj_dct, traj_dct_mod, traj_pad, traj_np

class KnowledgeDistillationTrainer(Trainer):
    """知识蒸馏训练器"""
    
    def __init__(self, teacher_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        
        # 蒸馏参数
        self.distill_weight = 0.5
        self.temperature = 4.0
        
    def distillation_loss(self, student_output, teacher_output, temperature):
        """计算蒸馏损失"""
        student_soft = torch.softmax(student_output / temperature, dim=-1)
        teacher_soft = torch.softmax(teacher_output / temperature, dim=-1)
        
        return nn.KLDivLoss(reduction='batchmean')(
            torch.log(student_soft), teacher_soft
        ) * (temperature ** 2)
    
    def run_train_step(self):
        """带知识蒸馏的训练步骤"""
        # 实现知识蒸馏的训练逻辑
        # 这里可以添加具体的实现
        pass
