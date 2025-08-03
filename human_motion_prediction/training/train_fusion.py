"""
融合模型训练脚本
训练PISL-HumanMAC融合模型进行人体动作预测
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import argparse
from tqdm import tqdm
import json
import yaml
from datetime import datetime
import wandb

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fusion_model import PISLHumanMACFusion
from data.preprocessing import MotionDataProcessor, MotionDataLoader
from evaluation.metrics import MotionMetrics


class TrainingConfig:
    """训练配置类"""
    
    def __init__(self, config_path: str = None):
        # 默认配置
        self.model_config = {
            'n_joints': 22,
            'input_dim': 66,
            'd_model': 512,
            'spline_degree': 3,
            'n_control_points': 10,
            'n_encoder_layers': 6,
            'n_decoder_layers': 6,
            'n_heads': 8,
            'd_ff': 2048,
            'max_seq_len': 1000,
            'dropout': 0.1,
            'n_timesteps': 1000,
            'beta_schedule': 'cosine',
            'physics_weight': 1.0,
            'smoothness_weight': 0.1,
            'fusion_weight': 0.5
        }
        
        self.training_config = {
            'batch_size': 16,
            'learning_rate': 1e-4,
            'num_epochs': 100,
            'warmup_epochs': 10,
            'weight_decay': 1e-5,
            'gradient_clip_norm': 1.0,
            'save_interval': 10,
            'eval_interval': 5,
            'history_length': 25,
            'future_length': 25
        }
        
        self.data_config = {
            'data_dir': './data',
            'train_subjects': ['S1', 'S5', 'S6', 'S7', 'S8'],
            'val_subjects': ['S9', 'S11'],
            'normalize_method': 'standard',
            'augmentation': True,
            'augmentation_config': {
                'add_noise': True,
                'noise_std': 0.01,
                'rotation': True,
                'rotation_angles': [-10, 10],
                'mirror': True
            }
        }
        
        self.experiment_config = {
            'experiment_name': 'pisl_humanmac_fusion',
            'save_dir': './checkpoints',
            'log_dir': './logs',
            'use_wandb': False,
            'wandb_project': 'human_motion_prediction',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'num_workers': 4,
            'pin_memory': True
        }
        
        # 如果提供了配置文件，加载并覆盖默认配置
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def load_config(self, config_path: str):
        """从文件加载配置"""
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
        
        # 更新配置
        for key, value in config.items():
            if hasattr(self, key):
                getattr(self, key).update(value)
    
    def save_config(self, save_path: str):
        """保存配置到文件"""
        config = {
            'model_config': self.model_config,
            'training_config': self.training_config,
            'data_config': self.data_config,
            'experiment_config': self.experiment_config
        }
        
        with open(save_path, 'w') as f:
            if save_path.endswith('.yaml') or save_path.endswith('.yml'):
                yaml.dump(config, f, default_flow_style=False)
            else:
                json.dump(config, f, indent=2)


class MotionTrainer:
    """运动预测模型训练器"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.experiment_config['device'])
        
        # 创建保存目录
        self.save_dir = config.experiment_config['save_dir']
        self.log_dir = config.experiment_config['log_dir']
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 初始化日志
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # 初始化wandb（如果启用）
        if config.experiment_config['use_wandb']:
            wandb.init(
                project=config.experiment_config['wandb_project'],
                name=config.experiment_config['experiment_name'],
                config=config.__dict__
            )
        
        # 初始化模型
        self.model = self._create_model()
        self.model.to(self.device)
        
        # 初始化优化器和调度器
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # 初始化数据加载器
        self.train_loader, self.val_loader = self._create_data_loaders()
        
        # 初始化评估指标
        self.metrics = MotionMetrics()
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
    def _create_model(self) -> PISLHumanMACFusion:
        """创建融合模型"""
        model = PISLHumanMACFusion(**self.config.model_config)
        
        # 计算模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"模型参数总数: {total_params:,}")
        print(f"可训练参数数: {trainable_params:,}")
        
        return model
    
    def _create_optimizer(self) -> optim.Optimizer:
        """创建优化器"""
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config.training_config['learning_rate'],
            weight_decay=self.config.training_config['weight_decay']
        )
    
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """创建学习率调度器"""
        warmup_epochs = self.config.training_config['warmup_epochs']
        total_epochs = self.config.training_config['num_epochs']
        
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            else:
                return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def _create_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """创建数据加载器"""
        # 数据处理器
        processor = MotionDataProcessor(
            n_joints=self.config.model_config['n_joints'],
            normalize_method=self.config.data_config['normalize_method'],
            augmentation_config=self.config.data_config.get('augmentation_config')
        )
        
        # 训练数据加载器
        train_loader = MotionDataLoader(
            data_dir=self.config.data_config['data_dir'],
            processor=processor,
            batch_size=self.config.training_config['batch_size'],
            shuffle=True,
            augmentation=self.config.data_config['augmentation']
        )
        
        # 加载训练数据
        train_loader.load_h36m_data(self.config.data_config['train_subjects'])
        
        # 验证数据加载器
        val_loader = MotionDataLoader(
            data_dir=self.config.data_config['data_dir'],
            processor=processor,
            batch_size=self.config.training_config['batch_size'],
            shuffle=False,
            augmentation=False
        )
        
        # 加载验证数据
        val_loader.load_h36m_data(self.config.data_config['val_subjects'])
        
        print(f"训练样本数: {len(train_loader)}")
        print(f"验证样本数: {len(val_loader)}")
        
        # 获取PyTorch DataLoader
        train_dataloader = train_loader.get_dataloader()
        val_dataloader = val_loader.get_dataloader()
        
        return train_dataloader, val_dataloader
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        loss_components = {
            'total_loss': 0.0,
            'pisl_loss': 0.0,
            'diffusion_loss': 0.0,
            'fusion_loss': 0.0,
            'physics_loss': 0.0,
            'continuity_loss': 0.0
        }
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, (history, target) in enumerate(progress_bar):
            history = history.to(self.device)
            target = target.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            losses = self.model.compute_loss(history, target)
            
            # 反向传播
            losses['total_loss'].backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training_config['gradient_clip_norm']
            )
            
            self.optimizer.step()
            
            # 累计损失
            for key, value in losses.items():
                loss_components[key] += value.item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': losses['total_loss'].item(),
                'lr': self.optimizer.param_groups[0]['lr']
            })
        
        # 计算平均损失
        num_batches = len(self.train_loader)
        for key in loss_components:
            loss_components[key] /= num_batches
        
        return loss_components
    
    def validate_epoch(self) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        
        total_loss = 0.0
        loss_components = {
            'total_loss': 0.0,
            'pisl_loss': 0.0,
            'diffusion_loss': 0.0,
            'fusion_loss': 0.0,
            'physics_loss': 0.0,
            'continuity_loss': 0.0
        }
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for history, target in tqdm(self.val_loader, desc='Validation'):
                history = history.to(self.device)
                target = target.to(self.device)
                
                # 计算损失
                losses = self.model.compute_loss(history, target)
                
                # 获取预测结果
                predictions = self.model.predict(
                    history, 
                    target.shape[1],
                    use_pisl_only=False,
                    use_diffusion_only=False
                )
                
                # 累计损失
                for key, value in losses.items():
                    loss_components[key] += value.item()
                
                # 收集预测和目标用于计算指标
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        # 计算平均损失
        num_batches = len(self.val_loader)
        for key in loss_components:
            loss_components[key] /= num_batches
        
        # 计算评估指标
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        metrics = self.metrics.compute_metrics(all_predictions, all_targets)
        loss_components.update(metrics)
        
        return loss_components
    
    def train(self):
        """完整训练流程"""
        print("开始训练...")
        
        for epoch in range(self.config.training_config['num_epochs']):
            self.current_epoch = epoch
            
            # 训练
            train_losses = self.train_epoch()
            self.train_losses.append(train_losses['total_loss'])
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录训练损失
            for key, value in train_losses.items():
                self.writer.add_scalar(f'Train/{key}', value, epoch)
            
            # 验证
            if epoch % self.config.training_config['eval_interval'] == 0:
                val_losses = self.validate_epoch()
                self.val_losses.append(val_losses['total_loss'])
                
                # 记录验证损失和指标
                for key, value in val_losses.items():
                    self.writer.add_scalar(f'Val/{key}', value, epoch)
                
                # 保存最佳模型
                if val_losses['total_loss'] < self.best_val_loss:
                    self.best_val_loss = val_losses['total_loss']
                    self.save_checkpoint('best_model.pth', is_best=True)
                
                print(f"Epoch {epoch}: Train Loss = {train_losses['total_loss']:.6f}, "
                      f"Val Loss = {val_losses['total_loss']:.6f}, "
                      f"MPJPE = {val_losses.get('mpjpe', 0):.2f}mm")
                
                # wandb记录
                if self.config.experiment_config['use_wandb']:
                    wandb.log({
                        'epoch': epoch,
                        'train_loss': train_losses['total_loss'],
                        'val_loss': val_losses['total_loss'],
                        'learning_rate': self.optimizer.param_groups[0]['lr'],
                        **{f'train_{k}': v for k, v in train_losses.items()},
                        **{f'val_{k}': v for k, v in val_losses.items()}
                    })
            
            # 定期保存检查点
            if epoch % self.config.training_config['save_interval'] == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')
        
        print("训练完成!")
        
        # 保存最终模型
        self.save_checkpoint('final_model.pth')
        
        # 关闭日志
        self.writer.close()
        if self.config.experiment_config['use_wandb']:
            wandb.finish()
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config.__dict__
        }
        
        filepath = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, filepath)
        
        if is_best:
            print(f"保存最佳模型到: {filepath}")
        else:
            print(f"保存检查点到: {filepath}")
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """加载检查点"""
        if not os.path.exists(checkpoint_path):
            print(f"检查点文件不存在: {checkpoint_path}")
            return False
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        print(f"成功加载检查点: {checkpoint_path}")
        print(f"恢复到第 {self.current_epoch} 轮，最佳验证损失: {self.best_val_loss:.6f}")
        
        return True


def main():
    parser = argparse.ArgumentParser(description='训练PISL-HumanMAC融合模型')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    parser.add_argument('--data_dir', type=str, default='./data', help='数据目录')
    parser.add_argument('--batch_size', type=int, default=16, help='批大小')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--num_epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--use_wandb', action='store_true', help='使用wandb记录')
    
    args = parser.parse_args()
    
    # 加载配置
    config = TrainingConfig(args.config)
    
    # 命令行参数覆盖配置
    if args.data_dir:
        config.data_config['data_dir'] = args.data_dir
    if args.batch_size:
        config.training_config['batch_size'] = args.batch_size
    if args.learning_rate:
        config.training_config['learning_rate'] = args.learning_rate
    if args.num_epochs:
        config.training_config['num_epochs'] = args.num_epochs
    if args.use_wandb:
        config.experiment_config['use_wandb'] = True
    
    # 保存配置
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_save_path = os.path.join(config.experiment_config['save_dir'], f'config_{timestamp}.yaml')
    config.save_config(config_save_path)
    
    # 创建训练器
    trainer = MotionTrainer(config)
    
    # 恢复训练（如果指定）
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()