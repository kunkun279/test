#!/usr/bin/env python3
"""
训练监控脚本：实时监控训练进度和性能
"""

import os
import time
import matplotlib.pyplot as plt
import numpy as np
import re
from datetime import datetime
import argparse

def parse_log_file(log_path):
    """解析训练日志文件"""
    if not os.path.exists(log_path):
        return None, None, None
    
    epochs = []
    train_losses = []
    val_losses = []
    learning_rates = []
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        # 解析训练损失
        train_match = re.search(r'Epoch: (\d+).*Train Loss: ([\d.]+).*lr: ([\d.e-]+)', line)
        if train_match:
            epoch = int(train_match.group(1))
            train_loss = float(train_match.group(2))
            lr = float(train_match.group(3))
            epochs.append(epoch)
            train_losses.append(train_loss)
            learning_rates.append(lr)
        
        # 解析验证损失
        val_match = re.search(r'Epoch: (\d+).*Val Loss: ([\d.]+)', line)
        if val_match:
            val_loss = float(val_match.group(2))
            val_losses.append(val_loss)
    
    return epochs, train_losses, val_losses, learning_rates

def plot_training_curves(epochs, train_losses, val_losses, learning_rates, save_path=None):
    """绘制训练曲线"""
    if not epochs:
        print("No training data found!")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 训练和验证损失
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', alpha=0.7)
    if val_losses and len(val_losses) == len(epochs):
        ax1.plot(epochs, val_losses, 'r-', label='Val Loss', alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 学习率
    ax2.plot(epochs, learning_rates, 'g-', label='Learning Rate')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 损失下降速度
    if len(train_losses) > 1:
        loss_diff = np.diff(train_losses)
        ax3.plot(epochs[1:], loss_diff, 'purple', alpha=0.7)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss Change')
        ax3.set_title('Training Loss Change Rate')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 收敛分析
    if len(train_losses) > 10:
        # 计算最近10个epoch的平均损失变化
        recent_epochs = epochs[-10:]
        recent_losses = train_losses[-10:]
        if len(recent_losses) > 1:
            slope = np.polyfit(recent_epochs, recent_losses, 1)[0]
            ax4.text(0.1, 0.9, f'Recent slope: {slope:.6f}', transform=ax4.transAxes, fontsize=12)
            ax4.text(0.1, 0.8, f'Current loss: {train_losses[-1]:.6f}', transform=ax4.transAxes, fontsize=12)
            ax4.text(0.1, 0.7, f'Best loss: {min(train_losses):.6f}', transform=ax4.transAxes, fontsize=12)
            ax4.text(0.1, 0.6, f'Total epochs: {len(epochs)}', transform=ax4.transAxes, fontsize=12)
            
            # 收敛状态
            if abs(slope) < 1e-5:
                status = "🟢 Converged"
            elif slope < 0:
                status = "🟡 Decreasing"
            else:
                status = "🔴 Increasing"
            ax4.text(0.1, 0.5, f'Status: {status}', transform=ax4.transAxes, fontsize=12, weight='bold')
    
    ax4.set_title('Training Statistics')
    ax4.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    return fig

def monitor_training(results_dir, refresh_interval=30):
    """实时监控训练进度"""
    print(f"🔍 Monitoring training in {results_dir}")
    print(f"📊 Refreshing every {refresh_interval} seconds")
    print("Press Ctrl+C to stop monitoring")
    
    # 找到最新的训练目录
    subdirs = [d for d in os.listdir(results_dir) if d.startswith('h36m_')]
    if not subdirs:
        print("No training directories found!")
        return
    
    latest_dir = max(subdirs, key=lambda x: int(x.split('_')[1]))
    log_path = os.path.join(results_dir, latest_dir, 'log', 'log.txt')
    
    print(f"📁 Monitoring: {latest_dir}")
    print(f"📄 Log file: {log_path}")
    
    try:
        while True:
            epochs, train_losses, val_losses, learning_rates = parse_log_file(log_path)
            
            if epochs:
                # 清屏
                os.system('clear' if os.name == 'posix' else 'cls')
                
                print("=" * 80)
                print(f"🚀 HumanMAC Training Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("=" * 80)
                print(f"📁 Directory: {latest_dir}")
                print(f"📊 Total Epochs: {len(epochs)}")
                print(f"📈 Current Train Loss: {train_losses[-1]:.6f}")
                if val_losses:
                    print(f"📉 Current Val Loss: {val_losses[-1]:.6f}")
                print(f"🎯 Best Train Loss: {min(train_losses):.6f}")
                print(f"📚 Learning Rate: {learning_rates[-1]:.2e}")
                
                # 计算收敛趋势
                if len(train_losses) > 5:
                    recent_slope = np.polyfit(epochs[-5:], train_losses[-5:], 1)[0]
                    if abs(recent_slope) < 1e-5:
                        trend = "🟢 Converged"
                    elif recent_slope < 0:
                        trend = "🟡 Decreasing"
                    else:
                        trend = "🔴 Increasing"
                    print(f"📊 Trend: {trend} (slope: {recent_slope:.2e})")
                
                # 估计剩余时间
                if len(epochs) > 1:
                    avg_time_per_epoch = 40  # 假设每个epoch 40秒
                    remaining_epochs = 1000 - len(epochs)  # 假设总共1000个epoch
                    remaining_time = remaining_epochs * avg_time_per_epoch / 3600  # 小时
                    print(f"⏰ Estimated remaining time: {remaining_time:.1f} hours")
                
                print("=" * 80)
                
                # 生成图表
                plot_path = os.path.join(results_dir, latest_dir, 'training_monitor.png')
                plot_training_curves(epochs, train_losses, val_losses, learning_rates, plot_path)
                
            else:
                print(f"⏳ Waiting for training data... ({datetime.now().strftime('%H:%M:%S')})")
            
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped by user")

def analyze_training(results_dir, experiment_name=None):
    """分析训练结果"""
    if experiment_name:
        log_path = os.path.join(results_dir, experiment_name, 'log', 'log.txt')
    else:
        # 找到最新的训练目录
        subdirs = [d for d in os.listdir(results_dir) if d.startswith('h36m_')]
        if not subdirs:
            print("No training directories found!")
            return
        latest_dir = max(subdirs, key=lambda x: int(x.split('_')[1]))
        log_path = os.path.join(results_dir, latest_dir, 'log', 'log.txt')
        experiment_name = latest_dir
    
    epochs, train_losses, val_losses, learning_rates = parse_log_file(log_path)
    
    if not epochs:
        print("No training data found!")
        return
    
    print(f"📊 Training Analysis for {experiment_name}")
    print("=" * 50)
    print(f"Total epochs: {len(epochs)}")
    print(f"Initial loss: {train_losses[0]:.6f}")
    print(f"Final loss: {train_losses[-1]:.6f}")
    print(f"Best loss: {min(train_losses):.6f}")
    print(f"Loss reduction: {(train_losses[0] - train_losses[-1]) / train_losses[0] * 100:.1f}%")
    
    if len(train_losses) > 10:
        recent_slope = np.polyfit(epochs[-10:], train_losses[-10:], 1)[0]
        print(f"Recent convergence rate: {recent_slope:.2e}")
    
    # 生成详细图表
    plot_path = os.path.join(results_dir, experiment_name, 'training_analysis.png')
    plot_training_curves(epochs, train_losses, val_losses, learning_rates, plot_path)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Monitor HumanMAC training progress')
    parser.add_argument('--mode', choices=['monitor', 'analyze'], default='monitor',
                        help='Monitor real-time or analyze completed training')
    parser.add_argument('--results_dir', default='results',
                        help='Results directory path')
    parser.add_argument('--experiment', default=None,
                        help='Specific experiment name to analyze')
    parser.add_argument('--interval', type=int, default=30,
                        help='Refresh interval in seconds for monitoring')
    
    args = parser.parse_args()
    
    if args.mode == 'monitor':
        monitor_training(args.results_dir, args.interval)
    else:
        analyze_training(args.results_dir, args.experiment)

if __name__ == '__main__':
    main()
