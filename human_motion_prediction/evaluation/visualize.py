"""
运动预测可视化模块
包含3D人体骨架动画、轨迹对比、指标可视化等功能
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Union
import os


class MotionVisualizer:
    """人体运动可视化工具"""
    
    def __init__(self, n_joints: int = 22, fps: int = 30):
        self.n_joints = n_joints
        self.fps = fps
        
        # Human3.6M关节连接
        self.joint_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # 脊柱
            (0, 5), (5, 6), (6, 7),          # 左臂
            (0, 8), (8, 9), (9, 10),         # 右臂
            (0, 11), (11, 12), (12, 13),     # 左腿
            (0, 14), (14, 15), (15, 16),     # 右腿
            (1, 17), (17, 18), (18, 19),     # 头部
            (7, 20), (10, 21)                # 手部
        ]
        
        # 关节名称
        self.joint_names = [
            'Hip', 'Spine', 'Spine1', 'Spine2', 'Neck',
            'LeftShoulder', 'LeftArm', 'LeftForeArm',
            'RightShoulder', 'RightArm', 'RightForeArm',
            'LeftUpLeg', 'LeftLeg', 'LeftFoot',
            'RightUpLeg', 'RightLeg', 'RightFoot',
            'Head', 'LeftHand', 'RightHand',
            'LeftToe', 'RightToe'
        ]
        
        # 颜色配置
        self.colors = {
            'ground_truth': '#2E8B57',  # 海绿色
            'pisl_prediction': '#FF6347',  # 番茄红
            'diffusion_prediction': '#4169E1',  # 皇家蓝
            'fusion_prediction': '#FF1493',  # 深粉红
            'history': '#696969'  # 暗灰色
        }
    
    def plot_3d_skeleton(self, 
                        motion_data: np.ndarray,
                        title: str = "3D Human Skeleton",
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制3D人体骨架
        
        Args:
            motion_data: [seq_len, n_joints, 3] 运动数据
            title: 图标题
            save_path: 保存路径
        """
        if motion_data.ndim == 2:  # [seq_len, n_joints * 3]
            motion_data = motion_data.reshape(motion_data.shape[0], self.n_joints, 3)
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制第一帧作为示例
        frame = motion_data[0]
        
        # 绘制关节点
        ax.scatter(frame[:, 0], frame[:, 1], frame[:, 2], 
                  c='red', s=50, alpha=0.8)
        
        # 绘制骨骼连接
        for joint1, joint2 in self.joint_connections:
            if joint1 < self.n_joints and joint2 < self.n_joints:
                ax.plot([frame[joint1, 0], frame[joint2, 0]],
                       [frame[joint1, 1], frame[joint2, 1]],
                       [frame[joint1, 2], frame[joint2, 2]], 
                       'b-', linewidth=2, alpha=0.7)
        
        # 添加关节标签
        for i, name in enumerate(self.joint_names[:self.n_joints]):
            ax.text(frame[i, 0], frame[i, 1], frame[i, 2], 
                   name, fontsize=8, alpha=0.7)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        
        # 设置坐标轴比例相等
        max_range = np.array([frame[:, 0].max() - frame[:, 0].min(),
                            frame[:, 1].max() - frame[:, 1].min(),
                            frame[:, 2].max() - frame[:, 2].min()]).max() / 2.0
        mid_x = (frame[:, 0].max() + frame[:, 0].min()) * 0.5
        mid_y = (frame[:, 1].max() + frame[:, 1].min()) * 0.5
        mid_z = (frame[:, 2].max() + frame[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_3d_animation(self,
                           motion_data: np.ndarray,
                           title: str = "3D Human Motion",
                           save_path: Optional[str] = None) -> animation.FuncAnimation:
        """
        创建3D运动动画
        
        Args:
            motion_data: [seq_len, n_joints, 3] 运动数据
            title: 动画标题
            save_path: 保存路径（.gif或.mp4）
        """
        if motion_data.ndim == 2:
            motion_data = motion_data.reshape(motion_data.shape[0], self.n_joints, 3)
        
        seq_len = motion_data.shape[0]
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 设置坐标轴范围
        all_points = motion_data.reshape(-1, 3)
        max_range = np.array([all_points[:, 0].max() - all_points[:, 0].min(),
                            all_points[:, 1].max() - all_points[:, 1].min(),
                            all_points[:, 2].max() - all_points[:, 2].min()]).max() / 2.0
        mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
        mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
        mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        
        # 初始化绘图元素
        joints_plot = ax.scatter([], [], [], c='red', s=50, alpha=0.8)
        bone_lines = []
        for _ in self.joint_connections:
            line, = ax.plot([], [], [], 'b-', linewidth=2, alpha=0.7)
            bone_lines.append(line)
        
        def animate(frame_idx):
            frame = motion_data[frame_idx]
            
            # 更新关节点
            joints_plot._offsets3d = (frame[:, 0], frame[:, 1], frame[:, 2])
            
            # 更新骨骼连接
            for i, (joint1, joint2) in enumerate(self.joint_connections):
                if joint1 < self.n_joints and joint2 < self.n_joints:
                    bone_lines[i].set_data([frame[joint1, 0], frame[joint2, 0]],
                                         [frame[joint1, 1], frame[joint2, 1]])
                    bone_lines[i].set_3d_properties([frame[joint1, 2], frame[joint2, 2]])
            
            ax.set_title(f"{title} - Frame {frame_idx + 1}/{seq_len}")
            
            return [joints_plot] + bone_lines
        
        anim = animation.FuncAnimation(fig, animate, frames=seq_len, 
                                     interval=1000//self.fps, blit=False, repeat=True)
        
        if save_path:
            if save_path.endswith('.gif'):
                anim.save(save_path, writer='pillow', fps=self.fps)
            elif save_path.endswith('.mp4'):
                anim.save(save_path, writer='ffmpeg', fps=self.fps)
        
        return anim
    
    def compare_predictions(self,
                           history: np.ndarray,
                           ground_truth: np.ndarray,
                           predictions: Dict[str, np.ndarray],
                           joint_idx: int = 0,
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        比较不同预测方法的结果
        
        Args:
            history: [seq_len, n_joints, 3] 历史数据
            ground_truth: [seq_len, n_joints, 3] 真实数据
            predictions: 预测结果字典
            joint_idx: 要可视化的关节索引
            save_path: 保存路径
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        history_len = history.shape[0]
        future_len = ground_truth.shape[0]
        total_len = history_len + future_len
        
        time_history = np.arange(history_len)
        time_future = np.arange(history_len, total_len)
        
        for dim, dim_name in enumerate(['X', 'Y', 'Z']):
            ax = axes[dim]
            
            # 绘制历史数据
            ax.plot(time_history, history[:, joint_idx, dim], 
                   color=self.colors['history'], linewidth=2, 
                   label='History', alpha=0.8)
            
            # 绘制真实未来数据
            ax.plot(time_future, ground_truth[:, joint_idx, dim], 
                   color=self.colors['ground_truth'], linewidth=2, 
                   label='Ground Truth', alpha=0.8)
            
            # 绘制预测结果
            for pred_name, pred_data in predictions.items():
                if pred_data.ndim == 2:
                    pred_data = pred_data.reshape(pred_data.shape[0], self.n_joints, 3)
                
                color = self.colors.get(pred_name, np.random.rand(3,))
                ax.plot(time_future, pred_data[:, joint_idx, dim], 
                       color=color, linewidth=2, linestyle='--',
                       label=pred_name.replace('_', ' ').title(), alpha=0.8)
            
            # 添加分界线
            ax.axvline(x=history_len, color='black', linestyle=':', alpha=0.5)
            
            ax.set_xlabel('Frame')
            ax.set_ylabel(f'{dim_name} Position')
            ax.set_title(f'{self.joint_names[joint_idx]} - {dim_name} Coordinate')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_error_heatmap(self,
                          errors: np.ndarray,
                          title: str = "Joint Position Errors",
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制关节误差热力图
        
        Args:
            errors: [seq_len, n_joints] 每个关节每帧的误差
            title: 图标题
            save_path: 保存路径
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 创建热力图
        im = ax.imshow(errors.T, cmap='Reds', aspect='auto', interpolation='nearest')
        
        # 设置坐标轴
        ax.set_xlabel('Frame')
        ax.set_ylabel('Joint')
        ax.set_title(title)
        
        # 设置y轴标签
        joint_labels = [name[:10] for name in self.joint_names[:self.n_joints]]
        ax.set_yticks(range(len(joint_labels)))
        ax.set_yticklabels(joint_labels)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Error (mm)')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_metrics_comparison(self,
                               metrics_dict: Dict[str, Dict[str, float]],
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制不同方法的指标比较
        
        Args:
            metrics_dict: {method_name: {metric_name: value}}
            save_path: 保存路径
        """
        # 选择主要指标
        main_metrics = ['mpjpe', 'p_mpjpe', 'velocity_error', 'acceleration_error',
                       'bone_consistency', 'smoothness']
        
        # 准备数据
        methods = list(metrics_dict.keys())
        metric_values = {metric: [] for metric in main_metrics}
        
        for method in methods:
            for metric in main_metrics:
                value = metrics_dict[method].get(metric, 0)
                metric_values[metric].append(value)
        
        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(main_metrics):
            ax = axes[i]
            
            # 绘制柱状图
            bars = ax.bar(methods, metric_values[metric], alpha=0.7)
            
            # 添加数值标签
            for bar, value in zip(bars, metric_values[metric]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.2f}', ha='center', va='bottom')
            
            ax.set_title(metric.replace('_', ' ').title())
            ax.set_ylabel('Value')
            
            # 旋转x轴标签
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_interactive_3d_plot(self,
                                  motion_data: np.ndarray,
                                  title: str = "Interactive 3D Motion") -> go.Figure:
        """
        创建交互式3D运动可视化
        
        Args:
            motion_data: [seq_len, n_joints, 3] 运动数据
            title: 图标题
        """
        if motion_data.ndim == 2:
            motion_data = motion_data.reshape(motion_data.shape[0], self.n_joints, 3)
        
        seq_len = motion_data.shape[0]
        
        # 创建动画帧
        frames = []
        
        for t in range(seq_len):
            frame_data = motion_data[t]
            
            # 关节点
            scatter = go.Scatter3d(
                x=frame_data[:, 0],
                y=frame_data[:, 1],
                z=frame_data[:, 2],
                mode='markers+text',
                marker=dict(size=8, color='red'),
                text=self.joint_names[:self.n_joints],
                textposition="top center",
                name='Joints'
            )
            
            # 骨骼连接
            bone_traces = []
            for joint1, joint2 in self.joint_connections:
                if joint1 < self.n_joints and joint2 < self.n_joints:
                    bone_trace = go.Scatter3d(
                        x=[frame_data[joint1, 0], frame_data[joint2, 0]],
                        y=[frame_data[joint1, 1], frame_data[joint2, 1]],
                        z=[frame_data[joint1, 2], frame_data[joint2, 2]],
                        mode='lines',
                        line=dict(color='blue', width=5),
                        showlegend=False
                    )
                    bone_traces.append(bone_trace)
            
            frames.append(go.Frame(
                data=[scatter] + bone_traces,
                name=str(t)
            ))
        
        # 创建初始图形
        fig = go.Figure(
            data=frames[0].data,
            frames=frames
        )
        
        # 添加播放控件
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube'
            ),
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {'frame': {'duration': 1000//self.fps, 'redraw': True},
                                       'fromcurrent': True, 'transition': {'duration': 0}}]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {'frame': {'duration': 0, 'redraw': False},
                                         'mode': 'immediate', 'transition': {'duration': 0}}]
                    }
                ]
            }],
            sliders=[{
                'steps': [
                    {
                        'args': [[str(k)], {'frame': {'duration': 0, 'redraw': True},
                                           'mode': 'immediate', 'transition': {'duration': 0}}],
                        'label': str(k),
                        'method': 'animate'
                    }
                    for k in range(seq_len)
                ],
                'active': 0,
                'currentvalue': {'prefix': 'Frame: '},
                'len': 0.9,
                'x': 0.1,
                'xanchor': 'left',
                'y': 0,
                'yanchor': 'top'
            }]
        )
        
        return fig
    
    def plot_trajectory_3d(self,
                          motion_data: np.ndarray,
                          joint_idx: int = 0,
                          title: Optional[str] = None,
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制特定关节的3D轨迹
        
        Args:
            motion_data: [seq_len, n_joints, 3] 运动数据
            joint_idx: 关节索引
            title: 图标题
            save_path: 保存路径
        """
        if motion_data.ndim == 2:
            motion_data = motion_data.reshape(motion_data.shape[0], self.n_joints, 3)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        trajectory = motion_data[:, joint_idx, :]
        
        # 绘制轨迹线
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
               'b-', linewidth=2, alpha=0.7, label='Trajectory')
        
        # 标记起点和终点
        ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 
                  c='green', s=100, label='Start', alpha=0.8)
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], 
                  c='red', s=100, label='End', alpha=0.8)
        
        # 添加方向箭头
        for i in range(0, len(trajectory)-1, max(1, len(trajectory)//10)):
            direction = trajectory[i+1] - trajectory[i]
            ax.quiver(trajectory[i, 0], trajectory[i, 1], trajectory[i, 2],
                     direction[0], direction[1], direction[2],
                     length=0.1, normalize=True, alpha=0.5)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        if title is None:
            title = f'{self.joint_names[joint_idx]} Trajectory'
        ax.set_title(title)
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def save_comparison_video(self,
                             history: np.ndarray,
                             ground_truth: np.ndarray,
                             predictions: Dict[str, np.ndarray],
                             save_path: str,
                             fps: Optional[int] = None):
        """
        保存对比视频
        
        Args:
            history: 历史数据
            ground_truth: 真实数据
            predictions: 预测结果字典
            save_path: 保存路径
            fps: 帧率
        """
        if fps is None:
            fps = self.fps
        
        # 准备数据
        if history.ndim == 2:
            history = history.reshape(history.shape[0], self.n_joints, 3)
        if ground_truth.ndim == 2:
            ground_truth = ground_truth.reshape(ground_truth.shape[0], self.n_joints, 3)
        
        # 组合完整序列
        full_gt = np.concatenate([history, ground_truth], axis=0)
        
        # 创建动画
        fig = plt.figure(figsize=(15, 10))
        
        # 创建子图布局
        gs = fig.add_gridspec(2, len(predictions) + 1, hspace=0.3, wspace=0.3)
        
        # 主要的3D视图
        ax_main = fig.add_subplot(gs[0, :], projection='3d')
        
        # 各个方法的单独视图
        ax_methods = []
        for i in range(len(predictions) + 1):
            ax = fig.add_subplot(gs[1, i], projection='3d')
            ax_methods.append(ax)
        
        def animate(frame_idx):
            # 清空所有子图
            ax_main.clear()
            for ax in ax_methods:
                ax.clear()
            
            # 主视图：显示所有方法的对比
            if frame_idx < len(history):
                # 历史阶段
                frame = history[frame_idx]
                ax_main.scatter(frame[:, 0], frame[:, 1], frame[:, 2], 
                              c='gray', s=30, alpha=0.8, label='History')
            else:
                # 预测阶段
                pred_frame_idx = frame_idx - len(history)
                
                # 真实数据
                gt_frame = ground_truth[pred_frame_idx]
                ax_main.scatter(gt_frame[:, 0], gt_frame[:, 1], gt_frame[:, 2], 
                              c='green', s=30, alpha=0.8, label='Ground Truth')
                
                # 各种预测
                for i, (name, pred_data) in enumerate(predictions.items()):
                    if pred_data.ndim == 2:
                        pred_data = pred_data.reshape(pred_data.shape[0], self.n_joints, 3)
                    
                    pred_frame = pred_data[pred_frame_idx]
                    color = list(self.colors.values())[i % len(self.colors)]
                    ax_main.scatter(pred_frame[:, 0], pred_frame[:, 1], pred_frame[:, 2], 
                                  s=30, alpha=0.6, label=name)
            
            # 绘制骨骼连接
            if frame_idx < len(history):
                frame = history[frame_idx]
            else:
                frame = ground_truth[frame_idx - len(history)]
            
            for joint1, joint2 in self.joint_connections:
                if joint1 < self.n_joints and joint2 < self.n_joints:
                    ax_main.plot([frame[joint1, 0], frame[joint2, 0]],
                               [frame[joint1, 1], frame[joint2, 1]],
                               [frame[joint1, 2], frame[joint2, 2]], 
                               'b-', linewidth=1, alpha=0.5)
            
            ax_main.set_title(f'Motion Comparison - Frame {frame_idx + 1}')
            ax_main.legend()
            
            return []
        
        total_frames = len(history) + len(ground_truth)
        anim = animation.FuncAnimation(fig, animate, frames=total_frames, 
                                     interval=1000//fps, blit=False, repeat=True)
        
        # 保存视频
        if save_path.endswith('.mp4'):
            anim.save(save_path, writer='ffmpeg', fps=fps, bitrate=1800)
        elif save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=fps)
        
        plt.close(fig)


# 示例使用
if __name__ == "__main__":
    # 创建可视化器
    visualizer = MotionVisualizer(n_joints=22, fps=30)
    
    # 生成示例数据
    seq_len = 50
    n_joints = 22
    
    # 模拟运动数据
    t = np.linspace(0, 2, seq_len)
    motion_data = np.zeros((seq_len, n_joints, 3))
    
    for joint in range(n_joints):
        for dim in range(3):
            frequency = 0.5 + joint * 0.1 + dim * 0.2
            motion_data[:, joint, dim] = np.sin(2 * np.pi * frequency * t) + \
                                       0.1 * np.random.randn(seq_len)
    
    # 分割为历史和未来
    history_len = 25
    history = motion_data[:history_len]
    future = motion_data[history_len:]
    
    # 模拟预测结果
    predictions = {
        'pisl_prediction': future + 0.05 * np.random.randn(*future.shape),
        'diffusion_prediction': future + 0.03 * np.random.randn(*future.shape),
        'fusion_prediction': future + 0.02 * np.random.randn(*future.shape)
    }
    
    # 1. 绘制3D骨架
    print("绘制3D骨架...")
    fig_skeleton = visualizer.plot_3d_skeleton(motion_data, "Human Skeleton")
    plt.show()
    
    # 2. 创建3D动画
    print("创建3D动画...")
    anim = visualizer.create_3d_animation(motion_data[:20], "Human Motion Animation")
    plt.show()
    
    # 3. 比较预测结果
    print("比较预测结果...")
    fig_comparison = visualizer.compare_predictions(
        history, future, predictions, joint_idx=0
    )
    plt.show()
    
    # 4. 绘制误差热力图
    print("绘制误差热力图...")
    errors = np.random.rand(25, 22) * 50  # 模拟误差数据
    fig_heatmap = visualizer.plot_error_heatmap(errors, "Joint Position Errors")
    plt.show()
    
    # 5. 绘制指标比较
    print("绘制指标比较...")
    metrics_dict = {
        'PISL': {'mpjpe': 45.2, 'p_mpjpe': 38.1, 'velocity_error': 12.3, 
                'acceleration_error': 8.7, 'bone_consistency': 0.92, 'smoothness': 0.85},
        'Diffusion': {'mpjpe': 42.8, 'p_mpjpe': 36.5, 'velocity_error': 11.8, 
                     'acceleration_error': 9.2, 'bone_consistency': 0.89, 'smoothness': 0.88},
        'Fusion': {'mpjpe': 39.1, 'p_mpjpe': 33.7, 'velocity_error': 10.5, 
                  'acceleration_error': 7.9, 'bone_consistency': 0.94, 'smoothness': 0.91}
    }
    fig_metrics = visualizer.plot_metrics_comparison(metrics_dict)
    plt.show()
    
    # 6. 绘制3D轨迹
    print("绘制3D轨迹...")
    fig_trajectory = visualizer.plot_trajectory_3d(motion_data, joint_idx=0)
    plt.show()
    
    print("可视化示例完成!")