"""
数据预处理模块
处理人体运动数据，包括归一化、数据增强、序列分割等
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import h5py
import os


class MotionDataProcessor:
    """人体运动数据处理器"""
    
    def __init__(self, 
                 n_joints: int = 22,
                 fps: int = 30,
                 normalize_method: str = "standard",  # "standard", "minmax", "none"
                 augmentation_config: Optional[Dict] = None):
        self.n_joints = n_joints
        self.fps = fps
        self.normalize_method = normalize_method
        self.augmentation_config = augmentation_config or {}
        
        # 标准化器
        self.scaler = None
        self.is_fitted = False
        
        # 关节连接信息（Human3.6M格式）
        self.joint_connections = self._get_joint_connections()
        self.joint_names = self._get_joint_names()
        
    def _get_joint_connections(self) -> List[Tuple[int, int]]:
        """获取关节连接信息"""
        # Human3.6M关节连接
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # 脊柱
            (0, 5), (5, 6), (6, 7),          # 左臂
            (0, 8), (8, 9), (9, 10),         # 右臂
            (0, 11), (11, 12), (12, 13),     # 左腿
            (0, 14), (14, 15), (15, 16),     # 右腿
            (1, 17), (17, 18), (18, 19),     # 头部
            (7, 20), (10, 21)                # 手部
        ]
        return connections
    
    def _get_joint_names(self) -> List[str]:
        """获取关节名称"""
        return [
            'Hip', 'Spine', 'Spine1', 'Spine2', 'Neck',
            'LeftShoulder', 'LeftArm', 'LeftForeArm',
            'RightShoulder', 'RightArm', 'RightForeArm',
            'LeftUpLeg', 'LeftLeg', 'LeftFoot',
            'RightUpLeg', 'RightLeg', 'RightFoot',
            'Head', 'LeftHand', 'RightHand',
            'LeftToe', 'RightToe'
        ]
    
    def fit(self, data: np.ndarray) -> 'MotionDataProcessor':
        """
        拟合数据标准化器
        
        Args:
            data: [n_samples, seq_len, n_joints * 3] 或 [n_samples, seq_len, n_joints, 3]
        """
        if data.ndim == 4:
            # [n_samples, seq_len, n_joints, 3] -> [n_samples * seq_len, n_joints * 3]
            data_flat = data.reshape(-1, self.n_joints * 3)
        elif data.ndim == 3:
            # [n_samples, seq_len, n_joints * 3] -> [n_samples * seq_len, n_joints * 3]
            data_flat = data.reshape(-1, data.shape[-1])
        else:
            raise ValueError(f"Unsupported data shape: {data.shape}")
        
        if self.normalize_method == "standard":
            self.scaler = StandardScaler()
        elif self.normalize_method == "minmax":
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
        else:
            self.scaler = None
        
        if self.scaler is not None:
            self.scaler.fit(data_flat)
        
        self.is_fitted = True
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        标准化数据
        
        Args:
            data: 输入数据
            
        Returns:
            normalized_data: 标准化后的数据
        """
        if not self.is_fitted:
            raise RuntimeError("Processor must be fitted before transform")
        
        original_shape = data.shape
        
        if data.ndim == 4:
            data_flat = data.reshape(-1, self.n_joints * 3)
        elif data.ndim == 3:
            data_flat = data.reshape(-1, data.shape[-1])
        else:
            data_flat = data.reshape(-1, data.shape[-1])
        
        if self.scaler is not None:
            data_flat = self.scaler.transform(data_flat)
        
        return data_flat.reshape(original_shape)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        逆标准化数据
        """
        if not self.is_fitted or self.scaler is None:
            return data
        
        original_shape = data.shape
        data_flat = data.reshape(-1, data.shape[-1])
        data_flat = self.scaler.inverse_transform(data_flat)
        return data_flat.reshape(original_shape)
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """拟合并转换数据"""
        return self.fit(data).transform(data)
    
    def create_sequences(self, 
                        data: np.ndarray,
                        history_length: int = 25,
                        future_length: int = 25,
                        stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建序列数据
        
        Args:
            data: [total_frames, n_joints * 3] 或 [total_frames, n_joints, 3]
            history_length: 历史序列长度
            future_length: 未来序列长度
            stride: 步长
            
        Returns:
            history_sequences: [n_sequences, history_length, n_joints * 3]
            future_sequences: [n_sequences, future_length, n_joints * 3]
        """
        if data.ndim == 3:  # [total_frames, n_joints, 3]
            data = data.reshape(data.shape[0], -1)  # [total_frames, n_joints * 3]
        
        total_frames = data.shape[0]
        sequence_length = history_length + future_length
        
        if total_frames < sequence_length:
            raise ValueError(f"Data length {total_frames} is shorter than required sequence length {sequence_length}")
        
        # 计算序列数量
        n_sequences = (total_frames - sequence_length) // stride + 1
        
        history_sequences = []
        future_sequences = []
        
        for i in range(0, n_sequences * stride, stride):
            if i + sequence_length > total_frames:
                break
            
            # 历史序列
            history = data[i:i + history_length]
            history_sequences.append(history)
            
            # 未来序列
            future = data[i + history_length:i + sequence_length]
            future_sequences.append(future)
        
        return np.array(history_sequences), np.array(future_sequences)
    
    def augment_data(self, data: np.ndarray) -> List[np.ndarray]:
        """
        数据增强
        
        Args:
            data: [seq_len, n_joints, 3] 或 [seq_len, n_joints * 3]
            
        Returns:
            augmented_data: 增强后的数据列表
        """
        if data.ndim == 2:  # [seq_len, n_joints * 3]
            data = data.reshape(data.shape[0], self.n_joints, 3)
        
        augmented = [data]  # 原始数据
        
        # 1. 随机噪声
        if self.augmentation_config.get('add_noise', False):
            noise_std = self.augmentation_config.get('noise_std', 0.01)
            noise = np.random.normal(0, noise_std, data.shape)
            augmented.append(data + noise)
        
        # 2. 时间缩放
        if self.augmentation_config.get('time_scaling', False):
            scale_factors = self.augmentation_config.get('scale_factors', [0.8, 1.2])
            for scale in scale_factors:
                scaled_data = self._time_scale(data, scale)
                if scaled_data is not None:
                    augmented.append(scaled_data)
        
        # 3. 空间旋转
        if self.augmentation_config.get('rotation', False):
            rotation_angles = self.augmentation_config.get('rotation_angles', [-15, 15])
            for angle in rotation_angles:
                rotated_data = self._rotate_motion(data, angle)
                augmented.append(rotated_data)
        
        # 4. 镜像翻转
        if self.augmentation_config.get('mirror', False):
            mirrored_data = self._mirror_motion(data)
            augmented.append(mirrored_data)
        
        # 将结果重塑为原始格式
        result = []
        for aug_data in augmented:
            if aug_data.ndim == 3:
                aug_data = aug_data.reshape(aug_data.shape[0], -1)
            result.append(aug_data)
        
        return result
    
    def _time_scale(self, data: np.ndarray, scale_factor: float) -> Optional[np.ndarray]:
        """时间缩放"""
        seq_len = data.shape[0]
        new_seq_len = int(seq_len * scale_factor)
        
        if new_seq_len < 5:  # 太短的序列跳过
            return None
        
        # 使用线性插值进行时间缩放
        old_indices = np.linspace(0, seq_len - 1, seq_len)
        new_indices = np.linspace(0, seq_len - 1, new_seq_len)
        
        scaled_data = np.zeros((new_seq_len, self.n_joints, 3))
        
        for joint in range(self.n_joints):
            for dim in range(3):
                scaled_data[:, joint, dim] = np.interp(
                    new_indices, old_indices, data[:, joint, dim]
                )
        
        return scaled_data
    
    def _rotate_motion(self, data: np.ndarray, angle_degrees: float) -> np.ndarray:
        """绕Y轴旋转运动"""
        angle_rad = np.radians(angle_degrees)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        
        # Y轴旋转矩阵
        rotation_matrix = np.array([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a]
        ])
        
        rotated_data = np.zeros_like(data)
        for t in range(data.shape[0]):
            for joint in range(self.n_joints):
                rotated_data[t, joint] = rotation_matrix @ data[t, joint]
        
        return rotated_data
    
    def _mirror_motion(self, data: np.ndarray) -> np.ndarray:
        """镜像翻转运动（左右对称）"""
        mirrored_data = data.copy()
        
        # X坐标取反
        mirrored_data[:, :, 0] = -mirrored_data[:, :, 0]
        
        # 交换左右关节
        left_right_pairs = [
            (5, 8),   # 肩膀
            (6, 9),   # 上臂
            (7, 10),  # 前臂
            (11, 14), # 大腿
            (12, 15), # 小腿
            (13, 16), # 脚
            (18, 19), # 手
            (20, 21)  # 脚趾
        ]
        
        for left_idx, right_idx in left_right_pairs:
            if left_idx < self.n_joints and right_idx < self.n_joints:
                # 交换左右关节
                temp = mirrored_data[:, left_idx].copy()
                mirrored_data[:, left_idx] = mirrored_data[:, right_idx]
                mirrored_data[:, right_idx] = temp
        
        return mirrored_data
    
    def compute_bone_lengths(self, data: np.ndarray) -> np.ndarray:
        """
        计算骨骼长度
        
        Args:
            data: [seq_len, n_joints, 3]
            
        Returns:
            bone_lengths: [seq_len, n_connections]
        """
        if data.ndim == 2:
            data = data.reshape(data.shape[0], self.n_joints, 3)
        
        seq_len = data.shape[0]
        n_connections = len(self.joint_connections)
        bone_lengths = np.zeros((seq_len, n_connections))
        
        for t in range(seq_len):
            for i, (joint1, joint2) in enumerate(self.joint_connections):
                if joint1 < self.n_joints and joint2 < self.n_joints:
                    bone_lengths[t, i] = np.linalg.norm(
                        data[t, joint1] - data[t, joint2]
                    )
        
        return bone_lengths
    
    def validate_motion(self, data: np.ndarray) -> Dict[str, float]:
        """
        验证运动数据的质量
        
        Returns:
            quality_metrics: 质量指标字典
        """
        if data.ndim == 2:
            data = data.reshape(data.shape[0], self.n_joints, 3)
        
        metrics = {}
        
        # 1. 计算速度
        velocities = np.diff(data, axis=0)
        velocity_magnitudes = np.linalg.norm(velocities, axis=2)
        
        metrics['max_velocity'] = np.max(velocity_magnitudes)
        metrics['mean_velocity'] = np.mean(velocity_magnitudes)
        metrics['velocity_std'] = np.std(velocity_magnitudes)
        
        # 2. 计算加速度
        accelerations = np.diff(velocities, axis=0)
        acceleration_magnitudes = np.linalg.norm(accelerations, axis=2)
        
        metrics['max_acceleration'] = np.max(acceleration_magnitudes)
        metrics['mean_acceleration'] = np.mean(acceleration_magnitudes)
        
        # 3. 骨骼长度一致性
        bone_lengths = self.compute_bone_lengths(data)
        bone_length_stds = np.std(bone_lengths, axis=0)
        metrics['bone_consistency'] = np.mean(bone_length_stds)
        
        # 4. 运动平滑度
        jerk = np.diff(accelerations, axis=0)
        jerk_magnitudes = np.linalg.norm(jerk, axis=2)
        metrics['smoothness'] = 1.0 / (1.0 + np.mean(jerk_magnitudes))
        
        return metrics


class MotionDataLoader:
    """运动数据加载器"""
    
    def __init__(self, 
                 data_dir: str,
                 processor: MotionDataProcessor,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 augmentation: bool = False):
        self.data_dir = data_dir
        self.processor = processor
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentation = augmentation
        
        self.data_files = []
        self.sequences = []
        self.targets = []
        
    def load_h36m_data(self, subjects: List[str] = None) -> None:
        """加载Human3.6M数据"""
        if subjects is None:
            subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
        
        for subject in subjects:
            data_file = os.path.join(self.data_dir, f'{subject}.h5')
            if os.path.exists(data_file):
                self._load_h5_file(data_file)
    
    def _load_h5_file(self, file_path: str) -> None:
        """加载H5文件"""
        with h5py.File(file_path, 'r') as f:
            for action in f.keys():
                data = f[action][:]  # [n_frames, n_joints * 3]
                
                # 创建序列
                history_seq, future_seq = self.processor.create_sequences(
                    data, history_length=25, future_length=25
                )
                
                # 数据增强
                if self.augmentation:
                    for i in range(len(history_seq)):
                        aug_history = self.processor.augment_data(history_seq[i])
                        aug_future = self.processor.augment_data(future_seq[i])
                        
                        for h, f in zip(aug_history, aug_future):
                            self.sequences.append(h)
                            self.targets.append(f)
                else:
                    self.sequences.extend(history_seq)
                    self.targets.extend(future_seq)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        history = torch.FloatTensor(self.sequences[idx])
        future = torch.FloatTensor(self.targets[idx])
        return history, future
    
    def get_dataloader(self) -> torch.utils.data.DataLoader:
        """获取PyTorch DataLoader"""
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(np.array(self.sequences)),
            torch.FloatTensor(np.array(self.targets))
        )
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=4,
            pin_memory=True
        )


# 示例使用
if __name__ == "__main__":
    # 创建数据处理器
    processor = MotionDataProcessor(
        n_joints=22,
        fps=30,
        normalize_method="standard",
        augmentation_config={
            'add_noise': True,
            'noise_std': 0.01,
            'rotation': True,
            'rotation_angles': [-10, 10],
            'mirror': True
        }
    )
    
    # 生成示例数据
    n_frames = 1000
    n_joints = 22
    
    # 模拟人体运动数据
    t = np.linspace(0, 10, n_frames)
    motion_data = np.zeros((n_frames, n_joints, 3))
    
    for joint in range(n_joints):
        for dim in range(3):
            frequency = 0.5 + joint * 0.1 + dim * 0.2
            motion_data[:, joint, dim] = np.sin(2 * np.pi * frequency * t) + \
                                       0.1 * np.random.randn(n_frames)
    
    # 数据预处理
    normalized_data = processor.fit_transform(motion_data)
    print(f"原始数据形状: {motion_data.shape}")
    print(f"标准化后数据形状: {normalized_data.shape}")
    
    # 创建序列
    history_seq, future_seq = processor.create_sequences(
        normalized_data, history_length=25, future_length=25, stride=5
    )
    print(f"历史序列形状: {history_seq.shape}")
    print(f"未来序列形状: {future_seq.shape}")
    
    # 数据增强
    sample_motion = motion_data[:50]  # 取前50帧
    augmented_motions = processor.augment_data(sample_motion)
    print(f"增强后数据数量: {len(augmented_motions)}")
    
    # 质量验证
    quality_metrics = processor.validate_motion(sample_motion)
    print("\n运动质量指标:")
    for metric, value in quality_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # 骨骼长度分析
    bone_lengths = processor.compute_bone_lengths(sample_motion)
    print(f"\n骨骼长度数据形状: {bone_lengths.shape}")
    print(f"平均骨骼长度变化: {np.mean(np.std(bone_lengths, axis=0)):.4f}")