#!/usr/bin/env python3
"""
高级数据增强策略
"""

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

class MotionAugmentation:
    """运动数据增强"""
    
    def __init__(self, aug_prob=0.5):
        self.aug_prob = aug_prob
        
    def random_rotation(self, motion, max_angle=15):
        """随机旋转"""
        if np.random.random() > self.aug_prob:
            return motion
        
        # 随机旋转角度
        angle = np.random.uniform(-max_angle, max_angle)
        rotation = R.from_euler('y', angle, degrees=True)
        
        # 应用旋转
        motion_rotated = motion.copy()
        for i in range(motion.shape[0]):  # batch
            for j in range(motion.shape[1]):  # time
                motion_rotated[i, j] = rotation.apply(motion[i, j].reshape(-1, 3)).reshape(-1)
        
        return motion_rotated
    
    def random_scale(self, motion, scale_range=(0.9, 1.1)):
        """随机缩放"""
        if np.random.random() > self.aug_prob:
            return motion
        
        scale = np.random.uniform(*scale_range)
        return motion * scale
    
    def random_translation(self, motion, max_trans=0.1):
        """随机平移"""
        if np.random.random() > self.aug_prob:
            return motion
        
        translation = np.random.uniform(-max_trans, max_trans, 3)
        motion_translated = motion.copy()
        
        # 只平移根节点相关的坐标
        for i in range(0, motion.shape[-1], 3):
            motion_translated[..., i:i+3] += translation
        
        return motion_translated
    
    def temporal_masking(self, motion, mask_ratio=0.1):
        """时间掩码"""
        if np.random.random() > self.aug_prob:
            return motion
        
        seq_len = motion.shape[1]
        mask_len = int(seq_len * mask_ratio)
        start_idx = np.random.randint(0, seq_len - mask_len)
        
        motion_masked = motion.copy()
        # 用前一帧的值填充
        if start_idx > 0:
            motion_masked[:, start_idx:start_idx+mask_len] = motion_masked[:, start_idx-1:start_idx]
        else:
            motion_masked[:, start_idx:start_idx+mask_len] = motion_masked[:, start_idx+mask_len:start_idx+mask_len+1]
        
        return motion_masked
    
    def joint_dropout(self, motion, dropout_ratio=0.05):
        """关节dropout"""
        if np.random.random() > self.aug_prob:
            return motion
        
        motion_dropped = motion.copy()
        num_joints = motion.shape[-1] // 3
        num_drop = int(num_joints * dropout_ratio)
        
        drop_joints = np.random.choice(num_joints, num_drop, replace=False)
        
        for joint_idx in drop_joints:
            start_idx = joint_idx * 3
            end_idx = start_idx + 3
            # 用相邻关节的平均值替换
            if joint_idx > 0 and joint_idx < num_joints - 1:
                prev_joint = motion_dropped[..., (joint_idx-1)*3:(joint_idx-1)*3+3]
                next_joint = motion_dropped[..., (joint_idx+1)*3:(joint_idx+1)*3+3]
                motion_dropped[..., start_idx:end_idx] = (prev_joint + next_joint) / 2
        
        return motion_dropped
    
    def speed_perturbation(self, motion, speed_range=(0.8, 1.2)):
        """速度扰动"""
        if np.random.random() > self.aug_prob:
            return motion
        
        speed_factor = np.random.uniform(*speed_range)
        seq_len = motion.shape[1]
        new_len = int(seq_len / speed_factor)
        
        if new_len < seq_len:
            # 加速：下采样
            indices = np.linspace(0, seq_len-1, new_len).astype(int)
            motion_speed = motion[:, indices]
            # 补齐长度
            if new_len < seq_len:
                repeat_last = seq_len - new_len
                last_frame = motion_speed[:, -1:]
                motion_speed = np.concatenate([motion_speed] + [last_frame] * repeat_last, axis=1)
        else:
            # 减速：上采样
            motion_speed = np.repeat(motion, int(speed_factor), axis=1)
            motion_speed = motion_speed[:, :seq_len]  # 截断到原长度
        
        return motion_speed
    
    def apply_augmentation(self, motion):
        """应用所有增强"""
        motion = self.random_rotation(motion)
        motion = self.random_scale(motion)
        motion = self.random_translation(motion)
        motion = self.temporal_masking(motion)
        motion = self.joint_dropout(motion)
        motion = self.speed_perturbation(motion)
        
        return motion

class ConditionAugmentation:
    """Condition数据增强"""
    
    def __init__(self, aug_prob=0.3):
        self.aug_prob = aug_prob
    
    def gaussian_noise(self, condition, noise_std=0.1):
        """高斯噪声"""
        if np.random.random() > self.aug_prob:
            return condition
        
        noise = np.random.normal(0, noise_std, condition.shape)
        return condition + noise
    
    def feature_dropout(self, condition, dropout_ratio=0.1):
        """特征dropout"""
        if np.random.random() > self.aug_prob:
            return condition
        
        condition_aug = condition.copy()
        num_features = condition.shape[-1]
        num_drop = int(num_features * dropout_ratio)
        
        drop_indices = np.random.choice(num_features, num_drop, replace=False)
        condition_aug[..., drop_indices] = 0
        
        return condition_aug
    
    def feature_permutation(self, condition, perm_ratio=0.2):
        """特征置换"""
        if np.random.random() > self.aug_prob:
            return condition
        
        condition_aug = condition.copy()
        num_features = condition.shape[-1]
        num_perm = int(num_features * perm_ratio)
        
        perm_indices = np.random.choice(num_features, num_perm, replace=False)
        np.random.shuffle(perm_indices)
        
        # 重新排列选中的特征
        temp = condition_aug[..., perm_indices].copy()
        np.random.shuffle(temp.T)  # 按特征维度shuffle
        condition_aug[..., perm_indices] = temp
        
        return condition_aug
    
    def apply_augmentation(self, condition):
        """应用condition增强"""
        condition = self.gaussian_noise(condition)
        condition = self.feature_dropout(condition)
        condition = self.feature_permutation(condition)
        
        return condition

# 集成到训练中的增强训练器
class AugmentedTrainer:
    """带数据增强的训练器"""
    
    def __init__(self, base_trainer):
        self.base_trainer = base_trainer
        self.motion_aug = MotionAugmentation(aug_prob=0.6)
        self.condition_aug = ConditionAugmentation(aug_prob=0.4)
    
    def augment_batch(self, traj_np, condition_batch):
        """增强一个batch的数据"""
        # 运动数据增强
        traj_aug = self.motion_aug.apply_augmentation(traj_np)
        
        # Condition数据增强
        if condition_batch is not None:
            condition_aug = self.condition_aug.apply_augmentation(condition_batch.cpu().numpy())
            condition_batch = torch.tensor(condition_aug, device=condition_batch.device, dtype=condition_batch.dtype)
        
        return traj_aug, condition_batch
