# HumanMAC 检查点加载故障排除指南

本指南帮助解决HumanMAC项目中检查点加载相关的常见问题。

## 🚀 快速解决方案

### 一键检查和修复
```bash
# 运行完整的快速启动检查（推荐）
python quick_start_check.py --cfg h36m_fast --ckpt ./checkpoints/ckpt_ema_150.pt
```

### 分步检查工具

#### 1. 简单检查点测试
```bash
# 简单的检查点加载测试（无复杂依赖）
python simple_checkpoint_test.py --ckpt ./checkpoints/ckpt_ema_150.pt
```

#### 2. 简单条件数据检查
```bash
# 简单的条件数据检查和修复
python simple_condition_check.py
```

#### 3. 完整测试（如果简单测试通过）
```bash
# 完整的检查点加载测试
python test_checkpoint_loading.py --cfg h36m_fast --ckpt ./checkpoints/ckpt_ema_150.pt

# 完整的条件数据修复
python fix_condition_loading.py

# 检查点详细信息
python check_checkpoint.py --ckpt ./checkpoints/ckpt_ema_150.pt
```

## 🔧 已实现的修复

### 1. 灵活的检查点加载系统

新的检查点加载系统 (`utils/checkpoint_utils.py`) 可以处理多种检查点格式：

- **字典格式检查点**: 包含 `model_state_dict`, `optimizer_state_dict` 等
- **直接状态字典**: 直接保存的模型参数
- **不同的键名**: 支持 `model_state_dict`, `state_dict`, `model`, `net` 等键名
- **严格/非严格加载**: 自动尝试严格加载，失败时回退到非严格加载

### 2. 更新的主要文件

- `main.py`: 使用新的灵活检查点加载系统
- `evaluate_mpjpe.py`: 使用新的灵活检查点加载系统
- `utils/checkpoint_utils.py`: 新的检查点工具模块

## 🐛 常见问题及解决方案

### 问题1: RuntimeError: Error(s) in loading state_dict

**症状**: 
```
RuntimeError: Error(s) in loading state_dict for MotionTransformer:
```

**原因**: 检查点格式与模型期望的状态字典格式不匹配

**解决方案**:
1. 使用新的灵活加载系统（已自动应用）
2. 检查检查点内容：
   ```bash
   python check_checkpoint.py --ckpt your_checkpoint.pt
   ```

### 问题2: 条件数据加载失败

**症状**: 
```
FileNotFoundError: data/train directory not found
```

**原因**: 缺少条件数据文件

**解决方案**:
1. 运行条件数据修复工具：
   ```bash
   python fix_condition_loading.py
   ```
2. 如果没有真实数据，可以创建虚拟数据进行测试

### 问题3: 模型架构不匹配

**症状**: 
```
Missing keys: [...] 
Unexpected keys: [...]
```

**原因**: 检查点中的模型架构与当前代码中的模型不匹配

**解决方案**:
1. 系统会自动使用非严格加载模式
2. 检查日志中的缺失和意外键
3. 确保使用正确的配置文件

### 问题4: CUDA内存不足

**症状**: 
```
RuntimeError: CUDA out of memory
```

**解决方案**:
1. 使用CPU进行评价：
   ```bash
   python main.py --cfg h36m_fast --mode eval --ckpt ./checkpoints/ckpt_ema_150.pt --device cpu
   ```
2. 减少批量大小（修改配置文件中的 `batch_size`）

## 📋 检查点格式支持

### 支持的检查点格式

1. **完整训练检查点**:
   ```python
   {
       'model_state_dict': {...},
       'optimizer_state_dict': {...},
       'epoch': 150,
       'loss': 0.123
   }
   ```

2. **仅模型状态**:
   ```python
   {
       'state_dict': {...}
   }
   ```

3. **直接状态字典**:
   ```python
   OrderedDict([
       ('layer1.weight', tensor(...)),
       ('layer1.bias', tensor(...)),
       ...
   ])
   ```

### 自动检测逻辑

系统按以下优先级查找模型状态：
1. `model_state_dict`
2. `state_dict` 
3. `model`
4. `net`
5. 整个字典（如果所有值都是张量）

## 🧪 测试工具

### 1. 检查点加载测试
```bash
python test_checkpoint_loading.py --cfg h36m_fast --ckpt ./checkpoints/ckpt_ema_150.pt
```

**功能**:
- 检查检查点文件信息
- 测试模型创建
- 测试检查点加载
- 验证模型前向传播

### 2. 条件数据修复
```bash
python fix_condition_loading.py
```

**功能**:
- 检查条件数据完整性
- 自动标准化处理
- 创建虚拟数据（测试用）
- 保存处理后的数据

### 3. 检查点信息查看
```bash
python check_checkpoint.py --ckpt ./checkpoints/ckpt_ema_150.pt
```

**功能**:
- 显示检查点结构
- 列出所有键和参数
- 检测模型架构信息
- 验证检查点完整性

## 🔄 使用流程

### 标准评价流程

1. **检查条件数据**:
   ```bash
   python fix_condition_loading.py
   ```

2. **测试检查点加载**:
   ```bash
   python test_checkpoint_loading.py --cfg h36m_fast --ckpt ./checkpoints/ckpt_ema_150.pt
   ```

3. **运行完整评价**:
   ```bash
   python main.py --cfg h36m_fast --mode eval --ckpt ./checkpoints/ckpt_ema_150.pt
   ```

4. **运行MPJPE评价**:
   ```bash
   python evaluate_mpjpe.py --cfg h36m_fast --ckpt ./checkpoints/ckpt_ema_150.pt
   ```

### 故障排除流程

1. **查看错误信息**: 仔细阅读完整的错误堆栈
2. **检查检查点**: 使用 `check_checkpoint.py` 查看检查点结构
3. **测试加载**: 使用 `test_checkpoint_loading.py` 进行隔离测试
4. **修复数据**: 如果是数据问题，使用 `fix_condition_loading.py`
5. **重新尝试**: 修复后重新运行评价

## 📝 日志和调试

### 启用详细日志

所有脚本都会在 `results/[dataset]/log/` 目录下生成详细日志。

### 关键日志信息

- **检查点加载**: 显示使用的键名和加载模式
- **模型创建**: 显示模型参数数量
- **条件数据**: 显示数据形状和统计信息
- **评价进度**: 显示MPJPE计算进度

## 🆘 获取帮助

如果问题仍然存在：

1. **检查日志文件**: 查看 `results/[dataset]/log/log.txt`
2. **运行测试脚本**: 使用提供的测试工具进行诊断
3. **检查配置**: 确保使用正确的配置文件
4. **验证数据**: 确保所有必需的数据文件存在

## 🔮 未来改进

- 自动检查点格式转换
- 更智能的模型架构匹配
- 增强的错误诊断信息
- 自动数据修复功能
