# utils.py
import json
from pathlib import Path
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save_json(obj, path):
    """
    将对象保存为JSON文件
    
    Args:
        obj: 要保存的对象
        path: 保存路径
    """
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path):
    """
    从JSON文件加载对象
    
    Args:
        path: JSON文件路径
        
    Returns:
        加载的对象
    """
    import json
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_checkpoint(model, optimizer, scheduler, epoch, path, extra=None):
    """
    保存模型检查点
    
    Args:
        model: 模型对象
        optimizer: 优化器对象
        scheduler: 学习率调度器对象
        epoch: 当前轮次
        path: 检查点保存路径
        extra: 额外信息（可选）
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'epoch': epoch
    }
    if extra:
        state.update(extra)
    torch.save(state, str(path))


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location=None):
    """
    加载模型检查点
    
    Args:
        path: 检查点路径
        model: 模型对象
        optimizer: 优化器对象（可选）
        scheduler: 学习率调度器对象（可选）
        map_location: 设备映射位置（可选）
        
    Returns:
        dict: 检查点状态字典
    """
    state = torch.load(str(path), map_location=map_location)
    model.load_state_dict(state['model_state_dict'])
    if optimizer is not None and state.get('optimizer_state_dict') is not None:
        optimizer.load_state_dict(state['optimizer_state_dict'])
    if scheduler is not None and state.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(state['scheduler_state_dict'])
    return state


def plot_train_curve(losses, out_path):
    """
    绘制训练损失曲线
    
    Args:
        losses: 损失值列表
        out_path: 图像保存路径
    """
    plt.figure()
    plt.plot(losses)
    plt.xlabel('步骤')
    plt.ylabel('训练损失')
    plt.title('训练损失曲线')
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path))
    plt.close()


def save_vocab(chars, path):
    """
    保存词汇表
    
    Args:
        chars: 字符列表，假定按索引到字符的映射顺序排列
        path: 保存路径
    """
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(chars, f, ensure_ascii=False, indent=2)


def load_vocab(path):
    """
    加载词汇表
    
    Args:
        path: 词汇表文件路径
        
    Returns:
        list: 字符列表
    """
    import json
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)