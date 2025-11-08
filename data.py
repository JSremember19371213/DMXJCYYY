# data.py
import os
from pathlib import Path
import requests
from typing import Tuple, List


def download_tiny_shakespeare(target_path: str):
    """下载tiny_shakespeare数据集，如果本地不存在则从URL下载
    
    Args:
        target_path (str): 数据集保存路径
    
    Returns:
        str: 数据集文件路径
    """
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    target = Path(target_path)
    if target.exists():
        return str(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"正在下载 tiny_shakespeare 数据集到 {target} ...")
    import requests
    r = requests.get(url)
    r.raise_for_status()
    target.write_text(r.text, encoding='utf-8')
    return str(target)


class CharDataset:
    """
    字符级数据集类，用于语言模型和序列到序列任务
    对于语言模型任务: 返回 (x, y) ，其中 y 是 x 向右偏移一位的结果（语言建模）
    对于序列到序列自编码任务: 返回 (src, tgt) ，src = x, tgt = y (同样偏移)
    如果需要真正的平行语料，可以扩展此类或添加其他加载器
    """
    def __init__(self, file_path: str, seq_len: int = 128, mode: str = 'lm'):
        """
        初始化字符级数据集
        
        Args:
            file_path (str): 数据文件路径
            seq_len (int): 序列长度，默认为128
            mode (str): 模式，'lm'表示语言模型，'seq2seq'表示序列到序列，默认为'lm'
        """
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"{file_path} 未找到。请先运行 download_tiny_shakespeare() 下载数据集。")
        text = p.read_text(encoding='utf-8')
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.idx2char = {i: ch for i, ch in enumerate(self.chars)}
        self.char2idx = {ch: i for i, ch in enumerate(self.chars)}
        self.data = [self.char2idx[ch] for ch in text]
        self.seq_len = seq_len
        self.mode = mode

    def __len__(self):
        # 返回数据集长度，确保不会越界
        return max(0, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx):
        # 获取指定索引的数据样本
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + 1 + self.seq_len]
        # 转换为列表（在collate_fn中会转换为张量）
        if self.mode == 'lm':
            # 语言模型模式：返回输入序列和向右偏移一位的目标序列
            return x, y
        elif self.mode == 'seq2seq':
            # 序列到序列自编码模式：源序列是x，目标序列是y（偏移后的）
            # 在真实的序列到序列任务中，会有独立的源和目标预处理流程
            return x, y
        else:
            raise ValueError("mode 必须是 'lm' 或 'seq2seq'")

    def collate_fn(self, batch):
        # 批处理函数，将样本列表转换为张量
        import torch
        xs = [b[0] for b in batch]
        ys = [b[1] for b in batch]
        return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


if __name__ == '__main__':
    # 快速测试下载器功能
    p = download_tiny_shakespeare('data/tiny_shakespeare.txt')
    ds = CharDataset(p, seq_len=64)
    print('词汇表大小:', ds.vocab_size, '数据集长度:', len(ds))
    x, y = ds[0]
    print('示例序列长度', len(x), len(y))