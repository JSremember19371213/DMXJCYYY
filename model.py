# model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力机制"""
    def __init__(self, dropout=0.1, use_relative=False, max_rel_pos=128):
        """
        初始化缩放点积注意力
        
        Args:
            dropout (float): Dropout比率，默认为0.1
            use_relative (bool): 是否使用相对位置编码，默认为False
            max_rel_pos (int): 最大相对位置距离，默认为128
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.use_relative = use_relative
        self.max_rel_pos = max_rel_pos
        # 如果use_relative为True，相对偏置表将由调用方（层）设置

    def forward(self, q, k, v, mask=None, rel_bias=None):
        """
        前向传播过程
        
        Args:
            q (Tensor): 查询张量，形状为(batch, heads, seq_len, dim)
            k (Tensor): 键张量，形状为(batch, heads, seq_len, dim)
            v (Tensor): 值张量，形状为(batch, heads, seq_len, dim)
            mask (Tensor, optional): 掩码张量
            rel_bias (Tensor, optional): 相对位置偏置
            
        Returns:
            tuple: 注意力输出和注意力权重
        """
        # q,k,v: (batch, heads, seq_len, dim)
        d_k = q.size(-1)
        # 计算查询和键的点积，并进行缩放
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if self.use_relative and rel_bias is not None:
            # rel_bias 形状应为 (heads, seq_len, seq_len) 或可广播的形状
            scores = scores + rel_bias.unsqueeze(0)  # 在批次维度上广播
        if mask is not None:
            # mask: (batch, 1, 1, seq_k) 或 (batch, 1, seq_q, seq_k)
            scores = scores.masked_fill(mask == 0, -1e9)
        # 对注意力分数应用softmax归一化
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        # 使用注意力权重加权值向量
        out = torch.matmul(attn, v)
        return out, attn


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, d_model, num_heads, dropout=0.1, use_relative=False, max_rel_pos=128):
        """
        初始化多头注意力
        
        Args:
            d_model (int): 模型维度
            num_heads (int): 注意力头数
            dropout (float): Dropout比率
            use_relative (bool): 是否使用相对位置编码
            max_rel_pos (int): 最大相对位置距离
        """
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 定义查询、键、值的线性变换层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(dropout, use_relative, max_rel_pos)
        self.dropout = nn.Dropout(dropout)

        # 如果需要，创建每个头的相对位置偏置表（Shaw风格）
        self.use_relative = use_relative
        if use_relative:
            # 我们将在层中创建相对距离桶
            # 偏置表形状: (2*max_rel_pos-1, num_heads)
            self.max_rel_pos = max_rel_pos
            self.relative_bias_table = nn.Parameter(torch.zeros((2 * max_rel_pos - 1, num_heads)))
            nn.init.trunc_normal_(self.relative_bias_table, std=0.02)

    def _split_heads(self, x):
        """
        将输入张量分割为多个注意力头
        
        Args:
            x (Tensor): 输入张量，形状为(batch, seq_len, d_model)
            
        Returns:
            Tensor: 分割后的张量，形状为(batch, heads, seq_len, d_k)
        """
        # x: (batch, seq_len, d_model) -> (batch, heads, seq_len, d_k)
        b, seq, _ = x.size()
        x = x.view(b, seq, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def _combine_heads(self, x):
        """
        合并多个注意力头的输出
        
        Args:
            x (Tensor): 多头注意力输出，形状为(batch, heads, seq_len, d_k)
            
        Returns:
            Tensor: 合并后的张量，形状为(batch, seq_len, d_model)
        """
        # x: (batch, heads, seq_len, d_k) -> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous()
        b, seq, _, _ = x.size()
        return x.view(b, seq, self.d_model)

    def _compute_rel_bias(self, qlen, klen, device):
        """
        计算相对位置偏置矩阵，形状为(num_heads, qlen, klen)
        采用简单的Shaw风格实现，使用学习的查找表
        
        Args:
            qlen (int): 查询序列长度
            klen (int): 键序列长度
            device: 设备类型
            
        Returns:
            Tensor: 相对位置偏置矩阵
        """
        max_rel = self.max_rel_pos
        # distances: (qlen, klen)
        q_pos = torch.arange(qlen, dtype=torch.long, device=device)[:, None]
        k_pos = torch.arange(klen, dtype=torch.long, device=device)[None, :]
        relative_pos = k_pos - q_pos  # 范围 -(qlen-1) .. (klen-1)
        relative_pos_clipped = relative_pos.clamp(-max_rel + 1, max_rel - 1) + (max_rel - 1)
        # 查找表: 形状 (2*max_rel-1, num_heads)
        table = self.relative_bias_table  # (2*max_rel-1, num_heads)
        bias = table[relative_pos_clipped]  # (qlen, klen, num_heads)
        bias = bias.permute(2, 0, 1).contiguous()  # (num_heads, qlen, klen)
        return bias

    def forward(self, q, k, v, mask=None):
        """
        前向传播过程
        
        Args:
            q (Tensor): 查询张量
            k (Tensor): 键张量
            v (Tensor): 值张量
            mask (Tensor, optional): 掩码张量
            
        Returns:
            tuple: 注意力输出和注意力权重
        """
        bsz = q.size(0)
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)
        rel_bias = None
        if self.use_relative:
            qlen = q.size(2)
            klen = k.size(2)
            device = q.device
            rel_bias = self._compute_rel_bias(qlen, klen, device)  # (heads, qlen, klen)
        if mask is not None:
            # 如有必要，调整掩码形状以进行广播 (batch, 1, qlen, klen)
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
        out, attn = self.attention(q, k, v, mask, rel_bias)
        out = self._combine_heads(out)
        out = self.w_o(out)
        out = self.dropout(out)
        return out, attn


class PositionwiseFFN(nn.Module):
    """位置前馈神经网络"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        初始化位置前馈网络
        
        Args:
            d_model (int): 模型维度
            d_ff (int): 前馈网络隐藏层维度
            dropout (float): Dropout比率
        """
        super().__init__()
        # 定义两层全连接网络
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        # 激活函数：选择ReLU以简化实现（可以更改）
        self.act = F.relu

    def forward(self, x):
        """
        前向传播过程
        
        Args:
            x (Tensor): 输入张量
            
        Returns:
            Tensor: 前馈网络输出
        """
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, use_relative=False, max_rel_pos=128, use_residual=True):
        """
        初始化Transformer编码器层
        
        Args:
            d_model (int): 模型维度
            num_heads (int): 注意力头数
            d_ff (int): 前馈网络隐藏层维度
            dropout (float): Dropout比率
            use_relative (bool): 是否使用相对位置编码
            max_rel_pos (int): 最大相对位置距离
            use_residual (bool): 是否使用残差连接
        """
        super().__init__()
        # 自注意力机制
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, use_relative, max_rel_pos)
        # 前馈神经网络
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.use_residual = use_residual

    def forward(self, x, mask=None):
        """
        前向传播过程
        
        Args:
            x (Tensor): 输入张量
            mask (Tensor, optional): 掩码张量
            
        Returns:
            tuple: 编码器层输出和注意力权重
        """
        attn_out, attn = self.self_attn(x, x, x, mask)
        if self.use_residual:
            # 使用残差连接
            x = x + self.dropout(attn_out)
        else:
            x = self.dropout(attn_out)
        x = self.norm1(x)
        ffn_out = self.ffn(x)
        if self.use_residual:
            # 使用残差连接
            x = x + self.dropout(ffn_out)
        else:
            x = self.dropout(ffn_out)
        x = self.norm2(x)
        return x, attn


class TransformerDecoderLayer(nn.Module):
    """Transformer解码器层"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, use_relative=False, max_rel_pos=128, use_residual=True):
        """
        初始化Transformer解码器层
        
        Args:
            d_model (int): 模型维度
            num_heads (int): 注意力头数
            d_ff (int): 前馈网络隐藏层维度
            dropout (float): Dropout比率
            use_relative (bool): 是否使用相对位置编码
            max_rel_pos (int): 最大相对位置距离
            use_residual (bool): 是否使用残差连接
        """
        super().__init__()
        # 自注意力机制（带掩码）
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, use_relative, max_rel_pos)
        # 编码器-解码器注意力机制
        self.enc_attn = MultiHeadAttention(d_model, num_heads, dropout, use_relative, max_rel_pos)
        # 前馈神经网络
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.use_residual = use_residual

    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        """
        前向传播过程
        
        Args:
            x (Tensor): 解码器输入张量
            memory (Tensor): 编码器输出
            tgt_mask (Tensor, optional): 目标序列掩码（用于自注意力）
            memory_mask (Tensor, optional): 记忆掩码（用于编码器-解码器注意力）
            
        Returns:
            tuple: 解码器层输出、自注意力权重和编码器-解码器注意力权重
        """
        # 带掩码的自注意力
        self_attn_out, self_attn = self.self_attn(x, x, x, tgt_mask)
        if self.use_residual:
            x = x + self.dropout(self_attn_out)
        else:
            x = self.dropout(self_attn_out)
        x = self.norm1(x)

        # 编码器-解码器注意力
        enc_attn_out, enc_attn = self.enc_attn(x, memory, memory, memory_mask)
        if self.use_residual:
            x = x + self.dropout(enc_attn_out)
        else:
            x = self.dropout(enc_attn_out)
        x = self.norm2(x)

        # 前馈神经网络
        ffn_out = self.ffn(x)
        if self.use_residual:
            x = x + self.dropout(ffn_out)
        else:
            x = self.dropout(ffn_out)
        x = self.norm3(x)
        return x, self_attn, enc_attn


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=5000):
        """
        初始化位置编码
        
        Args:
            d_model (int): 模型维度
            max_len (int): 最大序列长度
        """
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        前向传播过程
        
        Args:
            x (Tensor): 输入张量，形状为(batch, seq_len, d_model)
            
        Returns:
            Tensor: 添加了位置编码的张量
        """
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x


class TransformerEncoder(nn.Module):
    """Transformer编码器"""
    def __init__(self, vocab_size, d_model=128, num_layers=2, num_heads=4, d_ff=512, dropout=0.1,
                 max_len=5000, use_pos=True, use_relative=False, max_rel_pos=128, use_residual=True):
        """
        初始化Transformer编码器
        
        Args:
            vocab_size (int): 词汇表大小
            d_model (int): 模型维度
            num_layers (int): 编码器层数
            num_heads (int): 注意力头数
            d_ff (int): 前馈网络隐藏层维度
            dropout (float): Dropout比率
            max_len (int): 最大序列长度
            use_pos (bool): 是否使用位置编码
            use_relative (bool): 是否使用相对位置编码
            max_rel_pos (int): 最大相对位置距离
            use_residual (bool): 是否使用残差连接
        """
        super().__init__()
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 位置编码层（可选）
        self.pos_enc = PositionalEncoding(d_model, max_len) if use_pos else None
        # 编码器层堆叠
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, num_heads, d_ff, dropout, use_relative, max_rel_pos, use_residual) for _ in range(num_layers)])
        # 层归一化
        self.norm = nn.LayerNorm(d_model)
        # 输出层
        self.output = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self.use_pos = use_pos

    def forward(self, src, src_mask=None):
        """
        前向传播过程
        
        Args:
            src (Tensor): 源序列，形状为(batch, seq)
            src_mask (Tensor, optional): 源序列掩码
            
        Returns:
            tuple: 输出logits、最终隐藏状态和注意力权重列表
        """
        # src: (batch, seq)
        x = self.embedding(src) * math.sqrt(self.d_model)
        if self.pos_enc is not None:
            x = self.pos_enc(x)
        attns = []
        for layer in self.layers:
            x, attn = layer(x, src_mask)
            attns.append(attn)
        x = self.norm(x)
        logits = self.output(x)
        return logits, x, attns  # 返回logits和最终隐藏状态，用于编码器-解码器注意力


class TransformerDecoder(nn.Module):
    """Transformer解码器"""
    def __init__(self, vocab_size, d_model=128, num_layers=2, num_heads=4, d_ff=512, dropout=0.1,
                 max_len=5000, use_pos=True, use_relative=False, max_rel_pos=128, use_residual=True):
        """
        初始化Transformer解码器
        
        Args:
            vocab_size (int): 词汇表大小
            d_model (int): 模型维度
            num_layers (int): 解码器层数
            num_heads (int): 注意力头数
            d_ff (int): 前馈网络隐藏层维度
            dropout (float): Dropout比率
            max_len (int): 最大序列长度
            use_pos (bool): 是否使用位置编码
            use_relative (bool): 是否使用相对位置编码
            max_rel_pos (int): 最大相对位置距离
            use_residual (bool): 是否使用残差连接
        """
        super().__init__()
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 位置编码层（可选）
        self.pos_enc = PositionalEncoding(d_model, max_len) if use_pos else None
        # 解码器层堆叠
        self.layers = nn.ModuleList([TransformerDecoderLayer(d_model, num_heads, d_ff, dropout, use_relative, max_rel_pos, use_residual) for _ in range(num_layers)])
        # 层归一化
        self.norm = nn.LayerNorm(d_model)
        # 输出层
        self.output = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self.use_pos = use_pos

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        前向传播过程
        
        Args:
            tgt (Tensor): 目标序列，形状为(batch, seq)
            memory (Tensor): 编码器输出
            tgt_mask (Tensor, optional): 目标序列掩码
            memory_mask (Tensor, optional): 记忆掩码
            
        Returns:
            tuple: 输出logits、最终隐藏状态、自注意力权重列表和编码器-解码器注意力权重列表
        """
        # tgt: (batch, seq)
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        if self.pos_enc is not None:
            x = self.pos_enc(x)
        self_attns = []
        enc_attns = []
        for layer in self.layers:
            x, self_attn, enc_attn = layer(x, memory, tgt_mask, memory_mask)
            self_attns.append(self_attn)
            enc_attns.append(enc_attn)
        x = self.norm(x)
        logits = self.output(x)
        return logits, x, self_attns, enc_attns


class TransformerSeq2Seq(nn.Module):
    """序列到序列的Transformer模型"""
    def __init__(self, src_vocab, tgt_vocab, d_model=128, num_layers=2, num_heads=4, d_ff=512, dropout=0.1,
                 max_len=5000, use_pos=True, use_relative=False, max_rel_pos=128, use_residual=True):
        """
        初始化序列到序列的Transformer模型
        
        Args:
            src_vocab (int): 源词汇表大小
            tgt_vocab (int): 目标词汇表大小
            d_model (int): 模型维度
            num_layers (int): 编码器和解码器层数
            num_heads (int): 注意力头数
            d_ff (int): 前馈网络隐藏层维度
            dropout (float): Dropout比率
            max_len (int): 最大序列长度
            use_pos (bool): 是否使用位置编码
            use_relative (bool): 是否使用相对位置编码
            max_rel_pos (int): 最大相对位置距离
            use_residual (bool): 是否使用残差连接
        """
        super().__init__()
        # 编码器和解码器
        self.encoder = TransformerEncoder(src_vocab, d_model, num_layers, num_heads, d_ff, dropout, max_len, use_pos, use_relative, max_rel_pos, use_residual)
        self.decoder = TransformerDecoder(tgt_vocab, d_model, num_layers, num_heads, d_ff, dropout, max_len, use_pos, use_relative, max_rel_pos, use_residual)
        self.tgt_vocab = tgt_vocab

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        """
        前向传播过程
        
        Args:
            src (Tensor): 源序列，形状为(batch, src_len)
            tgt (Tensor): 目标序列，形状为(batch, tgt_len)
            src_mask (Tensor, optional): 源序列掩码
            tgt_mask (Tensor, optional): 目标序列掩码
            memory_mask (Tensor, optional): 记忆掩码
            
        Returns:
            tuple: 解码器输出logits、自注意力权重和编码器-解码器注意力权重
        """
        # src: (batch, src_len), tgt: (batch, tgt_len)
        logits_enc, memory, enc_attns = self.encoder(src, src_mask)
        logits_dec, dec_hidden, self_attns, enc_attns = self.decoder(tgt, memory, tgt_mask, memory_mask)
        return logits_dec, self_attns, enc_attns

    @staticmethod
    def make_src_mask(src):
        """
        创建源序列掩码
        
        Args:
            src (Tensor): 源序列
            
        Returns:
            None: 对于字符级数据集假设没有填充，返回None
        """
        # src padding mask (batch, 1, 1, src_len) - 这里我们假设字符数据集没有填充
        # 如果存在填充token，则相应地创建掩码
        return None

    @staticmethod
    def make_tgt_mask(tgt_len, device):
        """
        创建目标序列掩码（因果掩码）
        
        Args:
            tgt_len (int): 目标序列长度
            device: 设备类型
            
        Returns:
            Tensor: 因果掩码，形状为(1, 1, tgt_len, tgt_len)
        """
        # 用于解码器自注意力的因果掩码: 形状 (1, 1, tgt_len, tgt_len) 或 (batch, 1, tgt_len, tgt_len)
        mask = torch.tril(torch.ones((tgt_len, tgt_len), dtype=torch.uint8, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # (1,1,tgt_len,tgt_len)


def count_parameters(model):
    """计算模型参数数量
    
    Args:
        model (nn.Module): 模型实例
        
    Returns:
        int: 可训练参数总数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)