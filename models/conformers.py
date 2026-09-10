import torch
import torch.nn as nn
import torch.nn.functional as F

class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pos = torch.arange(-max_len + 1, max_len, dtype=torch.float)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        sinusoid = torch.einsum('i,j->ij', pos, inv_freq)
        pe = torch.zeros(2 * max_len - 1, d_model)
        pe[:, 0::2] = torch.sin(sinusoid)
        pe[:, 1::2] = torch.cos(sinusoid)
        self.register_buffer('rel_pe', pe)

    def forward(self, seq_len):
        # Returns relative position encodings from -(seq_len-1) to +(seq_len-1), enabling bidirectional attention
        mid = self.rel_pe.size(0) // 2
        return self.rel_pe[mid - (seq_len - 1): mid + seq_len]

class RelPosMultiHeadAttention(nn.Module):
    # Multi-head self-attention with relative positional encoding based on Transformer-XL
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.head_dim = d_model // n_heads
        self.n_heads = n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos_emb):
        B, T, _ = x.shape
        H, D = self.n_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        score_content = torch.matmul(q, k.transpose(-2, -1))

        rel_k = self.k_proj(pos_emb).view(2 * T - 1, H, D).permute(1, 0, 2)
        rel_logits = torch.einsum("bhtd,hkd->bhtk", q, rel_k)
        # rel_logits = F.pad(rel_logits, (0, 1))[:, :, :, T - 1:2 * T - 1]
        # scores = (score_content + rel_logits) / (D ** 0.5)
        # attn = self.dropout(F.softmax(scores, dim=-1))

        # Correct skewing for Transformer-XL relative positions
        rel_logits = F.pad(rel_logits, (1, 0))  # [B, H, T, 2*T]
        rel_logits = rel_logits.view(B, H, -1, T)
        rel_logits = rel_logits[:, :, 1:, :]  # [B, H, 2*T-1, T]
        rel_logits = rel_logits.view(B, H, T, 2*T-1)[:, :, :, :T]  # [B, H, T, T]
        scores = (score_content + rel_logits) / (D ** 0.5)
        attn = self.dropout(F.softmax(scores, dim=-1))

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, H * D)
        return self.out_proj(out)

class ConformerConvModule(nn.Module):
    # Captures local context using convolutional operations between attention layers
    def __init__(self, d_model, kernel_size=31, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2, groups=d_model, bias=False)
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = self.layer_norm(x).transpose(1, 2)
        y = self.glu(self.pointwise_conv1(y))
        y = self.activation(self.batch_norm(self.depthwise_conv(y)))
        y = self.dropout(self.pointwise_conv2(y)).transpose(1, 2)
        return x + y

class ConformerEncoderLayer(nn.Module):
    # Follows the Macaron-style sequence: FFN → MHSA → Conv → FFN used in Conformer
    def __init__(self, d_model, n_heads, d_ff, conv_kernel=31, dropout=0.1):
        super().__init__()
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.attn_norm = nn.LayerNorm(d_model)
        self.attention = RelPosMultiHeadAttention(d_model, n_heads, dropout)
        self.conv_module = ConformerConvModule(d_model, conv_kernel, dropout)

    def forward(self, x, pos_emb):
        x = x + 0.5 * self.ffn1(x)
        x = x + self.attention(self.attn_norm(x), pos_emb)
        x = self.conv_module(x)
        x = x + 0.5 * self.ffn2(x)
        return x

class ConformerEncoder(nn.Module):
    def __init__(self, input_dim, num_layers, d_model, n_heads, d_ff, conv_kernel=31, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model) if input_dim != d_model else nn.Identity()
        self.pos_enc = RelativePositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            ConformerEncoderLayer(d_model, n_heads, d_ff, conv_kernel, dropout)
            for _ in range(num_layers)
        ])
        self.final_ln = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.proj(x.transpose(1, 2))
        pos_emb = self.pos_enc(x.size(1)).to(x.device)
        for layer in self.layers:
            x = layer(x, pos_emb)
        return self.final_ln(x).transpose(1, 2)
