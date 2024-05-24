import torch 
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import math

try:
    from causal_conv1d import causal_conv1d_fn
except:
    assert 0, print(f"Need to install causal_conv1d: pip install causal_conv1d")


class SelfAttention(nn.Module):
    def __init__(self, attention_dropout=0.0):
        super().__init__()
        self.dropout_p = attention_dropout

    def forward(self, qk, v):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D)
        """
        seqlen = qk.shape[1]
        q, k = qk.unbind(dim=2)
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
        scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)
        causal_mask = torch.triu(
            torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1
        )
        scores = scores + causal_mask.to(dtype=scores.dtype)
        attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        attention_drop = F.dropout(attention, self.dropout_p if self.training else 0.0)
        output = torch.einsum("bhts,bshd->bthd", attention_drop, v)
        return output

class SelfLinAttention(nn.Module):
    def __init__(self, attention_dropout=0.0):
        super().__init__()
        self.dropout_p = attention_dropout
        print("using generalized lin attention")

    def forward(self, qk, v):
        """Implements the multihead linear attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D)
        """
        seqlen = qk.shape[1]
        q, k = qk.unbind(dim=2)
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        scores = torch.einsum("bthd,bshd->bhts", q, k)
        mask_mul = torch.tril(
            torch.full((seqlen, seqlen), 1, device=scores.device), 0
        )
        scores = scores * mask_mul.to(dtype=scores.dtype)
        nu = scores.sum(dim=-1)
        attention = torch.div(scores, nu[:,:,:,None])
        attention_drop = F.dropout(attention, self.dropout_p if self.training else 0.0)
        output = torch.einsum("bhts,bshd->bthd", attention_drop, v)
        return output
    
class MHA(nn.Module):
    """Multi-head self-attention
    """

    def __init__(
        self,
        d_model: int,
        d_qk: int=None,
        num_heads: int=1,
        bias: bool=True,
        lin_att: bool=True,
        dropout: float=0.0,
        layer_idx: int=None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        if d_qk is None:
            self.d_qk = d_model
        else:
            self.d_qk = d_qk

        self.layer_idx = layer_idx
        self.num_heads = num_heads
        assert (
            self.d_qk % num_heads == 0
        ), "self.kdim must be divisible by num_heads"
        assert (
            self.d_model % num_heads == 0
        ), "self.vdim must be divisible by num_heads"
        self.head_dim = self.d_qk // num_heads
        self.v_dim = self.d_model // num_heads
        self.Wqk = nn.Linear(
            d_model, 2 * self.d_qk, bias=bias
        )
        self.Wv = nn.Linear(d_model, d_model, bias=bias)
        if lin_att:
            self.inner_attn = SelfLinAttention(attention_dropout=dropout)
        else:
            self.inner_attn = SelfAttention(attention_dropout=dropout)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor):
        """"""
        qk = self.Wqk(x)
        v = self.Wv(x)
        qk = rearrange(
            qk, "... (two h d) -> ... two h d", two=2, d=self.head_dim
        )
        v = rearrange(
            v, "... (h d) -> ... h d", d=self.v_dim
        )
        context = self.inner_attn(qk, v)
        out = self.out_proj(rearrange(context, "... h d -> ... (h d)"))
        return out
    
    def state_size(self, batch_size: int=1, sequence_length: int=2048):
        return 2 * self.d_model * self.d_qk

class MHA_MAMBA(nn.Module):
    """Multi-head self-attention
    """

    def __init__(
        self,
        d_model: int,
        d_qk: int=None,
        num_heads: int=1,
        d_conv:int=4,
        bias: bool=False,
        conv_bias: bool=True,
        lin_att: bool=True,
        dropout: float=0.0,
        layer_idx: int=None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        if d_qk is None:
            self.d_qk = d_model
        else:
            self.d_qk = d_qk

        self.layer_idx = layer_idx
        self.num_heads = num_heads
        assert (
            self.d_qk % num_heads == 0
        ), "self.kdim must be divisible by num_heads"
        assert (
            self.d_model % num_heads == 0
        ), "self.vdim must be divisible by num_heads"
        self.head_dim = self.d_qk // num_heads
        self.v_dim = self.d_model // num_heads

        self.in_proj = nn.Linear(self.d_model, self.d_model * 2, bias=bias)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_model,
            out_channels=self.d_model,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_model,
            padding=d_conv - 1,
        )
        self.Wqk = nn.Linear(
            d_model, 2 * self.d_qk, bias=bias
        )
        self.Wv = nn.Linear(d_model, d_model, bias=bias)
        if lin_att:
            self.inner_attn = SelfLinAttention(attention_dropout=dropout)
        else:
            self.inner_attn = SelfAttention(attention_dropout=dropout)
        self.out_proj = nn.Linear(d_model, d_model)

        self.activation = "silu"
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor):
        """"""
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=2)

        x = causal_conv1d_fn(
                    x,
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias = self.conv1d.bias,
                    activation = self.activation
                )

        qk = self.Wqk(x)
        v = self.Wv(x)
        qk = rearrange(
            qk, "... (two h d) -> ... two h d", two=2, d=self.head_dim
        )
        v = rearrange(
            v, "... (h d) -> ... h d", d=self.v_dim
        )
        context = self.inner_attn(qk, v)
        out = rearrange(context, "... h d -> ... (h d)")
        out = out * F.silu(z)
        out = self.out_proj(out)
        return out
    
    def state_size(self, batch_size: int=1, sequence_length: int=2048):
        return 2 * self.d_model * self.d_qk

class SelfNormAttention(nn.Module):
    def __init__(self, attention_dropout=0.0):
        super().__init__()
        self.dropout_p = attention_dropout
        print("using normed lin attention")

    def forward(self, qk, v, n):
        """Implements the multihead linear attention with normalization.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D)
        """
        seqlen = qk.shape[1]
        q, k = qk.unbind(dim=2)
        scores = torch.einsum("bthd,bshd->bhts", q, k)
        mask_mul = torch.tril(
            torch.full((seqlen, seqlen), 1, device=scores.device), 0
        )
        n = torch.exp(rearrange(n, "b l h -> b h l"))
        scores = scores * mask_mul.to(dtype=scores.dtype)
        attention = torch.div(scores, n[:,:,:,None])
        attention_drop = F.dropout(attention, self.dropout_p if self.training else 0.0)
        output = torch.einsum("bhts,bshd->bthd", attention_drop, v)
        return output

class MHNA(nn.Module):
    """Multi-head self-attention with normalization
    """

    def __init__(
        self,
        d_model: int,
        d_qk: int=None,
        num_heads: int=1,
        bias: bool=True,
        lin_att: bool=True,
        dropout: float=0.0,
        layer_idx: int=None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        if d_qk is None:
            self.d_qk = d_model
        else:
            self.d_qk = d_qk

        self.layer_idx = layer_idx
        self.num_heads = num_heads
        assert (
            self.d_qk % num_heads == 0
        ), "self.kdim must be divisible by num_heads"
        assert (
            self.d_model % num_heads == 0
        ), "self.vdim must be divisible by num_heads"
        self.head_dim = self.d_qk // num_heads
        self.v_dim = self.d_model // num_heads

        self.Wqk = nn.Linear(
            d_model, 2 * self.d_qk, bias=bias
        )
        self.Wv = nn.Linear(d_model, d_model, bias=bias)
        self.Wn = nn.Linear(d_model, num_heads, bias=bias)
        self.inner_attn = SelfNormAttention(attention_dropout=dropout)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor):
        """"""
        qk = self.Wqk(x)
        v = self.Wv(x)
        n = self.Wn(x)
        qk = rearrange(
            qk, "... (two h d) -> ... two h d", two=2, d=self.head_dim
        )
        v = rearrange(
            v, "... (h d) -> ... h d", d=self.v_dim
        )
        context = self.inner_attn(qk, v, n)
        out = self.out_proj(rearrange(context, "... h d -> ... (h d)"))
        return out
    
    def state_size(self, batch_size: int=1, sequence_length: int=2048):
        return 2 * self.d_model * self.d_qk