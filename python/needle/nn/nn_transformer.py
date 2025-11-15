from typing import List
from needle.autograd import Tensor
import needle.backend_ndarray.ndarray as ndarray
from needle import ops
import needle.init as init
import numpy as np
from .nn_sequence import Embedding
from .nn_basic import (
    Module,
    ReLU,
    Dropout,
    LayerNorm1d,
    Linear,
)


class MultiHeadAttention(Module):
    """Multi-head attention primitive operating on pre-projected heads."""

    def __init__(
        self,
        *,
        dropout=0.0,
        causal=False,
        device=None,
        dtype="float32",
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.causal = causal
        self.dropout = Dropout(dropout)

    def create_causal_mask(self, i, j, device):
        mask = -np.finfo(np.float32).max * np.triu(
            np.ones((1, 1, i, j), dtype=np.float32), j - i + 1
        )
        return ndarray.array(mask, device=device)

    def matmul(self, a, b):
        a_shape = (*a.shape[:-1], 1, *a.shape[-1:])
        a = a.reshape(a_shape)

        b_shape = (*b.shape[:-2], 1, *b.shape[-2:])
        b = b.reshape(b_shape)

        lhs = a.broadcast_to(tuple(list(a_shape[:-2]) + [b_shape[-2], a_shape[-1]]))
        rhs = b.broadcast_to(tuple(list(b_shape[:-3]) + [a_shape[-3], *b_shape[-2:]]))
        return (lhs * rhs).sum(len(lhs.shape) - 1)

    def softmax(self, logits):
        max_val = Tensor(
            logits.realize_cached_data().max(axis=3),
            device=logits.device,
            dtype=logits.dtype,
            requires_grad=False,
        )
        max_val = max_val.reshape((*logits.shape[:-1], 1)).broadcast_to(logits.shape)
        exp_logits = ops.exp(logits - max_val)
        denom = exp_logits.sum(axes=3).reshape((*logits.shape[:-1], 1))
        denom = denom.broadcast_to(logits.shape)
        return exp_logits / denom

    def forward(self, q, k, v):
        batch_size, num_head, queries_len, dim = q.shape
        _, _, key_len, _ = k.shape

        scale = 1.0 / np.sqrt(dim)
        attn_scores = self.matmul(q, k) * scale

        if self.causal:
            mask = Tensor(
                self.create_causal_mask(queries_len, key_len, q.device),
                device=q.device,
                dtype=q.dtype,
                requires_grad=False,
            )
            attn_scores = attn_scores + ops.broadcast_to(mask, attn_scores.shape)

        probs = self.dropout(self.softmax(attn_scores))
        output = self.matmul(probs, ops.transpose(v, axes=(2, 3)))
        return output, probs


class AttentionLayer(Module):

    def __init__(
        self,
        q_features: int,
        num_head: int,
        dim_head: int,
        *,
        k_features: int = None,
        v_features: int = None,
        out_features: int = None,
        dropout = 0.,
        causal = False,
        device = None,
        dtype = "float32"
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        if k_features is None:
            k_features = q_features
        if v_features is None:
            v_features = q_features
        if out_features is None:
            out_features = q_features

        self.q_features = q_features
        self.k_features = k_features
        self.v_features = v_features
        self.out_features = out_features

        self.num_head = num_head
        self.dim_head = dim_head

        self.prenorm_q = LayerNorm1d(
            q_features, device=device, dtype=dtype)
        self.prenorm_k = LayerNorm1d(
            k_features, device=device, dtype=dtype)
        self.prenorm_v = LayerNorm1d(
            v_features, device=device, dtype=dtype)

        inner_dim = num_head * dim_head
        
        self.q_projection = Linear(
            q_features, inner_dim, bias=False,
            device=device, dtype=dtype)
        self.k_projection = Linear(
            k_features, inner_dim, bias=False,
            device=device, dtype=dtype)
        self.v_projection = Linear(
            v_features, inner_dim, bias=False,
            device=device, dtype=dtype)

        self.attn = MultiHeadAttention(
            dropout=dropout, causal=causal,
            device=device, dtype=dtype)

        self.out_projection = Linear(
            inner_dim, out_features, bias=False,
            device=device, dtype=dtype)

    def forward(
        self,
        q, k=None, v=None,
    ):
        """
        The forward function of the self-attention layer.
        Input: `q` with shape (batch_size, q_len, q_dim)
               `k` (if not None) with shape (batch_size, kv_len, k_dim)
               `v` (if not None) with shape (batch_size, kv_len, v_dim)
        Output: the output `result` with shape (batch_size, kv_len, out_features)
        """

        if k is None:
            k = q
        if v is None:
            v = q

        batch_size, queries_len, q_dim = q.shape
        _, keys_values_len, k_dim = k.shape
        _, _, v_dim = v.shape

        def _normalize_project(x, prenorm, projection, feature_dim, length):
            x = x.reshape((batch_size * length, feature_dim))
            x = prenorm(x)
            projected = projection(x)
            projected = projected.reshape((batch_size, length, self.num_head, self.dim_head))
            return ops.transpose(projected, axes=(0, 2, 1, 3))

        q_heads = _normalize_project(q, self.prenorm_q, self.q_projection, q_dim, queries_len)
        k_heads = _normalize_project(k, self.prenorm_k, self.k_projection, k_dim, keys_values_len)
        v_heads = _normalize_project(v, self.prenorm_v, self.v_projection, v_dim, keys_values_len)

        attn_out, _ = self.attn(q_heads, k_heads, v_heads)
        attn_out = ops.transpose(attn_out, axes=(0, 2, 1, 3))
        attn_out = attn_out.reshape((batch_size * queries_len, self.num_head * self.dim_head))
        output = self.out_projection(attn_out)
        return output.reshape((batch_size, queries_len, self.out_features))


class TransformerLayer(Module):

    def __init__(
        self,
        q_features: int,
        num_head: int,
        dim_head: int,
        hidden_size: int,
        *,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        self.attention = AttentionLayer(
            q_features, num_head, dim_head, dropout=dropout, causal=causal, device=device, dtype=dtype
        )
        self.attn_dropout = Dropout(dropout)

        self.ff_norm = LayerNorm1d(q_features, device=device, dtype=dtype)
        self.ff_linear1 = Linear(q_features, hidden_size, device=device, dtype=dtype)
        self.relu = ReLU()
        self.ff_dropout = Dropout(dropout)
        self.ff_linear2 = Linear(hidden_size, q_features, device=device, dtype=dtype)
        self.output_dropout = Dropout(dropout)

    def forward(
        self,
        x
    ):
        """
        The forward function of a Transformer Layer.
        Input: the hidden states from previous layers `x` with shape (batch_size, seq_len, x_dim)
        Ouput: the hidden states after the Transformer Layer `x` with shape (batch_size, seq_len, x_dim)
        """

        batch_size, seq_len, x_dim = x.shape

        attn_out = self.attention(x)
        x = x + self.attn_dropout(attn_out)

        normed = self.ff_norm(x.reshape((batch_size * seq_len, x_dim)))
        ff = self.ff_linear1(normed)
        ff = self.relu(ff)
        ff = self.ff_dropout(ff)
        ff = self.ff_linear2(ff)
        ff = ff.reshape((batch_size, seq_len, x_dim))
        x = x + self.output_dropout(ff)
        return x


class Transformer(Module):

    def __init__(
        self,
        embedding_size: int,
        hidden_size: int,
        num_layers: int, 
        *,
        num_head: int = 8,
        dim_head: int = 32,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
        batch_first = False,
        sequence_len = 2048
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype
        self.batch_first = batch_first

        self.embedding_size = embedding_size
        self.sequence_len = sequence_len
        self.positional_embedding = Embedding(sequence_len, embedding_size, device=device, dtype=dtype)

        layers = [
            TransformerLayer(
                embedding_size,
                num_head,
                dim_head,
                hidden_size,
                dropout=dropout,
                causal=causal,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ]
        self.layers = Sequential(*layers)

    def forward(
        self,
        x, h=None
    ):

        if not self.batch_first:
            x = ops.transpose(x, axes=(0, 1))

        batch_size, seq_len, _ = x.shape
        assert seq_len <= self.sequence_len

        position_ids = np.arange(seq_len, dtype=np.int32)
        position_ids = np.repeat(position_ids[:, None], batch_size, axis=1)
        position_ids = Tensor(position_ids, device=x.device, dtype=self.dtype, requires_grad=False)

        pos_embed = self.positional_embedding(position_ids)
        pos_embed = ops.transpose(pos_embed, axes=(0, 1))

        x = x + pos_embed
        x = self.layers(x)

        if not self.batch_first:
            x = ops.transpose(x, axes=(0, 1))

        return x, init.zeros_like(x)
