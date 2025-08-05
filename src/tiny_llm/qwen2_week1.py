import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear


class Qwen2MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        pass

    def __call__(
        self,
        x: mx.array,
        offset: int,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        pass


class Qwen2MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array,  # shape (I, E)
        w_up: mx.array,    # shape (I, E)
        w_down: mx.array,  # shape (E, I)
    ):
        self.D = dim
        self.H = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    def __call__(self, x: mx.array) -> mx.array:
        """
        N.. is zero or more dimensions for batches
        E is hidden_size (embedding dimension of the model)
        I is intermediate_size (dimension of the hidden layer in MLP)
        L is the sequence length

        input: N.. x L x E
        w_gate: I x E
        w_up: I x E
        w_down: E x I
        output: N.. x L x E
        """
        # 1. Linear projection for gate and up: x @ W.T (E → I)
        gate_proj = mx.matmul(x, self.w_gate.T)  # shape: N.. x L x I
        up_proj = mx.matmul(x, self.w_up.T)      # shape: N.. x L x I

        # 2. Apply SiLU activation to gate projection
        activated_gate = silu(gate_proj)      # shape: N.. x L x I

        # 3. Element-wise multiply (gating)
        gated = activated_gate * up_proj      # shape: N.. x L x I

        # 4. Final down projection: I → E
        out = mx.matmul(gated, self.w_down.T)    # shape: N.. x L x E

        return out

class Qwen2TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        pass

    def __call__(
        self,
        x: mx.array,
        offset: int,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        pass


class Qwen2ModelWeek1:
    def __init__(self, mlx_model: Any):
        pass

    def __call__(
        self,
        inputs: mx.array,
        offset: int,
    ) -> mx.array:
        pass
