import mlx.core as mx


class RMSNorm:
    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
        self.D = dim
        self.w = weight
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        assert x.shape[-1] == self.D, \
                f"The last dimension: {x.shape[-1]} should equal to RMSNorm dim: {self.D}"
        x_squared = mx.square(x.astype(mx.float32))  # same shape as x
        rms = mx.sqrt(mx.mean(x_squared, axis=-1, keepdims=True) + self.eps)  # shape (..., 1)
        x_normed = (x / rms) * self.w  # broadcasts self.w over last dim
        return x_normed