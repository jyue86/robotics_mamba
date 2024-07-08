from flax import linen as nn
from flax.struct import dataclass
import jax
import jax.numpy as jnp
from typing import Union
from einops import repeat, einsum
import math


@dataclass
class ModelArgs:
    d_model: int
    d_inner: int
    n_layers: int
    vocab_size: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = "auto"
    d_conv: int = 4 
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False

    @classmethod
    def init(
        self,
        d_model: int,
        n_layers: int,
        vocab_size: int,
        d_state: int = 16,
        expand: int = 2,
        dt_rank: Union[int, str] = "auto",
        d_conv: int = 4,
        pad_vocab_size_multiple: int = 8,
        conv_bias: bool = True,
        bias: bool = False
    ):
        d_inner = expand * d_model
        if dt_rank == "auto":
            dt_rank = math.ceil(d_model/16)
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += (pad_vocab_size_multiple - vocab_size % pad_vocab_size_multiple)

        return ModelArgs(
            d_model,
            d_inner,
            n_layers,
            vocab_size,
            d_state,
            expand,
            dt_rank,
            d_conv,
            pad_vocab_size_multiple,
            conv_bias,
            bias
        )


class Mamba(nn.Module):
    args: ModelArgs
    
    def setup(self) -> None:
        self.embedding = nn.Embed(self.args.vocab_size, self.args.d_model)
        self.layers = nn.Sequential([ResidualBlock(self.args) for i in range(self.args.n_layers)])
        self.norm_f = RMSNorm(self.args.d_model)
    
    def __call__(self, input_ids: jnp.ndarray) -> jnp.ndarray:
        x = self.embedding(input_ids)
        x = self.layers(x)
        x = self.norm_f(x)
        # use same weights to convert back to token dimensions
        out = self.embedding.attend(x)
        return out 

    @staticmethod
    def from_pretrained():
        raise NotImplementedError


class ResidualBlock(nn.Module):
    args: ModelArgs

    def setup(self) -> None:
        self.mixer = MambaBlock(self.args)
        self.norm = RMSNorm(self.args.d_model)

    def __call__(self, x) -> jnp.ndarray:
        return self.mixer(self.norm(x)) + x


class MambaBlock(nn.Module):
    args: ModelArgs
    
    def setup(self):
        self.conv1d = nn.Conv(
            features=self.args.d_inner,
            use_bias=self.args.conv_bias,
            kernel_size=self.args.d_conv,
            feature_group_count=self.args.d_inner,
            padding=self.args.d_conv - 1
        )
        
        self.in_proj = nn.Dense(self.args.d_inner * 2, use_bias=self.args.bias)
        # project x to corresponding dt, B, C
        self.x_proj = nn.Dense(self.args.dt_rank + self.args.d_state * 2, use_bias=False)
        
        self.dt_proj = nn.Dense(self.args.d_inner, use_bias=True)
        A = repeat(jnp.arange(1, self.args.d_state + 1), "n -> d n", d=self.args.d_inner)
        self.A_log = self.param("A_log", lambda rng, x: jnp.log(x), A)
        self.D = self.param("D", nn.initializers.ones, self.args.d_inner)
        self.out_proj = nn.Dense(self.args.d_model, use_bias=self.args.bias)
        
    def select_scan(self, u: jnp.ndarray, delta: jnp.ndarray, A: jnp.ndarray, B: jnp.ndarray, C: jnp.ndarray, D: jnp.ndarray):
        b, l, d_inner = u.shape
        n = A.shape[1] # 
        
        deltaA = jnp.exp(einsum(delta, A, "b l d_in, d_in n -> b l d_in n"))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
        
        x = jnp.zeros((b, self.args.d_inner, n)) # , device=deltaA.devices

        def compute_scan(x, i):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            return x, y
        _, y = jax.lax.scan(compute_scan, x, jnp.arange(l))
        y = y.swapaxes(0, 1)
        y = y + u * D
    
        return y
    
    def ssm(self, x: jnp.ndarray):
        d_inner, n = self.A_log.shape
        A = -jnp.exp(self.A_log)
        D = self.D
        
        mid = self.x_proj(x)  # 96
        delta, B, C = jnp.split(mid, [self.args.dt_rank, self.args.d_state + self.args.dt_rank], axis=-1)
        delta = nn.softplus(self.dt_proj(delta))
        y = self.select_scan(x, delta, A, B, C, D)
        
        return y

    def __call__(self, x: jnp.ndarray):
        b, l, d = x.shape
        x_res = self.in_proj(x)
        x, res = jnp.split(x_res, [self.args.d_inner], axis=-1)
        
        x = self.conv1d(x)[:,:l]
        x = nn.silu(x)
        y = self.ssm(x)
        y = y * nn.silu(res)
        return self.out_proj(y)

class RMSNorm(nn.Module):
    d_model: int
    eps: float = 1e-15

    def setup(self) -> None:
        self.weight = self.param("weight", nn.initializers.ones, (1,))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x_squared = jnp.pow(x, 2).mean(axis=-1, keepdims=True)
        rsqrt_term = jnp.power(jnp.sqrt(x_squared + self.eps), -1)
        return x * rsqrt_term * self.weight


if __name__ == "__main__":
    seed = 0
    rng = jax.random.PRNGKey(seed)
    BATCH_SIZE = 1

    args = ModelArgs.init(d_model=1024, n_layers=48, vocab_size=50277)

    # norm_layer = RMSNorm(4)
    # x = jnp.zeros((BATCH_SIZE, 4))
    # norm_params = norm_layer.init(rng, x)
    # y = norm_layer.apply(norm_params, x)

    # mamba_block = MambaBlock(args)
    # # input shape is (BATCH_SIZE, l, d)
    # length = 4
    # d = 1024
    # x = jnp.zeros((BATCH_SIZE, length, d))
    # mamba_block_params = mamba_block.init(rng, x)
    # y = mamba_block.apply(mamba_block_params, x)

    # residual_block = ResidualBlock(args)
    # residual_block_params = residual_block.init(rng, x)
    # y = residual_block.apply(residual_block_params, x)

    input_ids = jnp.zeros((1, 2), dtype=jnp.int32)
    mamba = Mamba(args)
    mamba_params = mamba.init(rng, input_ids)
    y = mamba.apply(mamba_params, input_ids)