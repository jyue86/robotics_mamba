from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax


class MLP(nn.Module):
    hid_dims: tuple
    out_dim: int
    act: nn.Module = nn.relu
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        layers = []
        for i in range(len(self.hid_dims)):
            layers.append(nn.Dense(self.hid_dims[i], dtype=self.dtype))
            layers.append(self.act)
            
        layers.append(nn.Dense(self.out_dim, dtype=self.dtype))
        self.layers = nn.ModuleList(layers)
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.layers(x)


class Policy(nn.Module):
    hid_dims: tuple
    hid_dims: tuple
    out_dim: int
    act: nn.Module = nn.relu
    
    
    def _setup(self):
        self.mlp = MLP(self.hid_dims, self.out_dim, self.act)
        self.log_std = self.param('log_std', jnp.zeros, (self.out_dim,))
        
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.mlp(x)