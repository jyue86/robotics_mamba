import torch
from flax.training import train_state, common_utils
import jax.numpy as jnp
import jax
import flax.linen as nn
import optax
import os
import math

from tqdm.auto import tqdm
import torch.utils.data as data


random_seed = 42
rng = jax.random.PRNGKey(random_seed)


if __name__ == "__main__":
    print("Hello World!")
