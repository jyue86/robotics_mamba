import jax
import jax.numpy as jnp
from flax.struct import dataclass
import gymnax
from typing import NamedTuple
from flax.training.train_state import TrainState
from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper
from jax_rl.net import DiscretePolicy
from animation import CartPoleVisualizer
import optax


@dataclass
class Transition:
    done: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

config = {
    "LR": 2.5e-4,
    "NUM_ENVS": 4,
    "NUM_STEPS": 128,
    "TOTAL_TIMESTEPS": 5e5,
    "UPDATE_EPOCHS": 4,
    "NUM_MINIBATCHES": 4,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.01,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "ACTIVATION": "tanh",
    "ENV_NAME": "CartPole-v1",
    "ANNEAL_LR": True,
    "HID_DIMMS": [32, 32, 32]
}


@jax.jit
def make_train_cartpole(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    # Instantiate the environment & its settings.
    env, env_params = gymnax.make("CartPole-v1")
    env.action
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

    def train(rng):
        # init policy
        policy = DiscretePolicy(
            hid_dims=config["HID_DIMS"],
            out_dim=env.action_space(env_params).n,
        )
        rng, _rng = jax.random.split(rng)
        init_x = jax.zeros(env.observation_space(env_params).shape)
        model_params = policy.init(_rng, init_x)
        tx = optax.adam(learning_rate=1e-3)
        train_state = TrainState.create(
            apply_fn=policy.apply,
            params=model_params,
            tx=tx
        )

        # Reset the environment.
        rng, _rng = jax.random.split(rng, 2)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        def _update_step(runner_state, _):
            # collect trajectory data
            def _env_step(runner_state, _):
                train_state, env_state, last_obs, rng = runner_state
                rng, _rng = jax.random.split(rng)
                action_dist = policy.apply(train_state.params, last_obs)
                action = action_dist.sample(seed=_rng)
                log_prob = action_dist.log_prob(action)

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obs, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(rng_step, env_state, action, env_params)
                transition = Transition(done, action, reward, log_prob, last_obs, info)
                runner_state = (train_state, env_state, obs, rng)
                return runner_state, transition 

            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config["NUM_STEPS"])   # n_episodes, n_envs, transition
            train_state, env_state, last_obs, rng = runner_state
            
            def _update_epoch(update_state, unused):
                # update model params
                
                def _update_minbatch(update_state, batch_info):
                    # FIXME: shape of batch_info 
                    _, cum_rewards = jax.lax.scan(
                        lambda x, y: (x + y.reward * config["GAMMA"], x + y.reward * config["GAMMA"]), 
                        0.0, 
                        batch_info,
                        length=config["NUM_STEPS"]
                    )

                train_state, traj_batch, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch,)
                shuffled_batch = jax.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]
            
            runner_state = (train_state, env_state, obs, rng)
            return runner_state, metric
            
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obs, rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}
    
    return train
    

if __name__ == "__main__":
    make_train_cartpole()
    # visualizer = CartPoleVisualizer(all_states)
    # visualizer.visualize()