import jax
import jax.numpy as jnp
import gymnax

from animation import PendulumVisualizer


def visualize_example():
    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_act, key_step = jax.random.split(rng, 4)

    # Instantiate the environment & its settings.
    env, env_params = gymnax.make("Pendulum-v1")

    # Perform the step transition.
    rng, rng_reset = jax.random.split(rng)
    obs, env_state = env.reset(rng_reset, env_params)
    all_thetas = [env_state.theta]

    jit_step = jax.jit(env.step)

    while True:
        rng, rng_act, rng_step = jax.random.split(rng, 3)
        action = env.action_space(env_params).sample(rng_act)
        next_obs, next_env_state, reward, done, info = jit_step(
            rng_step, env_state, action, env_params
        )
        all_thetas.append(next_env_state.theta)
        if done:
            break
        else:
            obs = next_obs
        env_state = next_env_state

    visualizer = PendulumVisualizer(all_thetas)
    visualizer.visualize()

if __name__ == "__main__":
    visualize_example()