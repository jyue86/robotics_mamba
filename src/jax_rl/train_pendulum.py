import jax
import jax.numpy as jnp
import gymnax

from animation import PendulumVisualizer


def visualize_example():
    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_act, key_step = jax.random.split(rng, 4)

    # Instantiate the environment & its settings.
    env, env_params = gymnax.make("Pendulum-v1")
    print(env_params)
    print("Observation space:", env.observation_space(env_params).low, env.observation_space(env_params).high)
    print("Observation space:", env.observation_space(env_params).shape)
    print("Action space:", env.action_space().low, env.action_space().high)
    print("Action space:", env.action_space().shape)

    # Reset the environment.
    obs, state = env.reset(key_reset, env_params)
    print("Observation:", obs)
    print("State:", state)

    # Sample a random action.
    action = env.action_space(env_params).sample(key_act)
    jax.debug.print("{}", action)

    # Perform the step transition.
    n_obs, n_state, reward, done, _ = env.step(key_step, state, action, env_params)
    state_seq, reward_seq = [], []
    rng, rng_reset = jax.random.split(rng)
    obs, env_state = env.reset(rng_reset, env_params)
    all_thetas = [env_state.theta]

    jit_step = jax.jit(env.step)

    while True:
        state_seq.append(env_state)
        rng, rng_act, rng_step = jax.random.split(rng, 3)
        action = env.action_space(env_params).sample(rng_act)
        next_obs, next_env_state, reward, done, info = jit_step(
            rng_step, env_state, action, env_params
        )
        reward_seq.append(reward)
        all_thetas.append(next_env_state.theta)
        if done:
            break
        else:
            obs = next_obs
        env_state = next_env_state

    cum_rewards = jnp.cumsum(jnp.array(reward_seq))
    visualizer = PendulumVisualizer(all_thetas)
    visualizer.visualize()

if __name__ == "__main__":
    visualize_example()