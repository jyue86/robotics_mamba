import jax
import gymnax

from animation import CartPoleVisualizer


def train_cartpole():
    rng = jax.random.PRNGKey(0)

    # Instantiate the environment & its settings.
    env, env_params = gymnax.make("CartPole-v1")
    print(env_params)
    print("Observation space:", env.observation_space(env_params).low, env.observation_space(env_params).high)
    print("Observation space:", env.observation_space(env_params).shape)
    print("Action space:", env.action_space())
    print("Action space:", env.action_space().n)
    print("Action space:", env.action_space().shape)

    # Reset the environment.
    rng, key_reset = jax.random.split(rng, 2)
    obs, env_state = env.reset(key_reset, env_params)
    print("Observation:", obs)
    print("State:", env_state)

    jit_step = jax.jit(env.step)
    all_states = [(env_state.x, env_state.theta)]

    while True:
        rng, rng_act, rng_step = jax.random.split(rng, 3)
        action = jax.random.bernoulli(rng_act, 0.5)
        next_obs, next_env_state, reward, done, info = jit_step(
            rng_step, env_state, action, env_params
        )
        all_states.append((next_env_state.x, next_env_state.theta))

        if done:
            break
        else:
            obs = next_obs
        env_state = next_env_state
    
    visualizer = CartPoleVisualizer(all_states)
    visualizer.visualize()

if __name__ == "__main__":
    train_cartpole()