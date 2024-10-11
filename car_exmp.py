import gymnasium as gym

from stable_baselines3 import PPO, SAC

env = gym.make("CarRacing-v2", render_mode="rgb_array")

model = SAC("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
env = gym.make("CarRacing-v2", render_mode="human")

obs, _ = env.reset()
for i in range(500):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    # VecEnv resets automatically
    # if terminated or truncated:
    #     obs, _ = env.reset()

env.close()