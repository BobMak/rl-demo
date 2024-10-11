import gymnasium as gym

from stable_baselines3 import PPO

env = gym.make("CartPole-v1", render_mode="human")

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_0)

obs, _ = env.reset()
for i in range(500):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    # VecEnv resets automatically
    # if terminated or truncated:
    #     obs, _ = env.reset()

env.close()