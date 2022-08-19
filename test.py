import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

env_name = "ALE/Breakout-v5"
# env = make_atari_env(env_name, n_envs=4, seed=0)
# env = VecFrameStack(env, n_stack=4)
env = gym.make(env_name)

save_path = "./model/breakout_ppo3.pkl"
model = PPO.load(save_path, env=env)
print(model)

obs = env.reset()
done = False
score = 0

while not done:
    action, _ = model.predict(observation=obs)
    state, reward, done, info = env.step(action=action)
    score += reward
    env.render()
env.close()

print("score : {}".format(score))