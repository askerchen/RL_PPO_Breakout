'''
使用stable_baseline3在breakout环境下训练PPO模型
'''
import gym
import stable_baselines3
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import PPO

env_name = "ALE/Breakout-v5"
# env = make_atari_env(env_name, n_envs=4, seed=0)
# # Frame-stacking with 4 frames
# env = VecFrameStack(env, n_stack=4)

env = gym.make(env_name)

model = PPO(
    policy='CnnPolicy',
    env=env,
    verbose=1,
    tensorboard_log='./tensorboard/breakout',
    n_steps=128,
    n_epochs=4,
    batch_size=256,
    learning_rate=2.5e-4,
    clip_range=0.1,
    vf_coef=0.5,
    ent_coef=0.01
)
model.learn(total_timesteps=int(2e5))  # agent走timesteps步

save_path = "./model/breakout_ppo3.pkl"
model.save(save_path)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=False)
print(mean_reward, std_reward)

# Enjoy trained agent
# obs = env.reset()
# for i in range(1000):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, rewards, dones, info = env.step(action)
#     env.render()
# env.close()