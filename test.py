#%%
from stable_baselines3 import PPO
import gym
import time
import wandb
from wandb.integration.sb3 import WandbCallback
import argparse
import tensorboard

from clearml import Task
import os

os.environ['WANDB_API_KEY'] = 'e08ae8b88ab980fc97e670c026c961dd757a66f8'

task = Task.init(project_name='Pendulum-v1/Dean', task_name='Experiment1',output_uri=True)
task.set_base_docker('deanis/robosuite:py3.8-2')
task.execute_remotely(queue_name="default")

env = gym.make('Pendulum-v1',g=9.81)
#env = gym.make('CartPole-v1')
run = wandb.init(project="sb3_pendulum_demo",sync_tensorboard=True)
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=f"runs/{run.id}")
wandb_callback = WandbCallback(model_save_freq=1000,
                                model_save_path=f"models/{run.id}",
                                verbose=2,
                                )

timesteps = 10000
for iter in range(10):
    model.learn(total_timesteps=timesteps, callback=wandb_callback, progress_bar=True, reset_num_timesteps=False,tb_log_name=f"runs/{run.id}")
    model.save(f"models/{run.id}/{timesteps*(iter+1)}")

#Test the trained model
obs = env.reset()
for i in range(1000):
    action, _ = model.predict(obs,deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    time.sleep(0.025)
    if done:
        env.reset()
# %%
