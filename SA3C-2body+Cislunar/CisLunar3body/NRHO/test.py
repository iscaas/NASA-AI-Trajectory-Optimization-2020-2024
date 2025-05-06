
import stable_baselines3
from stable_baselines3 import SAC
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import os

import time
# import gym
import gymnasium as gym
#from enviornment import Enviornment
import numpy as np
import tensorflow as tf 
import torch
import os.path
import time
from datetime import date
# from gym.wrappers import TimeLimit
tf.debugging.set_log_device_placement(True)
import argparse
from configs_3 import args
import matlab.engine
from enviornment_3 import Enviornment 

from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3 import HER
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from Spacecraft_Env_3 import  SpacecraftEnv
import csv




current_folder = os.getcwd()
weights_dir = os.path.join(current_folder, 'Final Weights')
weight_file = args.single_weight_file + ".zip"
model_path = os.path.join(weights_dir, weight_file)


gym.envs.register(
    id='SpacecraftEnv-v0',
    entry_point='Spacecraft_Env_3:SpacecraftEnv',
    kwargs={'args': args, 'pre_tr_path': model_path}
)
env = gym.make('SpacecraftEnv-v0')
# env = SpacecraftEnv(args, model_path)
env.seed(10)
env.reset()

print("Load Weights")
model = SAC.load(model_path, env =env , seed=10)


episodes = 500
with tf.device('/GPU:0'):
    for ep in range (episodes):
        obs = env.reset()
        done = False
        steps = 0
        model = SAC.load(model_path, env =env )
        env.seed(10)
        while not done:
            steps = steps + 1
            action = model.predict(obs)
            obs, reward, done, info = env.step(action[0])
        
env.close()
print("All EPISODES DONE ! ")