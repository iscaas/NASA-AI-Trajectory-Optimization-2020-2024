
import stable_baselines3
from stable_baselines3 import SAC
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import os
from Spacecraft_Env import  Spacecraft_Env
import time
#from enviornment import Enviornment
import numpy as np
import tensorflow as tf 
import os.path
import time
from datetime import date

tf.debugging.set_log_device_placement(True)

today = date.today()
day = today.strftime('%b-%d-%Y')
txt_dir = f'{int(time.time())}'
dir_name = day + "_" + txt_dir +'/'
current_folder = os.getcwd()
models_dir1 = os.path.join(current_folder, 'weights')
models_dir = os.path.join(models_dir1, dir_name)

if not os.path.exists(models_dir1):
	os.makedirs(models_dir1)
if not os.path.exists(models_dir):
	os.makedirs(models_dir)


env = Spacecraft_Env()
env.reset()


model = SAC('MlpPolicy', env, learning_rate=0.0003, buffer_size=10000, gamma=0.99, tau=0.005, train_freq=1,  verbose=1)


n_steps = 2000*50000
TIMESTEPS = 100
with tf.device('/GPU:0'):
    for step in range (n_steps):
        print("Step {}".format(step + 1))
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, log_interval=1, tb_log_name='SAC')
        file_name= str(step)+"_"+str(int(time.time()))
        model.save(f"{models_dir}/{file_name}")
    

print("All EPISODES DONE ! ")
	