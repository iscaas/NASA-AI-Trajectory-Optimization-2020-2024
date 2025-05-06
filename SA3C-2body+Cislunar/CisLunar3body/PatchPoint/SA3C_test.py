# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import argparse
import os
import random
import time
from distutils.util import strtobool

# import gymnasium as gym
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

######################################
####### Importing for NASA aslgo #####
######################################
from Spacecraft_Env_3 import  SpacecraftEnv
import csv
from configs_3 import args
import os.path
import time
from datetime import date
######################################

# Check for available GPUs
if torch.cuda.is_available():
    print("Torch CUDA is available. You have", torch.cuda.device_count(), "GPU(s) on your system.")
else:
    print("No Torch CUDA-enabled GPU found.")

num_gpus = torch.cuda.device_count()
for i in range(num_gpus):
    gpu = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {gpu.name} (CUDA Compute Capability {gpu.major}.{gpu.minor})")
    print(f"   Total Memory: {gpu.total_memory / (1024**2)} MB")
    print(f"   CUDA Cores: {gpu.multi_processor_count}\n")


def make_env(env_id=args.env_id, seed=args.seed, idx=None, capture_video=args.capture_video, run_name=None):
    # def thunk():
    #     env = gym.make(env_id)
    #     env = gym.wrappers.RecordEpisodeStatistics(env)
    #     if capture_video:
    #         if idx == 0:
    #             env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    #     env.seed(seed)
    #     env.action_space.seed(seed)
    #     env.observation_space.seed(seed)
    #     return env
    
    def thunk2():
        gym.envs.register(
        id='SpacecraftEnv-v0',
        entry_point='Spacecraft_Env_3:SpacecraftEnv',
        kwargs={'args': args, 'pre_tr_path': ""},
        max_episode_steps=args.max_steps_one_ep
        )
        env = gym.make('SpacecraftEnv-v0')
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk2

def create_folders(args):
    today = date.today()
    day = today.strftime('%b-%d-%Y')
    txt_dir = f'{int(time.time())}'
    dir_name = day + "_" + txt_dir +'/'

    current_folder = os.getcwd()
    parent_folder = os.path.dirname(current_folder)
    output_dir = os.path.join(parent_folder, 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    weights_dir = os.path.join(output_dir, 'weights_2')
    logs_dir = os.path.join(output_dir, 'logs_2')
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    models_dir = os.path.join(weights_dir, dir_name)
    logdir = os.path.join(logs_dir, dir_name)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)
    if args.single_weight_test == 0:
        temp_dir_for_saving_weights_folder_name = os.path.join(output_dir, 'temp_mainfolder_weights.dat')
    with open(temp_dir_for_saving_weights_folder_name, 'w+') as file1:
            file1.write(models_dir)

    return models_dir, logdir


# ALGO LOGIC: initialize agent here:
class AttentionNetwork(nn.Module):
    def __init__(self, env, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim + hidden_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 128)
        self.sigmoid = nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)

    def forward(self, af, cf):
        x = torch.cat([af, cf], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        x = x * 2
        return x





# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)
        self.reset_parameters_softQ()

    def reset_parameters_softQ(self):
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)
        nn.init.kaiming_normal_(self.fc4.weight)


    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x1 = F.relu(self.fc3(x))
        x = self.fc4(x1)
        q_features = x1
        return x, q_features
    



LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc_mean = nn.Linear(128, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(128, np.prod(env.single_action_space.shape))
        self.reset_parameters_actor()
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )
                

    def reset_parameters_actor(self):
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)
        nn.init.kaiming_normal_(self.fc_mean.weight)
        nn.init.kaiming_normal_(self.fc_logstd.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats
        actor_features = x

        return mean, log_std, actor_features
    
    def get_action(self, x, seed=0, testing_weights=0):
        if testing_weights == 1:
            torch.manual_seed(seed)
        mean, log_std, actor_features = self(x) 
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean, actor_features
    
    
    def Features_to_actions(self, actor_features):
        mean = self.fc_mean(actor_features)
        log_std = self.fc_logstd(actor_features)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std
    
    def get_action_Features_to_actions(self, actor_features):
        mean, log_std  = self.Features_to_actions(actor_features)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean



import torch
import zipfile
import io

def save_models_to_zip(actor_state_dict, qf1_state_dict, Att_net_state_dict, zip_file_path):
    with zipfile.ZipFile(zip_file_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
        for name, state_dict in [('actor_state_dict.pth', actor_state_dict),
                                 ('qf1_state_dict.pth', qf1_state_dict),
                                 ('Att_net_state_dict.pth', Att_net_state_dict)]:
            with io.BytesIO() as bytes_io:
                torch.save(state_dict, bytes_io)
                zipf.writestr(name, bytes_io.getvalue())

def load_models_from_zip(zip_file_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zipf:
        return {name: torch.load(io.BytesIO(zipf.read(name))) for name in zipf.namelist()}       



if __name__ == "__main__":
    #############################################
    RL_algo = args.algo
    testing_weights = args.testing_weights
    testing_converged_weights = args.testing_converged_weights
    single_weight_test = args.single_weight_test

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")


    current_folder = os.getcwd()
    model_path_folder = os.path.join(current_folder, 'Final Weights')
    weight_file = args.single_weight_file + ".zip"
    model_path = os.path.join(model_path_folder, weight_file)

    plots_dir = os.path.join(current_folder, 'plots')

    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video)]) 
    # obs = envs.reset()

    if args.zip_weights:
    ##########################################################################################################
        models_dict = load_models_from_zip(model_path)
        # Loading weights
        actor_file = models_dict.get('actor_state_dict.pth')
        qf1_file = models_dict.get('qf1_state_dict.pth')
        qf2_file = models_dict.get('qf1_state_dict.pth')
        qf1_target_file = models_dict.get('qf1_state_dict.pth')
        qf2_target_file = models_dict.get('qf1_state_dict.pth')
        Att_net_state_dict = models_dict.get('Att_net_state_dict.pth')
    else:
        #####################################################
        actor_file = f"actor.dat"
        att_net_file = f"Att_net.dat"
        qf1_file = f"qf1.dat"
        qf2_file = f"qf1.dat"
        qf1_target_file = f"qf1.dat"
        qf2_target_file = f"qf1.dat"
    ##################~########################################################################################


    # envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video)]) 
    envs.single_observation_space.dtype = np.float32

    actor = Actor(envs).to(device)
    Att_net = AttentionNetwork(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    if args.zip_weights:
    ##########################################################################################################
        # Load weights for the current episode   
        actor.load_state_dict(actor_file)
        Att_net.load_state_dict(Att_net_state_dict)
        qf1.load_state_dict( qf1_file)
        qf2.load_state_dict(qf2_file)
        qf1_target.load_state_dict( qf1_target_file)
        qf2_target.load_state_dict( qf2_target_file)
    #####################################################
    else:
        # Load weights for the current episode
        actor.load_state_dict(torch.load(os.path.join(model_path_folder, args.single_weight_file, actor_file)))
        Att_net.load_state_dict(torch.load(os.path.join(model_path_folder,args.single_weight_file, att_net_file)))
        qf1.load_state_dict(torch.load(os.path.join(model_path_folder,args.single_weight_file, qf1_file)))
        qf2.load_state_dict(torch.load(os.path.join(model_path_folder, args.single_weight_file, qf2_file)))
        qf1_target.load_state_dict(torch.load(os.path.join(model_path_folder, args.single_weight_file, qf1_target_file)))
        qf2_target.load_state_dict(torch.load(os.path.join(model_path_folder,args.single_weight_file,  qf2_target_file)))

    ##########################################################################################################

    lambda1_q_attention = args.lambda1_q_attention
    Lambda_attention = args.lambda1_q_attention

    # Set the models to evaluation mode
    actor.eval()
    qf1.eval()
    qf2.eval()
    qf1_target.eval()
    qf2_target.eval()

    start_time = time.time()
    episodic_returns = []
    moving_avg_window = 10


    # # Run inference loop
    # obs = envs.reset() 
    # model_path = os.path.join(model_folder, actor_file)

    start_time = time.time()
    episodic_returns = []
    moving_avg_window = 10


    # Run inference loop
    obs = envs.reset() 
    done = False
    steps = 0

    while not done:
        with torch.no_grad():
            steps += 1
            actions, _, _, _ = actor.get_action(torch.Tensor(obs).to(device),args.seed,  args.seed_test_flag)
            actions = actions.detach().cpu().numpy()
            # obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            # action = actor(obs_tensor)
            next_obs, rewards, done, infos = envs.step(actions)
            obs = next_obs

    envs.close()
    print("All EPISODES DONE ! ")
