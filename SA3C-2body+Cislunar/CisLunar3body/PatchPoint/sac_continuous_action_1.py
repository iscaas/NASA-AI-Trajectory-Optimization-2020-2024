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

    if testing_weights == 0:
        max_steps_one_ep = args.max_steps_one_ep
        max_nu_ep = args.max_nu_ep
        weights_save_steps = args.weights_save_steps
        buffer_size = args.buffer_size
        gamma = args.gamma

        
        models_dir, logsdir = create_folders(args)

        torch.cuda.empty_cache()
        device_1 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        #############################################
        # args = parse_args()
        run_name = f"CRL_ATT__seed{args.seed}__SAClmb{args.lambda1_q_attention}__Anetlmd{args.Lambda_attention}__{int(time.time())}"
        if args.track:
            import wandb
            wandb.init(
                project=args.wandb_project_name,
                #entity=args.wandb_entity,
                sync_tensorboard=True,
                config=vars(args),
                name=run_name,
                #monitor_gym=True,
                save_code=True,
            )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

        # TRY NOT TO MODIFY: seeding
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic

        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

        # env setup
        envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
        assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

        max_action = float(envs.single_action_space.high[0])

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
        Att_net_optimizer = optim.Adam(list(Att_net.parameters()), lr=args.Att_lr)

        lambda1_q_attention = args.lambda1_q_attention
        Lambda_attention = args.lambda1_q_attention

        # Automatic entropy tuning
        if args.autotune: 
            target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
            log_alpha = torch.zeros(1, requires_grad=True, device=device)
            alpha = log_alpha.exp().item()
            a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
        else:
            alpha = args.alpha

        envs.single_observation_space.dtype = np.float32
        rb = ReplayBuffer(
            args.buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            device,
            handle_timeout_termination=True,
        )
        start_time = time.time()
        episodic_returns = []
        moving_avg_window = 10

        # TRY NOT TO MODIFY: start the game
        obs = envs.reset()
        for global_step in range(args.total_timesteps):
            # ALGO LOGIC: put action logic here
            if global_step < args.learning_starts:
                actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            else:
                actions, _, _, _ = actor.get_action(torch.Tensor(obs).to(device))
                actions = actions.detach().cpu().numpy()

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, dones, infos = envs.step(actions)

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            for info in infos:
                if "episode" in info.keys() or envs.envs[0].red_flag or envs.envs[0].max_ep_flag:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    # For average return
                    episodic_return = info["episode"]["r"]
                    episodic_returns.append(episodic_return)
                    # Calculate the moving average of episodic returns
                    if len(episodic_returns) >= moving_avg_window:
                        avg_return = np.mean(episodic_returns[-moving_avg_window:])
                        writer.add_scalar("charts/avg_return", avg_return, global_step)
                    break

            # Save the weights for actor, critic and target critic models 
            if "episode" in info.keys() or (global_step % 500 == 0):
                # torch.save(actor.state_dict(), f'{models_dir}/actor_{global_step}.dat')
                # torch.save(qf1.state_dict(), f'{models_dir}/qf1_{global_step}.dat')
                # torch.save(Att_net.state_dict(), f'{models_dir}/Att_net_{global_step}.dat')

                zip_file_name = f"{models_dir}/models_{global_step}_{info['ep_nu']}_{info['ep_length']}.zip"
                save_models_to_zip(actor.state_dict(), qf1.state_dict(), Att_net.state_dict(), zip_file_name)

            # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
            real_next_obs = next_obs.copy()
            for idx, d in enumerate(dones):
                if d:
                    real_next_obs[idx] = infos[idx]["terminal_observation"]
            rb.add(obs, real_next_obs, actions, rewards, dones, infos)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

            # ALGO LOGIC: training.
            if global_step > args.learning_starts:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    next_state_actions, next_state_log_pi, _, _ = actor.get_action(data.next_observations)
                    qf1_next_target, _ = qf1_target(data.next_observations, next_state_actions)
                    qf2_next_target, _ = qf2_target(data.next_observations, next_state_actions)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

                qf1_a_values,_ = qf1(data.observations, data.actions)
                qf2_a_values,_ = qf2(data.observations, data.actions)
                qf1_loss = F.mse_loss(qf1_a_values.view(-1), next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values.view(-1), next_q_value)
                qf_loss = qf1_loss + qf2_loss

                ###################################################################################################################################################
                # FOR ATTENTION SAC Q LOSS:

                # Find actor features and action from actor features
                with torch.no_grad():
                    _, _, _, actor_Features_s = actor.get_action(data.observations)
                    action_aF_s, log_pi_aF_s, _ = actor.get_action_Features_to_actions(actor_Features_s)

                # Find Q1 attention:  Q1(obs, attention_action_1) where attention_action_1 = Actor_feature_net(att_net(actor_features, qf1_features) * actor_features)
                q1_af_s , qf1_features = qf1(data.observations, action_aF_s)
                Att_net_out_1 = Att_net(actor_Features_s, qf1_features)
                attention_features_1 = torch.mul(actor_Features_s, Att_net_out_1)
                with torch.no_grad():
                    attention_action_1, log_pi_att_aF_s_1, _ = actor.get_action_Features_to_actions(attention_features_1)
                q1_attention , _ = qf1(data.observations, attention_action_1)
                
                # Find Q2 attention:  Q2(obs, attention_action_2) where attention_action_2 = Actor_feature_net(att_net(actor_features, qf2_features) * actor_features)
                q2_af_s , qf2_features = qf2(data.observations, action_aF_s)
                Att_net_out_2 = Att_net(actor_Features_s, qf2_features)
                attention_features_2 = torch.mul(actor_Features_s, Att_net_out_2)
                with torch.no_grad():
                    attention_action_2, log_pi_att_aF_s_2, _ = actor.get_action_Features_to_actions(attention_features_2)
                q2_attention , _ = qf1(data.observations, attention_action_2)

                # For attention Net Loss regularization, Calculate the mean of log prob from actor with input of attention features 1 ,2 
                Mean_log_pi_attention_net_afqf1_afqf2_features = torch.mean(torch.cat((log_pi_att_aF_s_1.view(-1), log_pi_att_aF_s_2.view(-1))))

                # Minimum of Q values and min of attention q values
                q_attention  = torch.min(q1_attention.view(-1), q2_attention.view(-1))
                q_f_a_values = torch.min(qf1_a_values.view(-1), qf2_a_values.view(-1))

                # check which critic value is better, either from attention one or other one ann then calculate the attention loss
                Q_values_max = torch.max(q_attention, q_f_a_values)
                q_attention_loss = F.mse_loss(Q_values_max, next_q_value)

                # Calculate the final Q loss for Q networks
                q_loss = (lambda1_q_attention * qf_loss) + ((1-lambda1_q_attention) * q_attention_loss)

                ###################################################################################################################################################

                # Update the Q Network weights
                q_optimizer.zero_grad()
                q_loss.backward()
                q_optimizer.step()

                

                ###################################################################################################
                # For Attention Network Loss, Find the actor loss before updating actor weights ie actor_loss_old
                _, _, _, actor_Features_s  = actor.get_action(data.observations)
                action_aF_s, log_pi_aF_s, _ = actor.get_action_Features_to_actions(actor_Features_s)
                q1_af_s , qf1_features = qf1(data.observations, action_aF_s)
                q2_af_s , qf2_features = qf2(data.observations, action_aF_s)
                q_min  = torch.min(q1_af_s, q2_af_s).view(-1)
                actor_loss_old = ( log_pi_aF_s.view(-1) - q_min).mean()
                ###################################################################################################


                if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                    for _ in range(
                        args.policy_frequency
                    ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                        pi, log_pi, _, _ = actor.get_action(data.observations)
                        qf1_pi,_ = qf1(data.observations, pi)
                        qf2_pi,_ = qf2(data.observations, pi)
                        min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
                        actor_loss = ((alpha * log_pi.view(-1)) - min_qf_pi).mean()

                        # Update the Actor Network weights
                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        actor_optimizer.step()

                        ###################################################################################################
                        ##   ATTENTION NETWORK LOSS AND WEIGHTS UPDATE

                        actor_loss_old_detached = actor_loss_old.detach()
                        actor_loss_new = 0
                        # For Attention Network Loss, Find the actor loss after updating actor weights ie actor_loss_new
                        _, _, _, actor_Features_s  = actor.get_action(data.observations)
                        action_aF_s, log_pi_aF_s, _ = actor.get_action_Features_to_actions(actor_Features_s)
                        q1_af_s , qf1_features = qf1(data.observations, action_aF_s)
                        q2_af_s , qf2_features = qf2(data.observations, action_aF_s)
                        q_min  = torch.min(q1_af_s, q2_af_s).view(-1)
                        actor_loss_new = ( log_pi_aF_s.view(-1) - q_min).mean()
                        

                        torch.autograd.set_detect_anomaly(True)
                        # Calculate final Attention Network Loss with regularization term added
                        Attention_network_loss = Lambda_attention * (torch.tanh (actor_loss_new - actor_loss_old_detached)) + (1-Lambda_attention) * (Mean_log_pi_attention_net_afqf1_afqf2_features)

                        # Attention_network_loss = Lambda_attention * (torch.tanh (actor_loss_new)) + (1-Lambda_attention) * (Mean_log_pi_attention_net_afqf1_afqf2_features)

                        # Update the Attention Network Wrights
                        Att_net_optimizer.zero_grad()
                        Attention_network_loss.backward()
                        Att_net_optimizer.step()
                        ###################################################################################################


                        if args.autotune:
                            with torch.no_grad():
                             _, log_pi, _ ,_= actor.get_action(data.observations)
                            alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                            a_optimizer.zero_grad()
                            alpha_loss.backward()
                            a_optimizer.step()
                            alpha = log_alpha.exp().item()


                # update the target networks
                if global_step % args.target_network_frequency == 0:
                    for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                    for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                    writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                    writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                    writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                    writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                    writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                    writer.add_scalar("losses/alpha", alpha, global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                    if args.autotune:
                        writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

        envs.close()
        writer.close()


    ####################################################################################################################################
    ####################### For Testing the weights ####################################################################################
    ####################################################################################################################################

    if testing_weights == 1:
        if single_weight_test == 0:
            import matlab.engine
            from enviornment_3 import Enviornment 
            csv_infer_nu = args.csv_infer_nu
            
            current_folder = os.getcwd()
            parent_folder = os.path.dirname(current_folder)
            output_dir = os.path.join(parent_folder, 'outputs')
            csv_dir = os.path.join(output_dir, 'csv')
            os.makedirs(output_dir, exist_ok=True)
            weights_dir = os.path.join(output_dir, 'weights_2')
            plots_dir = os.path.join(output_dir, 'plots_2')

            temp_csv_file = os.path.join(csv_dir, 'temp_csv_files_inference', 'temp_mainfolder'+ csv_infer_nu + '.dat')
            temp_csv_file_weights = os.path.join(csv_dir, 'temp_csv_files_inference', 'temp_mainfolder_weights'+ csv_infer_nu + '.dat')
            temp_csv_file_only = os.path.join(csv_dir, 'temp_csv_files_inference', 'temp'+ csv_infer_nu + '.dat')
            if not os.path.exists(temp_csv_file):
                open(temp_csv_file, 'w').close()  # Create the file if it doesn't exist
            if not os.path.exists(temp_csv_file_weights):
                open(temp_csv_file_weights, 'w').close()  # Create the file if it doesn't exist
            if not os.path.exists(temp_csv_file_only):
                open(temp_csv_file_only, 'w').close()  # Create the file if it doesn't exist
            
            
            # models_dir = os.path.join(weights_dir, "Apr-12-2024_1712963287/")


            if args.Manual_weight_flag == 0:
                with open(temp_csv_file, 'r') as file11:
                    file_path_folder = file11.read()
                output_file_path_1 = file_path_folder
                with open(temp_csv_file_weights, 'r') as file12:
                    weights_file_path_folder = file12.read()
                model_folder = weights_file_path_folder
            else:
                output_file_path_1 = os.path.join(plots_dir, args.Manual_plot_folder)
                model_folder = os.path.join(weights_dir, args.Manual_weight_folder)
            

            # models_dir = "E:/CisLunar/1st_case_cislunar/Cis_lunar_with_clean_RL/outputs/weights_2/Apr-12-2024_1712963287"
            # weight_files = model_folder
            # Extract episode numbers from file names

            #############################################################################
            def Read_convrged_files (file_path):
                import re
                model_files_func = []
                with open(file_path, 'r') as file:
                    next(file)
                    next(file)
                    line_number = 1
                    for line in file:
                        if line_number % 2 == 1:
                            model_files_22 = re.findall(r'Model : ,([^,]+),', line)
                            model_files_func.append(model_files_22[1])           
                        line_number += 1 
                
                return model_files_func
            
            print ("--------------before reading model_folder-------------------", model_folder )
            model_files = os.listdir(model_folder)
            # print ("--------------after reading model_files-------------------", model_files )

            if testing_converged_weights:
                ####--------------------------------------------------------------------------------------------------------------#####
                with open(temp_csv_file_only, 'r') as file:
                    file_path_folder = file.read()
                file_path = file_path_folder + "/weights_test_results.dat"  
                model_files = Read_convrged_files (file_path)
                filtered_weights = model_files
                print(filtered_weights)
                ####--------------------------------------------------------------------------------------------------------------#####

            #############################################################################
            parts = output_file_path_1.split('\\')
            Training_folder_name = parts[-1]
            print ("--------------output_file_path_1-------------------", output_file_path_1 )
            print ("--------------Training_folder_name-------------------", Training_folder_name )
            # print (filtered_weights)

            today = date.today()
            day, txt_dir = today.strftime('%b-%d-%Y'), f"{int(time.time())}"
            if testing_converged_weights == 0:
                dir_name =  f"{day}_{txt_dir}_Tst_Wght_Tr_Fldr_{Training_folder_name}"
            else:
                dir_name =  f"{day}_{txt_dir}_Tst_Converg_Wght_{Training_folder_name}"
            folder_path = os.path.join(plots_dir, dir_name)
            if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
            successful_episodes = os.path.join(f"{folder_path}/{'Successful_episodes'}")
            if not os.path.exists(successful_episodes):
                    os.makedirs(successful_episodes)
            output_file_path = os.path.join(f"{folder_path}/{'weights_test_results.dat'}")
            if not os.path.exists(output_file_path):
                with open(output_file_path, 'w') as file:
                    file.close
            output_detail_file_path = os.path.join(f"{folder_path}/{'weights_detail_results.dat'}")
            if not os.path.exists(output_detail_file_path):
                with open(output_detail_file_path, 'w') as file:
                    file.close
            info_file_path = os.path.join(f"{folder_path}/{'info.dat'}")
            if not os.path.exists(info_file_path):
                with open(info_file_path, 'w') as file:
                    file.close

            temp1 =  ['Model_folder : ',  model_folder]
            with open(output_file_path, 'a') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(temp1)
                csvfile.close() 

            with open(info_file_path, 'w') as info_file:
                info_file.write("Command-line arguments:\n")
                for arg, value in vars(args).items():
                    info_file.write(f"{arg}: {value}\n")

            if testing_converged_weights == 0:
                with open(temp_csv_file_only, 'w+') as file:
                    file.write(folder_path)

            eng = matlab.engine.start_matlab()
            Enviornment_1 = Enviornment(eng, args)


            best_time = 1000000  
            counter = 0

            ##########################################################################################
            weights_folder = os.path.join(weights_dir, f"{day}_{txt_dir}_zipweights_{Training_folder_name}")
            os.makedirs(weights_folder, exist_ok=True)
            ##########################################################################################
            
            ##########################################################################################
             #####################  For ZIP FOLDER SAVED WEIGHTS and testing_converged_weights == 0
            if args.zip_weights:
                if testing_converged_weights == 0:
                    ep_nu_conv  = [] 
                    with open( os.path.join(output_file_path_1 , "output.dat"), 'r') as file:
                        lines = file.readlines()
                        for line in lines:
                            if ",ep : ," in line:
                                lineaaa = line.split("ep : ,")[1]
                                ep_nu_conv.append(int(lineaaa.split(",")[0]))

                    ep_nu_conv_2 = sorted(set(num + i for num in ep_nu_conv for i in range(-2, 3)))
                    ep_nu_conv_2 = sorted(set(num for num in ep_nu_conv_2 if num > 0 ))#  and num > 548) )
                    filtered_weights = [filename for filename in os.listdir(model_folder) if int(filename.split('_')[-2]) in ep_nu_conv_2]
            
            ##########################################################################################            
            else:
                ###### FOR NON ZIP WEIGHTS SAVED
                episodes = set()  
                if testing_converged_weights:
                    for file in filtered_weights:
                        episode = int(file.split("_")[-1].split(".")[0])
                        episodes.add(episode)
                else:
                    model_files = os.listdir(model_folder)
                    for file in model_files:
                        episode = int(file.split("_")[-1].split(".")[0])
                        episodes.add(episode)

            envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video)]) 
            envs.single_observation_space.dtype = np.float32
        
            ############################################################################
            if args.zip_weights:
                loopFor = sorted(filtered_weights, key=lambda x: int(x.split('_')[-2]), reverse=True)
            else:
                loopFor = sorted(episodes, reverse=True)
                
            for name in loopFor:
                if args.zip_weights:
                    zip_file_name = name 
                    episode = name
                    models_dict = load_models_from_zip(os.path.join(model_folder, zip_file_name))
                    # Loading weights
                    actor_file = models_dict.get('actor_state_dict.pth')
                    qf1_file = models_dict.get('qf1_state_dict.pth')
                    qf2_file = models_dict.get('qf1_state_dict.pth')
                    qf1_target_file = models_dict.get('qf1_state_dict.pth')
                    qf2_target_file = models_dict.get('qf1_state_dict.pth')
                    Att_net_state_dict = models_dict.get('Att_net_state_dict.pth')
                else:
                    episode = name
                ###Iterate over episodes and load corresponding weights
                    # episode = "1904000" 
                    actor_file = f"actor_{episode}.dat"
                    att_net_file = f"Att_net_{episode}.dat"
                    qf1_file = f"qf1_{episode}.dat"
                    qf2_file = f"qf1_{episode}.dat"
                    qf1_target_file = f"qf1_{episode}.dat"
                    qf2_target_file = f"qf1_{episode}.dat"
            ############################################################################    
                episodes = loopFor

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
                Att_net_optimizer = optim.Adam(list(Att_net.parameters()), lr=args.Att_lr)

                if args.zip_weights:
                ############################################################################
                # Load weights for the current episode
                    actor.load_state_dict(actor_file)
                    Att_net.load_state_dict(Att_net_state_dict)
                    qf1.load_state_dict( qf1_file)
                    qf2.load_state_dict(qf2_file)
                    qf1_target.load_state_dict( qf1_target_file)
                    qf2_target.load_state_dict( qf2_target_file)
                else:
                    # Load weights for the current episode
                    actor.load_state_dict(torch.load(os.path.join(model_folder, actor_file)))
                    Att_net.load_state_dict(torch.load(os.path.join(model_folder, att_net_file)))
                    qf1.load_state_dict(torch.load(os.path.join(model_folder, qf1_file)))
                    qf2.load_state_dict(torch.load(os.path.join(model_folder, qf2_file)))
                    qf1_target.load_state_dict(torch.load(os.path.join(model_folder, qf1_target_file)))
                    qf2_target.load_state_dict(torch.load(os.path.join(model_folder, qf2_target_file)))
                ############################################################################

                lambda1_q_attention = args.lambda1_q_attention
                Lambda_attention = args.lambda1_q_attention

                # Set the models to evaluation mode
                actor.eval()
                qf1.eval()
                qf2.eval()
                qf1_target.eval()
                qf2_target.eval()
                Att_net.eval()

                start_time = time.time()
                episodic_returns = []
                moving_avg_window = 10


                # Run inference loop
                obs = envs.reset()  #if counter == 0 else np.array([envs.envs[0].observation])
                done = False
                counter = counter +1 
                steps = 0
                
                if args.zip_weights:
                ############################################################################
                    actor_file = zip_file_name
                ############################################################################                

                model_path = os.path.join(model_folder, actor_file)

                while not done:
                    with torch.no_grad():
                        steps += 1
                        actions, _, _, _ = actor.get_action(torch.Tensor(obs).to(device),args.seed,  args.seed_test_flag) # testing_weights) #shcomand 1 with seed 2 withot seed
                        obs1 = obs
                        actions = actions.detach().cpu().numpy()
                        # obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                        # action = actor(obs_tensor)
                        next_obs, rewards, done, infos = envs.step(actions)
                        obs = next_obs
                        print(f"sh_nu: {args.shell_comand_nu}   Checking Weights folder: {counter}/{len(episodes)}!!!   Episodic Step: {steps}    Best_time(days): {best_time}       Done_ep: {done}  red_flag: {envs.envs[0].red_flag}     Current_ep_time(days): {envs.envs[0].time_in_days}  obs :{infos[0]['e'], infos[0]['a'], infos[0]['i'], }  ")
                        
                        # temp =  ['steps: ', steps, '    ', 'Model : ', actor_file, '    ',  'obs: ', obs1, '    ',  'ecc: ',infos[0]['e'] ,'    ', 'a: ', infos[0]['a'] , '    ', 'inc: ', infos[0]['i'] ,'      ', 'Current_ep_time(days)', envs.envs[0].time_in_days,  'actions', actions]
                        # with open(output_detail_file_path, 'a') as csvfile:
                        #         csvwriter = csv.writer(csvfile)
                        #         csvwriter.writerow(temp)
                        #         csvfile.close()
                        
                        
                        if "episode" in infos[0].keys() and envs.envs[0].red_flag == 0 and envs.envs[0].max_ep_flag == 0:
                            episode_time       = infos[0]["episode"]["t"]
                            episode_reward     = infos[0]["episode"]["r"]
                            episode_length     = infos[0]["episode"]["l"]
                            episode_path       = model_path

                            if episode_time  < best_time:
                                best_time = episode_time 
                                best_model = actor_file

                            temp =  ['Best_Time : ', best_time, '    ', 'Best_Model : ', best_model, '    ',  'Model : ', actor_file, '    ',  'Time: ', episode_time, '    ',
                                    'Reward: ', episode_reward,'    ', 'Episode Length: ', episode_length, '    ', 
                                    'a: ', envs.envs[0].a_history[-2][-1],'    ',  'ecc: ',envs.envs[0].ecc_history[-2][-1],'    ', 'inc: ', envs.envs[0].inclination_history[-2][-1]  , '    ', 
                                    'h: ', envs.envs[0].h_history[-2][-1],'    ', 'hx: ', envs.envs[0].hx_history[-2][-1],'    ', 'hy: ', envs.envs[0].hy_history[-2][-1],'    ', 
                                    'ex: ', envs.envs[0].ex_history[-2][-1], '    ', 'ey: ', envs.envs[0].ey_history[-2][-1], '    ', 
                                    'RAAN: ', envs.envs[0].RAAN_history[-2][-1], '    ', 'argp: ', envs.envs[0].argp_history[-2][-1]]
                            
                            with open(output_file_path, 'a') as csvfile:
                                csvwriter = csv.writer(csvfile)
                                csvwriter.writerow(temp)
                                csvfile.close()

                            Path_Succ_Ep_Weight = os.path.join(f"{successful_episodes}/{'wght_'}{actor_file[0:-4]}_{round(episode_time, 2)}_days__a{round(envs.envs[0].a_history[-2][-1])}_e{round(envs.envs[0].ecc_history[-2][-1], 2)}_i{round(envs.envs[0].inclination_history[-2][-1] , 2)}")
                            
                            if not os.path.exists(Path_Succ_Ep_Weight ):
                                    os.makedirs(Path_Succ_Ep_Weight)
                            
                            target_state_parameters = [0, 1, 2, Enviornment_1.tol_ecc_low, Enviornment_1.tol_ecc_high, Enviornment_1.tol_a_low, Enviornment_1.tol_a_high, Enviornment_1.tol_inc_low, Enviornment_1.tol_inc_high, 9, 10, 11, 12, 13, 14, Enviornment_1.ecc_ms_1_tol_low, Enviornment_1.ecc_ms_1_tol_high, Enviornment_1.a_ms_1_tol_low, Enviornment_1.a_ms_1_tol_high, Enviornment_1.inc_ms_1_tol_low, Enviornment_1.inc_ms_1_tol_high, Enviornment_1.RAAN_tol_low, Enviornment_1.RAAN_tol_high, Enviornment_1.argp_tol_low, Enviornment_1.argp_tol_high]
                            scaled_ecc_param = [0, 1, 2, Enviornment_1.ecc_min_scaled, Enviornment_1.ecc_max_scaled, 5, 6, 7, 8]

                            Enviornment_1.plot_variable("H", envs.envs[0].h_history, Path_Succ_Ep_Weight, envs.envs[0].ep_counter-1, flag_saving_with_no_ep_nu =1,  h_flag=1, tol_h=envs.envs[0].tol_h)
                            Enviornment_1.plot_two_variable("hx_hy","hx","hy", envs.envs[0].hx_history,envs.envs[0].hy_history,  Path_Succ_Ep_Weight, envs.envs[0].ep_counter-1, flag_saving_with_no_ep_nu =1, hx_hy_flag=1, tolhx= envs.envs[0].tol_hx, tolhy= envs.envs[0].tol_hy)
                            Enviornment_1.plot_two_variable("ex_ey","ex","ey", envs.envs[0].ex_history,envs.envs[0].ey_history,  Path_Succ_Ep_Weight, envs.envs[0].ep_counter-1, flag_saving_with_no_ep_nu =1, ex_ey_flag=1, tolex= envs.envs[0].tol_ex, toley= envs.envs[0].tol_ey)
                            Enviornment_1.plot_variable("ecc", envs.envs[0].ecc_history, Path_Succ_Ep_Weight, envs.envs[0].ep_counter-1, flag_ter_values=1, tsp=target_state_parameters, tsp_indexes=[3,4,15,16], flag_saving_with_no_ep_nu =1, flag_ms_1 = envs.envs[0].milestone)
                            Enviornment_1.plot_variable("a", envs.envs[0].a_history, Path_Succ_Ep_Weight, envs.envs[0].ep_counter-1, flag_ter_values=1, tsp=target_state_parameters, tsp_indexes=[5,6,17,18], flag_saving_with_no_ep_nu =1, flag_ms_1 = envs.envs[0].milestone)
                            Enviornment_1.plot_variable("inc", envs.envs[0].inclination_history, Path_Succ_Ep_Weight, envs.envs[0].ep_counter-1, flag_ter_values=1, tsp=target_state_parameters, tsp_indexes=[7,8,19,20], flag_saving_with_no_ep_nu =1, flag_ms_1 = envs.envs[0].milestone)
                            Enviornment_1.plot_two_variable("actions","alpha","beta", envs.envs[0].alpha_history,envs.envs[0].beta_history,   Path_Succ_Ep_Weight, envs.envs[0].ep_counter-1, flag_saving_with_no_ep_nu =1)
                            #Enviornment_1.plot_3d_profile("3d_traj", envs.envs[0].R_gl_history, envs.envs[0].rMoon_history, envs.envs[0].rMoonnf, Path_Succ_Ep_Weight, envs.envs[0].ep_counter, flag_saving_with_no_ep_nu =1)
                            Enviornment_1.plot_two_variable("ex_ey_scaled","ex_sc","ey_sc", envs.envs[0].ex_scaled_history,envs.envs[0].ey_scaled_history,   Path_Succ_Ep_Weight, envs.envs[0].ep_counter-1, flag_saving_with_no_ep_nu =1, ex_ey_flag=1, tolex= Enviornment_1.tol_ex_scaled, toley= Enviornment_1.tol_ex_scaled, scaled=1, target_ex=Enviornment_1.target_state_scaled[3], target_ey=Enviornment_1.target_state_scaled[4])
                            Enviornment_1.plot_variable("ecc_scaled", envs.envs[0].ecc_scaled_history,  Path_Succ_Ep_Weight, envs.envs[0].ep_counter-1, flag_saving_with_no_ep_nu =1, flag_ter_values=1, tsp=scaled_ecc_param, tsp_indexes=[3,4])
                            Enviornment_1.plot_variable("mass", envs.envs[0].mass_history, Path_Succ_Ep_Weight, envs.envs[0].ep_counter-1, flag_saving_with_no_ep_nu =1)
                            Enviornment_1.plot_variable("force", envs.envs[0].force_history, Path_Succ_Ep_Weight, envs.envs[0].ep_counter-1, flag_saving_with_no_ep_nu =1)
                            Enviornment_1.plot_variable("Reward", envs.envs[0].score_detailed_data, Path_Succ_Ep_Weight, envs.envs[0].ep_counter-1, flag_saving_with_no_ep_nu =1)
                            Enviornment_1.plot_variable("Phi", envs.envs[0].phi_history, Path_Succ_Ep_Weight, envs.envs[0].ep_counter-1, flag_saving_with_no_ep_nu =1)
                            Enviornment_1.plot_variable("RAAN", envs.envs[0].RAAN_history, Path_Succ_Ep_Weight, envs.envs[0].ep_counter-1, flag_ter_values=1, tsp=target_state_parameters, tsp_indexes=[21,22,15,16],flag_saving_with_no_ep_nu =1)
                            Enviornment_1.plot_variable("argp", envs.envs[0].argp_history, Path_Succ_Ep_Weight, envs.envs[0].ep_counter-1, flag_ter_values=1, tsp=target_state_parameters, tsp_indexes=[23,24,17,18],flag_saving_with_no_ep_nu =1)
                            Enviornment_1.plot_variable("nurev", envs.envs[0].nurev_history, Path_Succ_Ep_Weight, envs.envs[0].ep_counter-1, flag_saving_with_no_ep_nu =1)

                print("#"*100)
                print("#"*100)
                print(counter, " number of weights checked!!")
                print("#"*100)
                print("#"*100)

                envs.close()  # Close the environment after inference

    ####################################################################################################################################
    ####################### For Testing Single Weight ####################################################################################
    ####################################################################################################################################        

        if single_weight_test == 1:
            current_folder = os.getcwd()
            parent_folder1 = os.path.dirname(current_folder)
            parent_folder = os.path.join(parent_folder1, 'outputs')
            weights_dir = os.path.join(parent_folder, 'weights_2')
            model_path_folder = os.path.join(weights_dir, args.Manual_weight_folder)
            weight_file = args.single_weight_file + ".zip"
            model_path = os.path.join(model_path_folder, weight_file)

            plots_dir = os.path.join(parent_folder, 'plots')

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
                actor_file = f"actor_{args.single_weight_file}.dat"
                att_net_file = f"Att_net_{args.single_weight_file}.dat"
                qf1_file = f"qf1_{args.single_weight_file}.dat"
                qf2_file = f"qf1_{args.single_weight_file}.dat"
                qf1_target_file = f"qf1_{args.single_weight_file}.dat"
                qf2_target_file = f"qf1_{args.single_weight_file}.dat"
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
                actor.load_state_dict(torch.load(os.path.join(model_path_folder, actor_file)))
                Att_net.load_state_dict(torch.load(os.path.join(model_path_folder, att_net_file)))
                qf1.load_state_dict(torch.load(os.path.join(model_path_folder, qf1_file)))
                qf2.load_state_dict(torch.load(os.path.join(model_path_folder, qf2_file)))
                qf1_target.load_state_dict(torch.load(os.path.join(model_path_folder, qf1_target_file)))
                qf2_target.load_state_dict(torch.load(os.path.join(model_path_folder, qf2_target_file)))

            ##########################################################################################################
        
            lambda1_q_attention = args.lambda1_q_attention
            Lambda_attention = args.lambda1_q_attention
            
            ########################################################################################33
            # current_folder = os.getcwd()
            # parent_folder = os.path.dirname(current_folder)
            # output_dir = os.path.join(parent_folder, 'outputs')
            # csv_dir = os.path.join(output_dir, 'csv')
            # os.makedirs(output_dir, exist_ok=True)
            # weights_dir = os.path.join(output_dir, 'weights_2')
            # plots_dir = os.path.join(output_dir, 'plots_2')
            # output_file_path_1 = os.path.join(plots_dir, args.Manual_plot_folder)
            # model_folder = os.path.join(weights_dir, args.Manual_weight_folder)
            # episode = args.single_weight_file


            
            # envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video)]) 
            # envs.single_observation_space.dtype = np.float32

            # actor_file = f"actor_{episode}.dat"
            # att_net_file = f"Att_net_{episode}.dat"
            # qf1_file = f"qf1_{episode}.dat"
            # qf2_file = f"qf1_{episode}.dat"
            # qf1_target_file = f"qf1_{episode}.dat"
            # qf2_target_file = f"qf1_{episode}.dat"

            # envs.single_observation_space.dtype = np.float32

            # actor = Actor(envs).to(device)
            # Att_net = AttentionNetwork(envs).to(device)
            # qf1 = SoftQNetwork(envs).to(device)
            # qf2 = SoftQNetwork(envs).to(device)
            # qf1_target = SoftQNetwork(envs).to(device)
            # qf2_target = SoftQNetwork(envs).to(device)
            # qf1_target.load_state_dict(qf1.state_dict())
            # qf2_target.load_state_dict(qf2.state_dict())
            # q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
            # actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

            # # Load weights for the current episode
            # actor.load_state_dict(torch.load(os.path.join(model_folder, actor_file)))
            # Att_net.load_state_dict(torch.load(os.path.join(model_folder, att_net_file)))
            # qf1.load_state_dict(torch.load(os.path.join(model_folder, qf1_file)))
            # qf2.load_state_dict(torch.load(os.path.join(model_folder, qf2_file)))
            # qf1_target.load_state_dict(torch.load(os.path.join(model_folder, qf1_target_file)))
            # qf2_target.load_state_dict(torch.load(os.path.join(model_folder, qf2_target_file)))
            ########################################################################################

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

