import argparse
# Create ArgumentParser object 
from scenerios import cases

parser = argparse.ArgumentParser()
# Define command-line arguments with default values and help messages


parser.add_argument('--case', choices=cases.keys(), default='8', help='Choose one of the predefined cases, 1: GTO1_1st , 2: GTO1_2nd, 3: GTO2_1st , 4: GTO2_2nd, 5: GTO3_1st , 6: GTO3_2nd, 7: superGTO_1st , 8: superGTO_2nd,     ')

for arg_name, arg_vals in cases['1'].items():
    parser.add_argument(f'--{arg_name}_1', type=type(arg_vals[0]), nargs=len(arg_vals), default=arg_vals, help=f'{arg_name} for case1')
    
for arg_name, arg_vals in cases['2'].items():
    parser.add_argument(f'--{arg_name}_2', type=type(arg_vals[0]), nargs=len(arg_vals), default=arg_vals, help=f'{arg_name} for case2')
    
for arg_name, arg_vals in cases['3'].items():
    parser.add_argument(f'--{arg_name}_3', type=type(arg_vals[0]), nargs=len(arg_vals), default=arg_vals, help=f'{arg_name} for case3')
    
for arg_name, arg_vals in cases['4'].items():
    parser.add_argument(f'--{arg_name}_4', type=type(arg_vals[0]), nargs=len(arg_vals), default=arg_vals, help=f'{arg_name} for case4')
    
for arg_name, arg_vals in cases['5'].items():
    parser.add_argument(f'--{arg_name}_5', type=type(arg_vals[0]), nargs=len(arg_vals), default=arg_vals, help=f'{arg_name} for case5')
    
for arg_name, arg_vals in cases['6'].items():
    parser.add_argument(f'--{arg_name}_6', type=type(arg_vals[0]), nargs=len(arg_vals), default=arg_vals, help=f'{arg_name} for case6')
    
for arg_name, arg_vals in cases['7'].items():
    parser.add_argument(f'--{arg_name}_7', type=type(arg_vals[0]), nargs=len(arg_vals), default=arg_vals, help=f'{arg_name} for case7')
    
for arg_name, arg_vals in cases['8'].items():
    parser.add_argument(f'--{arg_name}_8', type=type(arg_vals[0]), nargs=len(arg_vals), default=arg_vals, help=f'{arg_name} for case8')

parser.add_argument('--max_steps_one_ep', type=int, default=20000, help='Max number of steps in one episode ')
parser.add_argument('--max_nu_ep', type=int, default=600, help='Max number of episodes')
parser.add_argument('--weights_save_steps', type=int, default=500, help='Number of steps after which weights will save')
parser.add_argument('--lr', type=float, default=0.003, help='learning rate')
parser.add_argument('--shell_comand_nu', type=int, default=1, help='1,2,3,4  which nu of sheel command executing ')
parser.add_argument('--csv_file_nu', type=str, default="1", help='pass the csv file number for matlab ')


parser.add_argument('--algo', type=str, default="SAC", help='Select either PPO or SAC reinforcement learning algorithms')
parser.add_argument('--testing_weights', type=int, default=1, help='0: training or 1: Testing ')
parser.add_argument('--testing_converged_weights', type=int, default=0, help='0: training or 1: Testing ')
parser.add_argument('--Manual_weight_flag', type=int, default=1, help='0:auto from temp file  weihts_plot_folder or 1: manual  weihts_plot_folder ')
parser.add_argument('--Manual_weight_folder', type=str, default="May-02-2024_1714694062", help='weights_folder')
parser.add_argument('--Manual_plot_folder', type=str, default="May-02-2024_1714694078", help='plots_folder ')
parser.add_argument('--single_weight_test', type=int, default=0, help='0: Singke weight testing ')
parser.add_argument('--single_weight_file', type=str, default="models_2011000_313_4877", help='weight_file ')
parser.add_argument('--csv_infer_nu', type=str, default="1", help='1,2,3,4  which nu of sheel command executing ')



import os
from distutils.util import strtobool
# parser = argparse.ArgumentParser()
parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
    help="the name of this experiment")
parser.add_argument("--seed", type=int, default=10,
    help="seed of the experiment")
parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
    help="if toggled, `torch.backends.cudnn.deterministic=False`")
parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
    help="if toggled, cuda will be enabled by default")
parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
    help="if toggled, this experiment will be tracked with Weights and Biases")
parser.add_argument("--wandb-project-name", type=str, default="CisLunar_CRL_Attention_Agent1",
    help="the wandb's project name")
parser.add_argument("--wandb-entity", type=str, default="Prob1_Agent1",
    help="the entity (team) of wandb's project")
parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
    help="whether to capture videos of the agent performances (check out `videos` folder)")

# Algorithm specific arguments
parser.add_argument("--env-id", type=str, default="Cislunar-CRL-Agent1",
    help="the id of the environment")
parser.add_argument("--total-timesteps", type=int, default=7000000,
    help="total timesteps of the experiments")
parser.add_argument("--buffer-size", type=int, default=int(1e6),
    help="the replay memory buffer size")
parser.add_argument("--gamma", type=float, default=0.99,
    help="the discount factor gamma")
parser.add_argument("--tau", type=float, default=0.005,
    help="target smoothing coefficient (default: 0.005)")
parser.add_argument("--batch-size", type=int, default=256,
    help="the batch size of sample from the reply memory")
parser.add_argument("--learning-starts", type=int, default=5e3,
    help="timestep to start learning")
parser.add_argument("--policy-lr", type=float, default=3e-4,
    help="the learning rate of the policy network optimizer")
parser.add_argument("--q-lr", type=float, default=1e-3,
    help="the learning rate of the Q network network optimizer")
parser.add_argument("--policy-frequency", type=int, default=2,
    help="the frequency of training policy (delayed)")
parser.add_argument("--target-network-frequency", type=int, default=2, # Denis Yarats' implementation delays this by 2.
    help="the frequency of updates for the target nerworks")
parser.add_argument("--noise-clip", type=float, default=0.5,
    help="noise clip parameter of the Target Policy Smoothing Regularization")
parser.add_argument("--alpha", type=float, default=0.2,
        help="Entropy regularization coefficient.")
parser.add_argument("--autotune", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
    help="automatic tuning of the entropy coefficient")

parser.add_argument("--Att-lr", type=float, default=2e-3,
    help="the learning rate of the Attention network optimizer")
parser.add_argument("--lambda1_q_attention", type=float, default=0.8,
        help="how much weight you want to give to orignal SAC and the the rest will be given to attention in q network update. If 1 then only SAC.")
parser.add_argument("--Lambda_attention",type=float, default=0.9,
    help="In atttention network loss, how much weight for (policy_new - policy_old) and the rest will be given to regularization ie mean of attention log distribution q1q2")







# Parse the arguments
args = parser.parse_args()