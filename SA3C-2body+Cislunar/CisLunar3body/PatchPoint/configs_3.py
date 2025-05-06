




###################################################################################################################################

import argparse


cases = {
    '1': {
        'GTO_state': [101055.5640477494, 8993.12420483933, -44988.2096816413, 0.634234162794366, 0.1422920473692417, 1000.0, 58754.655],
        'Lunar_state': [278602.375283225, -51398.8391, -121477.133685, -0.2595738387892, -0.2949081557105014, 810.0, 58793.43],
        'force': 1,
        'tol_inc': 1.5, #[0.1],
        'tol_ecc': 0.1,#[0.002],
        'tol_a': 5450,#[50],
        'tol_RAAN': 100,
        'single_weight_file': "SGTO_to_PP_DRL1",
        'max_steps_in_ep': 10000,
        'var_seg': 1,
        'phi0': 0.0,
        'seg': 10.0,
        'var_small_seg': 0.005 ,
        'zip_weights' : 1
    },
    
    
    '2': {
        'GTO_state': [275174.07025642437, 17695.470568900328, -125124.96644703242, 0.35455203427086485, -0.17125742588144438, 838.97, 58793.1948 ], #   e0.00186 i 0.06 a50
        'Lunar_state': [278602.375283225, -51398.8391, -121477.133685, -0.2595738387892, -0.2949081557105014, 810.0, 58793.43],
        'force': 1,
        'tol_inc': 1.5,
        'tol_ecc': 0.01, #[0.0002],
        'tol_a': 1,#[0.6],
        'tol_RAAN': 20,
        'tol_argp': 30.0,
        'single_weight_file': "SGTO_to_PP_DRL2",
        'max_steps_in_ep': 2500,
        'var_seg': 0,
        'phi0': 1,
        'seg': 0.005,
        'var_small_seg': 0.005,
        'zip_weights' : 1,
        'milestone' : 0
    }

}



# Create ArgumentParser object 
parser = argparse.ArgumentParser()

parser.add_argument('--case', type=str, default='2', choices=cases.keys(), help='1: SGTO to PP DRL1    2: SGTO to PP DRl2')

# Define command-line arguments with default values and help messages
parser.add_argument('--GTO_state', type=float, nargs=7, default=[101055.56404, 8993.12420, -44988.20968, 0.634234170, 0.142292050, 1000.0, 59780.6  ], help='               ecc,a,inc= 0.65,    44364,      27      final super GTO orbit state [h,hx,hy,ex,ey,mass,time]')   # final super GTO   ecc,a, inc= 0.65, 44364, 27
parser.add_argument('--intermediate_state', type=float, nargs=7, default=[198606.2185,16159.117,-104684.9953,0.3714,0.0621,913.6318,0] , help='                       ecc,a,inc= 0.3766 , 115309.811, 32.23   intermediate state [h,hx,hy,ex,ey,mass,time]')  #prob11_opt line 1394  1 rev  phi 6.2    
# Patch point
parser.add_argument('--Lunar_state', type=float, nargs=7, default=[278602.375283225, -51398.8391, -121477.133685, -0.2595738387892, -0.2949081557105014,  770, 58793.43 ], help=' ecc,a,inc= 0.1267,  376118.00,  18.15   exact NRHO lunar orbit  [h,hx,hy,ex,ey,mass,time]') #prev old matlab code cislunar state: 384073.94322, 43240.35671, -111563.29463, -0.020972952, -0.124993226, 750,    5167756800     ##with new matlab code: 382063.920612, 38403.36842, -111751.18758, -0.0231943, -0.133244460, 830,    5167756800  # 306458.21327,16533.85812,-201552.451785,0.1051374865,0.1815227469,870,5161186796.34271717   

parser.add_argument('--he_ele_conv',  type=int, default=1, help='0: use a,e,i parameters for convergence, 1: use h,hx,hy,ex,ey elements for conv')
parser.add_argument('--sac_paper_cond',  type=int, default=1, help='0: use own cond for action normalization and reward weights , 1: use paper conditions')
parser.add_argument('--simpl_or_exp_rew',  type=int, default=1, help='0: simple reward, 1: exp reward')
parser.add_argument('--reward_normalize',  type=int, default=0, help='0: simple reward, 1: normalized reward with running average and mean')
parser.add_argument('--r_norm_mult',  type=int, default=10, help='multiplier to weight up the normlaized reward')


#Scaling Parameters
parser.add_argument('--new_scaling_flag',  type=int, default=1, help='0: no new scaling, 1: new scaling')
parser.add_argument('--Px',  type=float, nargs=7, default=[1, 1, 1, 1e-5, 1e-5, 1, 1], help='scaling values for states')
parser.add_argument('--qx',  type=float, nargs=7, default=[0, 0, 0, 0, 0, 0, 59812], help='scaling values for states')
parser.add_argument('--Pu',  type=float, nargs=7, default=[1, 1], help='scaling values for actions')
parser.add_argument('--qu',  type=float, nargs=7, default=[1, 1], help='scaling values for actions')
parser.add_argument('--SCst_NORMst_ORGNst_Reward',  type=int, default=2, help='1: scaled state values used in reward, 2: normalized state values used in reward,  3: Orignal state values used in reward')
parser.add_argument('--scaler_type_ST_MM',  type=int, default=2, help='1: Standard, 2: Minmax')

# Boundary State Stuck or terminate
parser.add_argument('--bounday_term_or_stuck', type=int, default=0, help='0: terminate episode if boundary conditions reach,     1: dont terminate, instead for agent to come out/back from it')

# discrete_reward or not
parser.add_argument('--discrete_reward', type=int, default=0, help='0: NO dicreter reward, Other reward,     1: dicreter reward')

# Conv/reward parameters flags
parser.add_argument('--a_conv', type=int, default=1, help='0: a not included,     1:  a included in rewad and terminal condition')
parser.add_argument('--ecc_conv', type=int, default=1, help='0: ecc not included, 1:  ecc included in rewad and terminal condition')
parser.add_argument('--inc_conv', type=int, default=1, help='0: inc not included, 1:  inc included in rewad and terminal condition')
parser.add_argument('--h_conv', type=int, default=0, help='0: h not included,     1:  h included in rewad and terminal condition')
parser.add_argument('--hx_conv', type=int, default=0, help='0: hx notr included,  1:  hx included in rewad and terminal condition')
parser.add_argument('--hy_conv', type=int, default=0, help='0: hy not included,   1:  hy included in rewad and terminal condition')
parser.add_argument('--ex_conv', type=int, default=0, help='0: ex not included,   1:  ex included in rewad and terminal condition')
parser.add_argument('--ey_conv', type=int, default=0, help='0: ey not included,   1:  ey included in rewad and terminal condition')
parser.add_argument('--RAAN_conv', type=int, default=0, help='0: argp not included,   1:  RAAN included in rewad and terminal condition')
parser.add_argument('--argp_conv', type=int, default=0, help='0: argp not included,   1:  argp included in rewad and terminal condition')
parser.add_argument('--mass_conv', type=int, default=0, help='0: mass not included,   1:  mass reached, force will become zero')



#Toerance Values
parser.add_argument('--tol_a', type=float, default=1000  , help='tolerance for semimajoraxis km')
parser.add_argument('--tol_ecc', type=float, default=0.05 , help='tolerance for eccentricity')
parser.add_argument('--tol_inc', type=float, default=0.5, help='tolerance for inclination')
parser.add_argument('--tol_h', type=float, default=10, help='tolerance for h')
parser.add_argument('--tol_hx', type=float, default=10 , help='tolerance for hx')
parser.add_argument('--tol_hy', type=float, default=10 , help='tolerance for hy')
parser.add_argument('--tol_ex', type=float, default=0.05  , help='tolerance for ex')
parser.add_argument('--tol_ey', type=float, default=0.05  , help='tolerance for ey')
parser.add_argument('--tol_RAAN', type=float, default=10  , help='tolerance for RAAN')
parser.add_argument('--tol_argp', type=float, default=10  , help='tolerance for argp')

# Simple reward weights
parser.add_argument('--w_h_s', type=float, default=0.7, help='reward_simple fun weight for h ')
parser.add_argument('--w_hx_s', type=float, default=0.7 , help='reward_simple fun weight for hx')
parser.add_argument('--w_hy_s', type=float, default=0.7 , help='reward_simple fun weight for hy')
parser.add_argument('--w_ex_s', type=float, default=0.2  , help='reward_simple fun weight for ex')
parser.add_argument('--w_ey_s', type=float, default=0.2  , help='reward_simple fun weight for ey')
parser.add_argument('--w_a_s', type=float, default=0.7 , help='reward_simple fun weight for hy')
parser.add_argument('--w_ecc_s', type=float, default=0.2  , help='reward_simple fun weight for ex')
parser.add_argument('--w_inc_s', type=float, default=0.2  , help='reward_simple fun weight for ey')
parser.add_argument('--reward_weight_upscale', type=float, default=200  , help='reward func weight upscale')


# Discrete reward weights
parser.add_argument('--w_a_dr', type=float, default=1 , help='reward_discrete fun weight for hy')
parser.add_argument('--w_e_dr', type=float, default=1  , help='reward_discrete fun weight for ex')
parser.add_argument('--w_i_dr', type=float, default=1  , help='reward_discrete fun weight for ey')
parser.add_argument('--w_h_dr', type=float, default=1, help='reward_discrete fun weight for h ')
parser.add_argument('--w_hx_dr', type=float, default=1 , help='reward_discrete fun weight for hx')
parser.add_argument('--w_hy_dr', type=float, default=1 , help='reward_discrete fun weight for hy')
parser.add_argument('--w_ex_dr', type=float, default=2  , help='reward_discrete fun weight for ex')
parser.add_argument('--w_ey_dr', type=float, default=2  , help='reward_discrete fun weight for ey')
parser.add_argument('--w_raan_dr', type=float, default=1  , help='reward_discrete fun weight for RAAN')
parser.add_argument('--w_argp_dr', type=float, default=1  , help='reward_discrete fun weight forargp')




# complex reward weights general case
parser.add_argument('--w1_h', type=float, default=800000.1500 , help='Reward function weights for w_h')
parser.add_argument('--w1_hx', type=float, default=800000.1500 , help='Reward function weights for w_hx')
parser.add_argument('--w1_hy', type=float, default=800000.1500 , help='Reward function weights for w_hy')
parser.add_argument('--w1_ex', type=float, default=760000.2000, help='Reward function weights for w_ex')
parser.add_argument('--w1_ey', type=float, default=760000.2000, help='Reward function weights for w_ey')
parser.add_argument('--w1_h_', type=float, default= 30 , help='Reward function weights for w_h_')
parser.add_argument('--w1_hx_', type=float, default= 30 , help='Reward function weights for w_hx_')
parser.add_argument('--w1_hy_', type=float, default= 30 , help='Reward function weights for w_hy_')
parser.add_argument('--w1_ex_', type=float, default=40.000000198, help='Reward function weights for w_ex_')
parser.add_argument('--w1_ey_', type=float, default=40.000000198, help='Reward function weights for w_ey_')
parser.add_argument('--c1_h', type=float, default=360000.600 , help='Reward function weights for c_h')
parser.add_argument('--c1_hx', type=float, default=360000.600 , help='Reward function weights for c_hx')
parser.add_argument('--c1_hy', type=float, default=360000.600 , help='Reward function weights for c_hy')
parser.add_argument('--c1_ex', type=float, default=200000.1200, help='Reward function weights for c_ex')
parser.add_argument('--c1_ey', type=float, default=200000.1200, help='Reward function weights for c_ey')

parser.add_argument('--w1_a', type=float, default=800000.1500 , help='Reward function weights for w_a')
parser.add_argument('--w1_e', type=float, default=760000.2000, help='Reward function weights for w_ecc')
parser.add_argument('--w1_i', type=float, default=200000 , help='Reward function weights for w_inc')
parser.add_argument('--w1_a_', type=float, default= 30 , help='Reward function weights for w_a_')
parser.add_argument('--w1_e_', type=float, default=40.000000198, help='Reward function weights for w_ecc_')
parser.add_argument('--w1_i_', type=float, default=40.00003 , help='Reward function weights for w_inc_')
parser.add_argument('--c1_a', type=float, default=360000.600 , help='Reward function weights for c_a')
parser.add_argument('--c1_e', type=float, default=200000.1200, help='Reward function weights for c_ecc')
parser.add_argument('--c1_i', type=float, default=50000.1500 , help='Reward function weights for c_inc')

parser.add_argument('--w1_RAAN', type=float, default=800000.1500 , help='Reward function weights for w_a')
parser.add_argument('--w1_argp', type=float, default=760000.2000, help='Reward function weights for w_ecc')
parser.add_argument('--w1_RAAN_', type=float, default= 30 , help='Reward function weights for w_a_')
parser.add_argument('--w1_argp_', type=float, default=40.000000198, help='Reward function weights for w_ecc_')
parser.add_argument('--c1_RAAN', type=float, default=360000.600 , help='Reward function weights for c_a')
parser.add_argument('--c1_argp', type=float, default=200000.1200, help='Reward function weights for c_ecc')

parser.add_argument('--done_ep_reward', type=int, default=1000, help='positive reward value when episode is completed/converged successfuly')
parser.add_argument('--rf', type=float, default=10 , help='penalty when it goes out the defined window')
# parser.add_argument('--tau', type=float, default=0.003 , help='Reward function constant tau value')

# complex reward weights SAC paper case
parser.add_argument('--w1_h_sac', type=float, default=1000 , help='Reward function weights for w_h for sac paper reward weights')
parser.add_argument('--w1_hx_sac', type=float, default=100, help='Reward function weights for w_hx for sac paper reward weights')
parser.add_argument('--w1_hy_sac', type=float, default=100, help='Reward function weights for w_hy for sac paper reward weights')
parser.add_argument('--w1_ex_sac', type=float, default=1000, help='Reward function weights for w_ex for sac paper reward weights')
parser.add_argument('--w1_ey_sac', type=float, default=1000, help='Reward function weights for w_ey for sac paper reward weights')
parser.add_argument('--w1_h_sac_', type=float, default= 0.005   , help='Reward function weights for w_h_ for sac paper reward weights')
parser.add_argument('--w1_hx_sac_', type=float, default= 0.005   , help='Reward function weights for w_hx_ for sac paper reward weights')
parser.add_argument('--w1_hy_sac_', type=float, default= 0.005   , help='Reward function weights for w_hy_ for sac paper reward weights')
parser.add_argument('--w1_ex_sac_', type=float, default=100, help='Reward function weights for w_ex_ for sac paper reward weights')
parser.add_argument('--w1_ey_sac_', type=float, default=100, help='Reward function weights for w_ey_ for sac paper reward weights')
parser.add_argument('--c1_h_sac', type=float, default=10 , help='Reward function weights for c_h for sac paper reward weights')
parser.add_argument('--c1_hx_sac', type=float, default=10 , help='Reward function weights for c_hx for sac paper reward weights')
parser.add_argument('--c1_hy_sac', type=float, default=10 , help='Reward function weights for c_hy for sac paper reward weights')
parser.add_argument('--c1_ex_sac', type=float, default=10, help='Reward function weights for c_ex for sac paper reward weights')
parser.add_argument('--c1_ey_sac', type=float, default=10, help='Reward function weights for c_ey for sac paper reward weights')
parser.add_argument('--w1_a_sac', type=float, default=800000.0 , help='Reward function weights for w_a for sac paper reward weights')
parser.add_argument('--w1_e_sac', type=float, default=800000.0, help='Reward function weights for w_ecc for sac paper reward weights')
parser.add_argument('--w1_i_sac', type=float, default=800000.0, help='Reward function weights for w_inc for sac paper reward weights')
parser.add_argument('--w1_a_sac_', type=float, default= 30 , help='Reward function weights for w_a_ for sac paper reward weights')
parser.add_argument('--w1_e_sac_', type=float, default=30, help='Reward function weights for w_ecc_ for sac paper reward weights')
parser.add_argument('--w1_i_sac_', type=float, default=30 , help='Reward function weights for w_inc_ for sac paper reward weights')
parser.add_argument('--c1_a_sac', type=float, default=10, help='Reward function weights for c_a for sac paper reward weights')
parser.add_argument('--c1_e_sac', type=float, default=10, help='Reward function weights for c_ecc for sac paper reward weights')
parser.add_argument('--c1_i_sac', type=float, default=10 , help='Reward function weights for c_inc for sac paper reward weights')
parser.add_argument('--w1_RAAN_sac', type=float, default=0.03 , help='Reward function weights for w_a')
parser.add_argument('--w1_argp_sac', type=float, default=0.03, help='Reward function weights for w_ecc')
parser.add_argument('--w1_RAAN_sac_', type=float, default= 0.005 , help='Reward function weights for w_a_')
parser.add_argument('--w1_argp_sac_', type=float, default= 0.005, help='Reward function weights for w_ecc_')
parser.add_argument('--c1_RAAN_sac', type=float, default= 10 , help='Reward function weights for c_a')
parser.add_argument('--c1_argp_sac', type=float, default= 10, help='Reward function weights for c_ecc')
parser.add_argument('--done_ep_reward_sac', type=float, default=10000 , help='Reward function weights for done_ep_reward_sac for sac paper reward weights')
parser.add_argument('--rf_sac', type=float, default=10000 , help='Reward function weights for redflag for sac paper reward weights')




parser.add_argument('--r_norm',  type=int, default=0, help='0: reward with orignal values, 1: reward with normalized values')
parser.add_argument('--milestone',  type=int, default=1, help='0: no milestone used, 1: milesone values used')
parser.add_argument('--e_a_i_ms_1', type=float, nargs=3, default=[0.15, 15000, 1], help=' ecc,a,inc= 0.1267,  376118.00,  18.15  Milestone 1 values , worked only if milestone == 1') 
parser.add_argument('--ms_reward',  type=float, default=50, help='additional reward value per step if milestone achieved')  
parser.add_argument('--max_steps_one_ep', type=int, default=5000, help='Max number of steps in one episode ')
parser.add_argument('--max_nu_ep', type=int, default=2500, help='Max number of episodes')
parser.add_argument('--weights_save_steps', type=int, default=200, help='Number of steps after which weights will save')
# parser.add_argument('--buffer_size', type=int, default=10000, help='size of replay buffer')
# parser.add_argument('--gamma', type=float, default=0.99, help='value of discounting parametr gamma')
parser.add_argument('--lr', type=float, default=0.003, help='learning rate')
parser.add_argument('--pre_tr_weight', type=int, default=0, help='0: train from scratch, 1: train from pretrain weights')
parser.add_argument('--var_seg', type=int, default=1, help='0: no var segment, 1: var segment')


parser.add_argument('--test',  type=int, default=0, help='0: no testing/debugging, 1/2/3..: testing/debugging')

parser.add_argument('--normalization_case', type=int, default=1, help='1: normalization of state vector(h,hx,hy) with general value, 2: normalization of state vector(h,hx,hy) with seprate values . ')
parser.add_argument('--shell_comand_nu', type=int, default=1, help='1,2,3,4  which nu of sheel command executing ')
parser.add_argument('--intermediate_state_use', type=int, default=0, help='0: (NO) Target is final position , 1: (YES) Target is intermediate state')
parser.add_argument('--manual_weights', type=int, default=0, help='0: (NO) selct reward weights through args , 1: (YES) selec manualy reward weights in code')
parser.add_argument('--algo', type=str, default="SAC", help='Select either PPO or SAC reinforcement learning algorithms')
parser.add_argument('--HER', type=int, default=0, help='0: (NO) HER , 1: (YES) HER')
parser.add_argument('--one_W_save', type=int, default=0, help='0: weights saved after every 2000/nsteps HER , 1: weights saved only oncce at the end of training. ')
parser.add_argument('--HER_maxep_len', type=int, default=20000, help='if HER is on, define max len of one episode')
parser.add_argument('--HER_tr_start_epoch', type=int, default=40000, help='if HER is on define training start epoch ')
parser.add_argument('--sigma_1', type=float, default=0.2, help='added noise standard daviation for actions ')


parser.add_argument('--force', type=float, default=1, help='in N eg 1N')
parser.add_argument('--phi0', type=float, default=0, help='Starting value of phi from moon to earth it is 1.4046,  new matlabcode: 1.3984,    easy-1.0438')
parser.add_argument('--seg', type=float, default=6 , help='segment size in degrees')
parser.add_argument('--var_small_seg', type=float, default=0.05 , help='small segment size for var seg (in degrees)')
# parser.add_argument('--m0', type=float, default=830 , help='mass in kg at lunar orbit or starting point if its from moon to earth')


parser.add_argument('--a_boundary_flag', type=int, default=1, help='0: a max min boudary conditions not included, 1: vice versa')
parser.add_argument('--inc_boundary_flag', type=int, default=1 , help='0: inc max min boudary conditions not included, 1: vice versa')
parser.add_argument('--ecc_boundary_flag', type=int, default=1 , help='0: ecc max min boudary conditions not included, 1: vice versa')
parser.add_argument('--a_min', type=float, default=40000 , help='[target_earth_a(44364) - value(4364)], Minimum a value, after which we terminate the episode. works if a_boundary_flag is 1')
parser.add_argument('--a_max', type=float, default=420000 , help='[inital_lunar_a(376118) + value(24000)], Maximum a value, after which we terminate the episode. works if a_boundary_flag is 1')
parser.add_argument('--inc_min', type=float, default=10 , help='[initial_lunar_inc(18.15) - value(3)], Minimum inc value, after which we terminate the episode. works if inc_boundary_flag is 1')
parser.add_argument('--inc_max', type=float, default=52, help='[target_earth_inc(27) + value(3)], Maximum inc value, after which we terminate the episode. works if inc_boundary_flag is 1')
parser.add_argument('--ecc_min', type=float, default=0.0067  , help='[initial_lunar_ecc(0.1267) - value(0.12)], Minimum ecc value, after which we terminate the episode. works if ecc_boundary_flag is 1')
parser.add_argument('--ecc_max', type=float, default=0.94 , help='[target_earth_ecc(0.65) + value(0.35)], Maximum ecc value, after which we terminate the episode. works if ecc_boundary_flag is 1')

parser.add_argument('--sh_flag', type=float, default=0, help='for considering shadow. if 1 shadow consider if 0 shadow not considerable')

parser.add_argument('--testing_weights', type=int, default=0, help='0: training or 1: Testing ')
parser.add_argument('--testing_converged_weights', type=int, default=0, help='0: training or 1: Testing ')
parser.add_argument('--Manual_weight_flag', type=int, default=1, help='0:auto from temp file  weihts_plot_folder or 1: manual  weihts_plot_folder ')
parser.add_argument('--Manual_weight_folder', type=str, default="May-01-2024_1714609244", help='weights_folder')
parser.add_argument('--Manual_plot_folder', type=str, default="May-01-2024_1714609255", help='plots_folder ')
parser.add_argument('--single_weight_test', type=int, default=0, help='0: Singke weight testing ')
parser.add_argument('--single_weight_file', type=str, default="", help='weight_file ')
parser.add_argument('--csv_infer_nu', type=str, default="1222222", help='1,2,3,4  which nu of sheel command executing ')

parser.add_argument('--GTO_to_Lunar', type=int, default=1, help='0: Lunar_to_GTO or 1: GTO_to_Lunar  ')
parser.add_argument('--csv_file_nu', type=str, default="122225257474", help='pass the csv file number for matlab ')


parser.add_argument('--seed_test_flag', type=int, default=0, help='1,2,3,4  which nu of sheel command executing ')
parser.add_argument('--zip_weights', type=int, default=1, help='1,2,3,4  which nu of sheel command executing ')


from distutils.util import strtobool
import os
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

# Override default values with the values from the selected case
selected_case = cases.get(args.case, {})
for key, value in selected_case.items():
    setattr(args, key, value)

print(args)