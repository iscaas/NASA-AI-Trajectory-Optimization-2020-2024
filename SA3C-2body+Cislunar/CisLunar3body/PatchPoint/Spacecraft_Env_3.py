import numpy as np
import gym
from gym import spaces
# import gymnasium as gym
# from gymnasium import spaces
from enviornment_3 import Enviornment , Normalization
##################################################################
import math
import numpy as np
import random
#from IPython.display import clear_output
from collections import deque
import matlab.engine         # IMPORTING MATLAB ENGINE
#import progressbar          
import random
from sklearn.preprocessing import StandardScaler ,  MinMaxScaler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.image as img
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
import matlab.engine
import csv
from numpy.core.fromnumeric import shape         
# from tensorflow.keras import Model
# from tensorflow.keras.layers import Dense, Embedding, Reshape
# from tensorflow.keras.optimizers import Adam
import os.path
import time
from datetime import date
import matplotlib.cm as cm
import matplotlib.animation as animation
# import tensorflow as tf 
# tf.debugging.set_log_device_placement(True)
import argparse
from configs_3 import args

##################################################################
if args.HER:  
  gym_passing_argument = gym.GoalEnv
else:
  gym_passing_argument = gym.Env
##################################################################  

class SpacecraftEnv(gym_passing_argument):
  """
  Custom Environment that follows gym interface.
  This is a spacecraft env where agent learns to find the optimal satellite path. 
  """
  def __init__(self, args=args, pre_tr_path=''):
    super(SpacecraftEnv, self).__init__()
    eng = matlab.engine.start_matlab()
    self.env = Enviornment(eng, args)
    self.norm = Normalization(eng, args)
    self.max_episode_steps = self.max_episode_length = args.max_steps_one_ep
    self.max_steps_one_ep = self.max_episode_steps
    self.sh_command_nu = args.shell_comand_nu
    self.milestone = args.milestone
    self.testing_weights = args.testing_weights
    self.GTO_to_Lunar =  args.GTO_to_Lunar
    if self.GTO_to_Lunar:
        self.init_state = np.array(args.GTO_state)   # [h;hx;hy;ex;ey;mass;time]  superGTO2 Adrean paper   Isp=1500 m0=1000kg F= 1N
        self.target_state  = np.array(args.Lunar_state)     # [h;hx;hy;ex;ey;mass;time]   NRHO lunar Adrean paper   Isp=1500 m0=1000kg F= 1N
    else:
        self.init_state = np.array(args.Lunar_state)   # [h;hx;hy;ex;ey;mass;time]  NRHO lunar Adrean paper   Isp=1500 m0=1000kg F= 1N
        self.target_state  = np.array(args.GTO_state)     # [h;hx;hy;ex;ey;mass;time]  superGTO2 Adrean paper   Isp=1500 m0=1000kg F= 1N 
    self.num_envs = 1
    self.rf = args.rf
    self.he_para_flag = args.he_ele_conv
    self.tol_h, self.tol_hx, self.tol_hy, self.tol_ex, self.tol_ey = args.tol_h, args.tol_hx, args.tol_hy, args.tol_ex, args.tol_ey
    self.sac_paper_cond = args.sac_paper_cond
    self.new_scaling_flag = args.new_scaling_flag
    self.Px, self.qx, self.Pu, self.qu = args.Px[0:-1], args.qx[0:-1], args.Pu, args.qu
    self.pre_tr_weights_path = pre_tr_path
    self.HER_flag = args.HER
    self.h_history, self.hx_history, self.hy_history, self.ex_history, self.ey_history, self.ex_scaled_history, self.ey_scaled_history, self.phi_history, self.mass_history, self.force_history, self.time_in_days_history, self.time_history, self.alpha_history, self.beta_history, self.thrust_history = [[] for _ in range(15)]
    self.ecc_history, self.ecc_scaled_history, self.a_history, self.inclination_history, self.ecc_history_nd, self.a_history_nd, self.inclination_history_nd, self.RAAN_history, self.argp_history, self.nurev_history = [[] for _ in range(10)]
    self.score_data, self.score_detailed_data = [[]], [[0]]
    self.a_plot_history, self.e_plot_history, self.i_plot_history, self.a_exp_plot_history, self.e_exp_plot_history, self.i_exp_plot_history, self.a_total_plot_history, self.e_total_exp_plot_history, self.i_total_exp_plot_history, self.reward_st_plot_history, self.reward_st1_exp_plot_history, self.reward_st1_m_st_plot_history, self.reward_st1_m_st_m_tau_plot_history, self.reward_st1_m_st_m_tau_m_100rf_plot_history, self.reward_st1_m_st_m_tau_m_100rf_p_ms_plot_history = [[] for _ in range(15)]
    self.seg_count, self.a_sum, self.time_rev, self.ep_time = 0, 0, 0, 0
    self.time = 59812 if not self.GTO_to_Lunar else 59781; self.phi_0 = 1.4046 if not self.GTO_to_Lunar else 0
    self.ep_time_1, self.ep_counter, self.counter, self.MAIN_Episode, self.done_counter =  0, 0, 0, 1, 0
    self.ep_counter_SAC_step, self.ep_Cont_len_counter, self.success_counter, self.time_mat, self.Force_1, self.terminate = -1, 0, 0, 0, args.force, 0
    self.acc_reward, self.initial_state_value = 0, []
    today = date.today()
    self.day, self.txt_dir = today.strftime('%b-%d-%Y'), f"{int(time.time())}"
    self.dir_name =  f"{self.day}_{self.txt_dir}"
    if self.testing_weights== 0 or args.single_weight_test == 1:
      if args.single_weight_test == 0:
        current_folder = os.getcwd()
        parent_folder = os.path.dirname(current_folder)
        output_dir = os.path.join(parent_folder, 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        plots_dir = os.path.join(output_dir, 'plots_2')
        os.makedirs(plots_dir, exist_ok=True)
        folder_path = os.path.join(plots_dir, self.dir_name)
      if args.single_weight_test == 1:
         current_folder = os.getcwd()
         parent_folder = current_folder
         output_dir = os.path.join(current_folder, 'outputs')
         os.makedirs(output_dir, exist_ok=True)
         plots_dir = os.path.join(current_folder, 'plots')
         os.makedirs(plots_dir, exist_ok=True)
         folder_path = os.path.join(plots_dir, self.dir_name)
      if args.single_weight_test == 0:
        temp_dir_for_saving_folder_name = os.path.join(output_dir, 'temp_mainfolder.dat')
        with open(temp_dir_for_saving_folder_name, 'w+') as file1:
              file1.write(folder_path)
      self.args_info_file_path = os.path.join(f"{folder_path}/{'args_info.dat'}")
      folders = ['h', 'hx', 'hy', 'ex', 'ey', 'a', 'ecc', 'inc', 'mass', 'phi', 'sum_reward', 'time_in_days', 'force', 'reward_analysis', 'hx_hy', 'ex_ey', 'RAAN', 'argp', 'nurev', 'ex_ey_scaled', 'ecc_scaled', 'actions']
      folder_paths = [os.path.join(f"{folder_path}/{folder}") for folder in folders]
      folder_path_time = os.path.join(f"{folder_path}/{'time'}")
      self.successful_episodes = os.path.join(f"{folder_path}/{'Successful_episodes'}")
      if not os.path.exists(folder_path):
          os.makedirs(folder_path)
          [os.makedirs(folder_path_1) for folder_path_1 in folder_paths]
          os.makedirs(folder_path_time)
          with open(self.args_info_file_path, 'w') as file:
              file.close
      self.completeName_successful, self.info_file, self.write_final_state, self.write_reward_analysis_data, self.completeName_all_data, self.figure_file_ecc, self.figure_file_a, self.figure_file_inclination, self.figure_file_reward, self.figure_file_time_in_days, self.figure_file_time_success, self.figure_file_time_success_actual = [os.path.join(f"{folder_path}/{'output.dat'}"), os.path.join(f"{folder_path}/{'info.dat'}"), os.path.join(f"{folder_path}/{'final_state.dat'}"), os.path.join(f"{folder_path}/{'reward_analysis.dat'}"), os.path.join(f"{folder_path}/{'output_all_data.dat'}"), os.path.join(f"{folder_path}/{'_ecc_history.dat'}"), os.path.join(f"{folder_path}/{'_a_history.dat'}"), os.path.join(f"{folder_path}/{'_inclination_history.dat'}"), os.path.join(f"{folder_path}/{'_reward_history.dat'}"), os.path.join(f"{folder_path}/{'_time_in_days.dat'}"), os.path.join(f"{folder_path}/{'_success_ep_time.dat'}"), os.path.join(f"{folder_path}/{'_success_ep_time_actual.dat'}")]
      self.folder_path_h, self.folder_path_hx, self.folder_path_hy, self.folder_path_ex, self.folder_path_ey, self.folder_path_a, self.folder_path_ecc, self.folder_path_inc, self.folder_path_mass, self.folder_path_phi, self.folder_path_sum_reward, self.folder_path_time_in_days, self.folder_path_force, self.folder_path_reward_analysis, self.folder_path_hx_hy, self.folder_path_ex_ey, self.folder_path_RAAN, self.folder_path_argp, self.folder_path_nurev, self.folder_path_exey_sc, self.folder_path_ecc_sc, self.folder_time, self.folder_path_actions  = [os.path.join(f"{folder_paths[0]}/{'h'}"), os.path.join(f"{folder_paths[1]}/{'hx'}"), os.path.join(f"{folder_paths[2]}/{'hy'}"), os.path.join(f"{folder_paths[3]}/{'ex'}"), os.path.join(f"{folder_paths[4]}/{'ey'}"), os.path.join(f"{folder_paths[5]}/{'a'}"), os.path.join(f"{folder_paths[6]}/{'ecc'}"), os.path.join(f"{folder_paths[7]}/{'inc'}"), os.path.join(f"{folder_paths[8]}/{'mass'}"), os.path.join(f"{folder_paths[9]}/{'phi'}"), os.path.join(f"{folder_paths[10]}/{'reward'}"), os.path.join(f"{folder_paths[11]}/{'time_in_days'}"), os.path.join(f"{folder_paths[12]}/{'force'}"), folder_paths[13], os.path.join(f"{folder_paths[14]}/{'hx_hy'}"), os.path.join(f"{folder_paths[15]}/{'ex_ey'}"), os.path.join(f"{folder_paths[16]}/{'RAAN'}"), os.path.join(f"{folder_paths[17]}/{'argp'}"), os.path.join(f"{folder_paths[18]}/{'nurev'}"), os.path.join(f"{folder_paths[19]}/{'ex_ey_scaled'}"), os.path.join(f"{folder_paths[20]}/{'ecc_scaled'}"), os.path.join(f"{folder_path_time}/{'ep_time'}"), os.path.join(f"{folder_paths[21]}/{'actions'}")]
              
    if self.sac_paper_cond:       
      self.action_space = spaces.Box(low=np.float32([-1, -1]),
                                        high=np.float32([1, 1]), shape=(2,),
                                          dtype=np.float32)
    else:
      self.action_space = spaces.Box(low=np.float32([-1, -1]),
                                        high=np.float32([1, 1]), shape=(2,),
                                          dtype=np.float32)

    self.observation_space = spaces.Box(low=-15, high=15,
                                        shape=(6,), dtype=np.float64)
    if self.HER_flag:
        self.a_min, self.a_max, self.inc_min, self.inc_max, self.ecc_min, self.ecc_max = args.a_min, args.a_max, args.inc_min, args.inc_max, args.ecc_min, args.ecc_max
        self.mu, self.target_state_1 = 1, np.array(args.GTO_state)
        self.ND_tar_state, self.ND_tar_obs = np.array(self.env.DimtoNonDim_states(np.array(self.target_state[:6]))), np.array(self.env.DimtoNonDim_states(np.array(self.target_state[:6])))
        self.tol_inc_her, self.tol_ecc_her, self.tol_a_her = args.tol_inc, args.tol_ecc, args.tol_a
        self.target_inc_her = ((math.asin(math.sqrt((self.ND_tar_state[1]**2)+(self.ND_tar_state[2]**2))/self.ND_tar_state[0])) / np.pi)*180 
        self.target_ecc_her = math.sqrt((self.ND_tar_state[3]**2)+(self.ND_tar_state[4]**2))
        self.target_a_her = ((self.ND_tar_state[0]**2)/self.mu) / (1 - (self.target_ecc_her ** 2))
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(low=-15, high=15, shape=(6,), dtype=np.float32),
            'desired_goal': spaces.Box(low=np.array([self.target_a_her, self.target_ecc_her, self.target_inc_her/10]), high=np.array([self.target_a_her, self.target_ecc_her, self.target_inc_her/10]), dtype=np.float32),
            'achieved_goal': spaces.Box(low=np.array([0, self.ecc_min, self.inc_min/10]), high=np.array([2, self.ecc_max, self.inc_max/10]), dtype=np.float32)
        })
    else:
        self.observation_space = spaces.Box(low=-15, high=15, shape=(6,), dtype=np.float64)

    w1 = {"a": args.w1_a, "e": args.w1_e, "i": args.w1_i}
    w1_ = {"a_": args.w1_a_, "e_": args.w1_e_, "i_": args.w1_i_}
    c1 = {"a": args.c1_a, "e": args.c1_e, "i": args.c1_i}
    self.weights = {
        "w1": w1,
        "w1_": w1_,
        "c1": c1,
        "tau": args.tau
    }
 
  # 84-94, 152, 201-202, 208-209, 214-217, 219,220      up- 613-639, 534-597, 477-501, 470-471, 286-313, 160-208     # transient 84-94, 152, 201-202, 208-209, 214-217, 220  
  def reset(self):
    """
    Important: the observation must be a numpy array
    :return: (np.array) 
    """
    # Initialize the agent (spacecraft) states
    self.state,  self.phi_0 = self.env.reset_csv()
    self.time = self.state[6]

    scaled_state = self.norm.Orignal_to_Scaled_st(self.state[0:-1])
    normalized_state = self.norm.Scaled_to_Normalized_st(scaled_state)
    self.ND_state =  normalized_state[0]
    self.target_scaled = self.norm.Orignal_to_Scaled_st(self.target_state[0:-1])

    if self.testing_weights == 0 or args.single_weight_test == 1:
      SpacecraftEnv.write_info_file(self, ep_counter= self.ep_counter , scaled_state=scaled_state, target_scaled=self.target_scaled)

    SpacecraftEnv.reset_apend_arrays(self, state= self.state , scaled_state=scaled_state)
   
    if self.HER_flag:
      ecc_ND = math.sqrt(self.ND_state[3]**2 + self.ND_state[4]**2)
      a_ND = (self.ND_state[0]**2 / 1) / (1 - ecc_ND**2)
      i_ND = (math.asin(math.sqrt(self.ND_state[1]**2 + self.ND_state[2]**2) / self.ND_state[0]) / np.pi) * 180
      self.observation = {'observation': self.ND_state,
                          'desired_goal': np.array([self.target_a_her,self.target_ecc_her,self.target_inc_her/10], dtype=np.float32),
                          'achieved_goal': np.array([a_ND,ecc_ND, i_ND/10], dtype=np.float32)}
    else:
      self.observation =  self.ND_state
    info = {}

    return self.observation

  
  def step(self, action):
    action = np.array([action[0], action[1]])
    ## Convert Non dimensional values to dimensional to pass through the matlab step function+
    if self.HER_flag:
        self.state1_ND, _ = np.array(self.env.NonDimtoDim_states(self.observation['observation']))  #[h, hx, hy, ex, ey, mass]
    else:
        self.state1 = self.observation
        self.state1_ND = self.norm.Normalized_to_Scaled_st(self.observation)
        self.orig_state = self.norm.Scaled_to_Orignal_st(self.state1_ND)

    if self.state1.shape != (1, 6):
        self.state1 = np.array([self.state1])
        
    if self.ep_fixed_len_counter == 0:
        self.initial_state = np.array(args.Lunar_state) if not self.GTO_to_Lunar else  np.array(args.GTO_state)   # [h;hx;hy;ex;ey;mass;time]  superGTO2 Adrean paper   Isp=1500 m0=1000kg F= 1N
        time, self.start_time = self.initial_state[-1], self.initial_state[-1]
        # time, self.start_time = (59812, 59812) if not self.GTO_to_Lunar else (59781, 59781)
    else:
      time = self.time
    phi_0 =  self.phi_0

    # Increase counter value
    self.ep_Cont_len_counter = self.ep_Cont_len_counter + 1
    self.ep_fixed_len_counter = self.ep_fixed_len_counter + 1
    self.counter = self.counter + 1

    ## matlab step function
    self.next_state,self.observation_1,self.state1_ND,   rewd,self.done , redflag, target_state_parameters, scaled_ecc_param,  segment, phi_1,time_new, time_in_days, reward_plot, Forc  , RAAN,argp, nurev, raise_error_counter= self.env.step(self.state1,self.state1_ND, self.ep_fixed_len_counter,  time, action[0], action[1] ,phi_0, self.Force_1 , self.target_scaled, self.Px, self.qx, self.Pu, self.qu, self.start_time )
    self.reward, self.phi_0, self.time, self.Force_1 = rewd[0], phi_1, time_new, float(round(Forc,4))
    
    self.reward = (self.reward - args.rf_sac ) if (self.ep_Cont_len_counter > (self.max_steps_one_ep-1)) else self.reward
    SpacecraftEnv.step_append_arrays(self, time_in_days=time_in_days, target_state_parameters= target_state_parameters, scaled_ecc_param=scaled_ecc_param, phi_0=phi_0, action=action, reward_plot=reward_plot, RAAN=RAAN, argp=argp, nurev=nurev)
 
    
    # if  self.ep_fixed_len_counter > 10:
    #   self.done, redflag = 1, 1

    if self.testing_weights== 0 or args.single_weight_test == 1:
      if self.ep_Cont_len_counter == 10000 and (self.ep_counter-1 ) % 10 == 0 and (self.ep_counter-1 ) != 0:
        self.env.plot_variable("Score_value", self.score_data, self.figure_file_reward, self.ep_counter, all_episode_plot_flag=1)
    
      if ((redflag==1) or (self.ep_Cont_len_counter % 2000 == 0)) or (self.ep_Cont_len_counter > (self.max_steps_one_ep-1)):
        SpacecraftEnv.redflag_plotvariables(self, target_state_parameters=target_state_parameters, scaled_ecc_param=scaled_ecc_param)

    if self.done:
      self.success_counter = self.success_counter +1
      self.time_history.append(float(round(time_in_days,6)))
      if self.testing_weights== 0  or args.single_weight_test == 1:
        SpacecraftEnv.step_writing_Successful_episodes(self, time_in_days=time_in_days, RAAN=RAAN, argp=argp, nurev=nurev)
        SpacecraftEnv.step_writing_final_states(self, RAAN=RAAN, argp=argp, nurev=nurev)
        SpacecraftEnv.succesful_folder_aei_plotvariables(self, target_state_parameters=target_state_parameters)
        SpacecraftEnv.succesful_folder_plotvariables(self, target_state_parameters=target_state_parameters, scaled_ecc_param=scaled_ecc_param)
    
    if self.testing_weights== 0 or args.single_weight_test == 1:
      SpacecraftEnv.step_print_values(self, time_in_days=time_in_days, target_state_parameters=target_state_parameters, raise_error_counter=raise_error_counter, reward_plot=reward_plot, current_state=self.next_state, target_state=self.target_state )     
      SpacecraftEnv.step_all_ep_writing(self, time_in_days=time_in_days, time=time_new  ,target_state_parameters=target_state_parameters, action=action, reward_plot=reward_plot, RAAN=RAAN, argp=argp, nurev=nurev, segment=segment, current_state=self.next_state, target_state=self.target_state)     
      
    if self.HER_flag:
            self.ND_tar_state, self.ND_tar_obs = np.array(self.env.DimtoNonDim_states(np.array(self.target_state[:6])))
            ecc_prev, ecc_new, ecc_target = math.sqrt(self.state1_ND[3]**2 + self.state1_ND[4]**2), math.sqrt(self.observation_1[3]**2 + self.observation_1[4]**2), math.sqrt(self.ND_tar_state[3]**2 + self.ND_tar_state[4]**2)
            a_prev, a_new, a_target = (self.state1_ND[0]**2) / (1 - ecc_prev**2), (self.observation_1[0]**2) / (1 - ecc_new**2), (self.ND_tar_state[0]**2) / (1 - ecc_target**2)
            i_prev, i_new, i_target = (math.asin(math.sqrt(self.state1_ND[1]**2 + self.state1_ND[2]**2) / self.state1_ND[0]) / np.pi) * 18, (math.asin(math.sqrt(self.observation_1[1]**2 + self.observation_1[2]**2) / self.observation_1[0]) / np.pi) * 18, (math.asin(math.sqrt(self.ND_tar_state[1]**2 + self.ND_tar_state[2]**2) / self.ND_tar_state[0]) / np.pi) * 18
            prev, new, target = {"a": a_prev, "e": ecc_prev, "i": i_prev}, {"a_": a_new, "e_": ecc_new, "i_": i_new}, {"a": a_target, "e": ecc_target, "i": i_target}
            info = {'info': np.array([[a_new, ecc_new, i_new]], dtype=np.float32)}
            self.observation = {
                'observation': np.array(self.observation_1),
                'desired_goal': np.array([self.target_a_her, self.target_ecc_her, i_target], dtype=np.float32),
                'achieved_goal': np.array([a_new, ecc_new, i_new], dtype=np.float32)            }
    else:
      self.observation =  np.array(self.observation_1)
    
    self.max_ep_flag = 0
    if (redflag) or (self.ep_fixed_len_counter > self.max_steps_one_ep ):    #50000 with 10 degree , 1000000 with 1 deg
      self.done=1
      self.terminate = 1
      if self.ep_fixed_len_counter > self.max_steps_one_ep :
          self.max_ep_flag = 1

    self.red_flag=  redflag
    self.time_in_days = time_in_days

    ecc1 = math.sqrt(self.orig_state[0][3]**2 + self.orig_state[0][4]**2)
    a1 = (self.orig_state[0][0]**2 / 398600.4418) / (1 - ecc1**2)
    i1 = (math.asin(math.sqrt(self.orig_state[0][1]**2 + self.orig_state[0][2]**2) / self.orig_state[0][0]) / np.pi) * 180

    info = {'ep_nu': self.ep_counter,
            'ep_length': self.ep_fixed_len_counter,
            'red_flag' : redflag,
            'max_ep_flag' : self.max_ep_flag ,
            'return': self.reward,
            'score':  self.acc_reward,
            'time': time_in_days,
             'a': a1,
              'e': ecc1,
              'i': i1}

    # if  self.ep_fixed_len_counter == 5:
    #   self.done =1
    if self.done == 1:# and redflag == 0 and self.max_ep_flag == 0:
      # Add the "episode" key to info
      info["episode"] = { "t": time_in_days, "l": self.ep_fixed_len_counter, "r": self.acc_reward}

    return self.observation, self.reward, self.done, info
  

  def render(self):
    pass

  def close(self):
    pass
  
  def seed(self, seed=10):
        np.random.seed(seed)  # Set the random seed for numpy
        random.seed(seed)  # Set the random seed for the 'random' module
        # Any other custom random seed settings you may have












  
  def write_info_file(self, ep_counter, scaled_state, target_scaled):
    if ep_counter <1:
      initial_state,target_state,inter_state, target_a,target_ecc,target_inc,tol_a_low,tol_a_high,tol_ecc_low,tol_ecc_high,tol_inc_low,tol_inc_high,weights_manual,intermediate_state_use,RL_algo,self.done_ep_reward, pre_trained_weights_flag , milestone, ecc_ms_1_tol_low,ecc_ms_1_tol_high,  a_ms_1_tol_low,a_ms_1_tol_high,  inc_ms_1_tol_low,inc_ms_1_tol_high, ms_reward,     he_para_flag,   a_conv,ecc_conv,inc_conv,    h_conv,hx_conv,hy_conv,ex_conv,ey_conv,    simpl_or_exp_rew , tol_h,tol_hx , tol_hy , tol_ex ,tol_ey,tol_RAAN, tol_argp= self.env.get_param_values()
      with open(self.args_info_file_path, 'w') as info_file:
        info_file.write("Command-line arguments:\n")
        for arg, value in vars(args).items():
            info_file.write(f"{arg}: {value}\n")



      if self.sac_paper_cond:
          self.env.writing_info_file(initial_state,target_state,scaled_state, target_scaled, args.w1_a_sac,args.w1_e_sac,args.w1_i_sac,  args.w1_a_sac_,args.w1_e_sac_,args.w1_i_sac_,  args.c1_a_sac,args.c1_e_sac,args.c1_i_sac,  args.tau, 
                                    args.normalization_case ,args.shell_comand_nu, self.state[-2], args.seg , args.force,    args.tol_inc,args.tol_ecc,args.tol_a,   
                                    args.a_boundary_flag,args.inc_boundary_flag,args.ecc_boundary_flag,   args.a_min,args.a_max,args.inc_min,args.inc_max,args.ecc_min,args.ecc_max,    args.sh_flag,  
                                    target_a,target_ecc,target_inc,   tol_a_low,tol_a_high,    tol_ecc_low,tol_ecc_high,      tol_inc_low,tol_inc_high, weights_manual, intermediate_state_use, RL_algo,args.done_ep_reward_sac, 
                                    pre_trained_weights_flag , self.pre_tr_weights_path , milestone, ecc_ms_1_tol_low,ecc_ms_1_tol_high,  a_ms_1_tol_low,a_ms_1_tol_high,  inc_ms_1_tol_low,inc_ms_1_tol_high, ms_reward, 
                                    he_para_flag,   a_conv,ecc_conv,inc_conv,    h_conv,hx_conv,hy_conv,ex_conv,ey_conv,   args.RAAN_conv,args.argp_conv,  simpl_or_exp_rew,
                                    tol_h,tol_hx , tol_hy , tol_ex ,tol_ey,tol_RAAN, tol_argp, 
                                    args.w1_h_sac, args.w1_hx_sac, args.w1_hy_sac, args.w1_ex_sac, args.w1_ey_sac, args.w1_h_sac_, args.w1_hx_sac_, args.w1_hy_sac_, args.w1_ex_sac_, args.w1_ey_sac_, args.c1_h_sac, args.c1_hx_sac, args.c1_hy_sac, args.c1_ex_sac, args.c1_ey_sac,
                                    args.w1_RAAN_sac, args.w1_argp_sac,args.w1_RAAN_sac_, args.w1_argp_sac_,args.c1_RAAN_sac, args.c1_argp_sac,
                                    args.w_h_s, args.w_hx_s, args.w_hy_s, args.w_ex_s, args.w_ey_s, args.w_a_s, args.w_ecc_s, args.w_inc_s,args.reward_weight_upscale,
                                    args.max_nu_ep,  args.max_steps_one_ep, args.phi0,args.sac_paper_cond, self.new_scaling_flag, 
                                    args.discrete_reward, args.w_a_dr, args.w_e_dr, args.w_i_dr, args.w_h_dr, args.w_hx_dr, args.w_hy_dr, args.w_ex_dr, args.w_ey_dr, args.w_argp_dr, args.w_raan_dr,
                                    self.info_file  )
      else:  
          self.env.writing_info_file(initial_state,target_state,scaled_state, target_scaled, args.w1_a,args.w1_e,args.w1_i,  args.w1_a_,args.w1_e_,args.w1_i_,  args.c1_a,args.c1_e,args.c1_i,  args.tau, 
                                    args.normalization_case ,args.shell_comand_nu, self.state[-2], args.seg , args.force,    args.tol_inc,args.tol_ecc,args.tol_a,   
                                    args.a_boundary_flag,args.inc_boundary_flag,args.ecc_boundary_flag,   args.a_min,args.a_max,args.inc_min,args.inc_max,args.ecc_min,args.ecc_max,    args.sh_flag,  
                                    target_a,target_ecc,target_inc,   tol_a_low,tol_a_high,    tol_ecc_low,tol_ecc_high,      tol_inc_low,tol_inc_high, weights_manual, intermediate_state_use, RL_algo,self.done_ep_reward,
                                    pre_trained_weights_flag , self.pre_tr_weights_path , milestone, ecc_ms_1_tol_low,ecc_ms_1_tol_high,  a_ms_1_tol_low,a_ms_1_tol_high,  inc_ms_1_tol_low,inc_ms_1_tol_high, ms_reward, 
                                    he_para_flag,   a_conv,ecc_conv,inc_conv,    h_conv,hx_conv,hy_conv,ex_conv,ey_conv,  args.RAAN_conv,args.argp_conv,    simpl_or_exp_rew,
                                    tol_h,tol_hx , tol_hy , tol_ex ,tol_ey,tol_RAAN, tol_argp, 
                                    args.w1_h, args.w1_hx, args.w1_hy, args.w1_ex, args.w1_ey, args.w1_h_, args.w1_hx_, args.w1_hy_, args.w1_ex_, args.w1_ey_, args.c1_h, args.c1_hx, args.c1_hy, args.c1_ex, args.c1_ey,
                                    args.w1_RAAN, args.w1_argp,args.w1_RAAN_, args.w1_argp_,args.c1_RAAN, args.c1_argp,
                                    args.w_h_s, args.w_hx_s, args.w_hy_s, args.w_ex_s, args.w_ey_s, args.w_a_s, args.w_ecc_s, args.w_inc_s,args.reward_weight_upscale,
                                    args.max_nu_ep,  args.max_steps_one_ep, args.phi0,args.sac_paper_cond, self.new_scaling_flag ,
                                    args.discrete_reward, args.w_a_dr, args.w_e_dr, args.w_i_dr, args.w_h_dr, args.w_hx_dr, args.w_hy_dr, args.w_ex_dr, args.w_ey_dr, args.w_argp_dr, args.w_raan_dr,
                                    self.info_file  )
          
  def reset_apend_arrays(self, state, scaled_state):
    self.ep_Cont_len_counter = self.ep_fixed_len_counter = 0
    self.seg_count = 0
    self.ep_time = self.ep_time_1 = 0
    self.ep_counter += 1
    self.ep_counter_SAC_step += 1
    self.acc_reward = 0
    self.Force_1 = args.force
    self.terminate = 0
    
    ecc = math.sqrt(state[3]**2 + state[4]**2)
    a = (state[0]**2 / 398600.4418) / (1 - ecc**2)
    i = (math.asin(math.sqrt(state[1]**2 + state[2]**2) / state[0]) / np.pi) * 180
    ecc_scaled = self.norm.Single_value_Orignal_to_Scaled_st(ecc)
    
    
    self.ecc_history.append([ecc])
    self.ecc_scaled_history.append([ecc_scaled])
    self.a_history.append([a])
    self.inclination_history.append([i])
    
    self.h_history.append([scaled_state[0]])
    self.hx_history.append([scaled_state[1]])
    self.hy_history.append([scaled_state[2]])
    self.ex_history.append([state[3]])
    self.ey_history.append([state[4]])
    self.ex_scaled_history.append([scaled_state[3]])
    self.ey_scaled_history.append([scaled_state[4]])
    self.mass_history.append([scaled_state[5]])
    self.force_history.append([self.Force_1/1000])
    self.time_in_days_history.append([0])
    self.alpha_history.append([])
    self.beta_history.append([])
    self.phi_history.append([])
    self.thrust_history.append([]) 
    self.score_data.append([])
    self.score_detailed_data.append([])
    self.RAAN_history.append([])
    self.argp_history.append([])
    self.nurev_history.append([])
    
    # for visulaizing reward functions
    self.a_plot_history.append([]) 
    self.e_plot_history.append([]) 
    self.i_plot_history.append([]) 
    self.a_exp_plot_history.append([]) 
    self.e_exp_plot_history.append([]) 
    self.i_exp_plot_history.append([]) 
    self.a_total_plot_history.append([]) 
    self.e_total_exp_plot_history.append([]) 
    self.i_total_exp_plot_history.append([]) 
    self.reward_st_plot_history.append([]) 
    self.reward_st1_exp_plot_history.append([]) 
    self.reward_st1_m_st_plot_history.append([]) 
    self.reward_st1_m_st_m_tau_plot_history.append([]) 
    self.reward_st1_m_st_m_tau_m_100rf_plot_history.append([]) 
    self.reward_st1_m_st_m_tau_m_100rf_p_ms_plot_history.append([]) 

    self.initial_state = np.array(args.Lunar_state) if not args.GTO_to_Lunar else  np.array(args.GTO_state)   # [h;hx;hy;ex;ey;mass;time]  superGTO2 Adrean paper   Isp=1500 m0=1000kg F= 1N
    time_in_days = self.initial_state[-1] - state[-1]
    
    if self.testing_weights== 0 or args.single_weight_test == 1: 
      self.env.writing_all_episodes_data ( int(self.ep_counter_SAC_step) , int(0), 
                                float(round(state[0],6)),float(round(state[1],6)),float(round(state[2],6)),float(round(state[3],6)), float(round(state[4],6)),float(round(state[5],6)), 
                                float(round(scaled_state[0],6)),float(round(scaled_state[1],6)),float(round(scaled_state[2],6)),float(round(scaled_state[3],6)), float(round(scaled_state[4],6)),float(round(scaled_state[5],6)), 
                                float(round(state[0],6)), float(round(state[1],6)),float(round(state[2],6)),float(round(state[3],6)),float(round(state[4],6)),float(round(state[5],6)),                                     
                                float(round(ecc,6)), float(round(a,6)), float(round(i,6)), 
                                float(round(time_in_days,6)), float(round(state[-1],6)), float(round(self.phi_0,6)), 
                                float(round(a,6)), float(round(ecc,6)), float(round(i,6)),
                                float(round(state[0],6)), float(round(state[1],6)),float(round(state[2],6)),
                                float(round(state[3],6)), float(round(state[4],6)),
                                float(0), float(0),float(0), float(0),
                                float(0), float(0),float(0), args.seg*(180/3.14),
                                float(0), float(0),float(0), float(0), float(0),float(0),
                                float(0), float(0),float(0), float(0), float(0),float(0),
                                float(0), float(0),float(0), float(args.force), 
                                float(0),float(0), float(0),
                                self.completeName_all_data  )
      
  def step_append_arrays(self, time_in_days, target_state_parameters,scaled_ecc_param,phi_0, action, reward_plot, RAAN, argp, nurev):
    # for plotting append data
    self.h_history[self.ep_counter-1].append(self.state1_ND[0][0])
    self.hx_history[self.ep_counter-1].append(self.state1_ND[0][1])
    self.hy_history[self.ep_counter-1].append(self.state1_ND[0][2])
    self.ex_history[self.ep_counter-1].append(self.orig_state[0][3])
    self.ey_history[self.ep_counter-1].append(self.orig_state[0][4])
    
    self.ex_scaled_history[self.ep_counter-1].append(self.state1_ND[0][3])
    self.ey_scaled_history[self.ep_counter-1].append(self.state1_ND[0][4])
    
    self.mass_history[self.ep_counter-1].append(self.state1_ND[0][5])
    self.force_history[self.ep_counter-1].append(self.Force_1)
    self.time_in_days_history[self.ep_counter-1].append(time_in_days)
    self.ecc_history[self.ep_counter-1].append(target_state_parameters[0])
    self.ecc_scaled_history[self.ep_counter-1].append(scaled_ecc_param[0])
    self.a_history[self.ep_counter-1].append(target_state_parameters[1])
    self.inclination_history[self.ep_counter-1].append(target_state_parameters[2])
    self.phi_history[self.ep_counter-1].append(phi_0)
    self.alpha_history[self.ep_counter-1].append(action[0]*np.pi)
    self.beta_history[self.ep_counter-1].append(action[1]*(np.pi/2))
    self.acc_reward = self.acc_reward + self.reward
    self.score_data[self.ep_counter-1] = self.acc_reward 
    self.score_detailed_data[self.ep_counter-1].append(self.acc_reward)
    self.RAAN_history[self.ep_counter-1].append(RAAN)
    self.argp_history[self.ep_counter-1].append(argp)
    self.nurev_history[self.ep_counter-1].append(nurev)
    
    # for visulaizing reward functions
    self.a_plot_history[self.ep_counter-1].append(reward_plot[0])
    self.e_plot_history[self.ep_counter-1].append(reward_plot[1])
    self.i_plot_history[self.ep_counter-1].append(reward_plot[2])
    self.a_exp_plot_history[self.ep_counter-1].append(reward_plot[3])
    self.e_exp_plot_history[self.ep_counter-1].append(reward_plot[4])
    self.i_exp_plot_history[self.ep_counter-1].append(reward_plot[5])
    self.a_total_plot_history[self.ep_counter-1].append(reward_plot[6])
    self.e_total_exp_plot_history[self.ep_counter-1].append(reward_plot[7])
    self.i_total_exp_plot_history[self.ep_counter-1].append(reward_plot[8])
    self.reward_st_plot_history[self.ep_counter-1].append(reward_plot[9])
    self.reward_st1_exp_plot_history[self.ep_counter-1].append(reward_plot[10])
    self.reward_st1_m_st_plot_history[self.ep_counter-1].append(reward_plot[11])
    self.reward_st1_m_st_m_tau_plot_history[self.ep_counter-1].append(reward_plot[12])
    self.reward_st1_m_st_m_tau_m_100rf_plot_history[self.ep_counter-1].append(reward_plot[13])
    self.reward_st1_m_st_m_tau_m_100rf_p_ms_plot_history[self.ep_counter-1].append(reward_plot[14])
    

  def redflag_plotvariables(self, target_state_parameters, scaled_ecc_param):
      self.env.plot_variable("H", self.h_history, self.folder_path_h, self.ep_counter, h_flag=1, tol_h=self.tol_h)
      # self.env.plot_variable("Hx", self.hx_history, self.folder_path_hx, self.ep_counter)
      # self.env.plot_variable("Hy", self.hy_history, self.folder_path_hy, self.ep_counter)
      # self.env.plot_variable("ex", self.ex_history, self.folder_path_ex, self.ep_counter)
      # self.env.plot_variable("ey", self.ey_history, self.folder_path_ey, self.ep_counter)
      self.env.plot_two_variable("hx_hy","hx","hy", self.hx_history,self.hy_history,  self.folder_path_hx_hy, self.ep_counter, hx_hy_flag=1, tolhx= self.tol_hx, tolhy= self.tol_hy)
      self.env.plot_two_variable("ex_ey","ex","ey", self.ex_history,self.ey_history,  self.folder_path_ex_ey, self.ep_counter, ex_ey_flag=1, tolex= self.tol_ex, toley= self.tol_ey)
      self.env.plot_two_variable("ex_ey_scaled","ex_sc","ey_sc", self.ex_scaled_history,self.ey_scaled_history,  self.folder_path_exey_sc, self.ep_counter, ex_ey_flag=1, tolex= scaled_ecc_param[10], toley= scaled_ecc_param[10], scaled=1, target_ex=scaled_ecc_param[11], target_ey=scaled_ecc_param[12])
      self.env.plot_variable("ecc_scaled", self.ecc_scaled_history, self.folder_path_ecc_sc, self.ep_counter, flag_ter_values=1, tsp=scaled_ecc_param, tsp_indexes=[3,4])
      self.env.plot_variable("ecc", self.ecc_history, self.folder_path_ecc, self.ep_counter, flag_ter_values=1, tsp=target_state_parameters, tsp_indexes=[3,4,15,16], flag_ms_1 = self.milestone)
      self.env.plot_variable("a", self.a_history, self.folder_path_a, self.ep_counter, flag_ter_values=1, tsp=target_state_parameters, tsp_indexes=[5,6,17,18], flag_ms_1 = self.milestone)
      self.env.plot_variable("inc", self.inclination_history, self.folder_path_inc, self.ep_counter, flag_ter_values=1, tsp=target_state_parameters, tsp_indexes=[7,8,19,20], flag_ms_1 = self.milestone)  
      self.env.plot_variable("RAAN", self.RAAN_history, self.folder_path_RAAN, self.ep_counter, flag_ter_values=1, tsp=target_state_parameters, tsp_indexes=[21,22,15,16])
      self.env.plot_variable("argp", self.argp_history, self.folder_path_argp, self.ep_counter, flag_ter_values=1, tsp=target_state_parameters, tsp_indexes=[23,24,17,18])
      self.env.plot_variable("nurev", self.nurev_history, self.folder_path_nurev, self.ep_counter)
      self.env.plot_variable("mass", self.mass_history, self.folder_path_mass, self.ep_counter)
      self.env.plot_variable("force", self.force_history, self.folder_path_force, self.ep_counter)
      self.env.plot_variable("Time(days)", self.time_in_days_history, self.folder_path_time_in_days, self.ep_counter)
      self.env.plot_variable("Reward", self.score_detailed_data, self.folder_path_sum_reward, self.ep_counter)
      self.env.plot_variable("Phi", self.phi_history, self.folder_path_phi, self.ep_counter)
      self.env.plot_variable("RAAN", self.RAAN_history, self.folder_path_RAAN, self.ep_counter, flag_ter_values=1, tsp=target_state_parameters, tsp_indexes=[21,22,15,16])
      self.env.plot_variable("argp", self.argp_history, self.folder_path_argp, self.ep_counter, flag_ter_values=1, tsp=target_state_parameters, tsp_indexes=[23,24,17,18])
      self.env.plot_variable("nurev", self.nurev_history, self.folder_path_nurev, self.ep_counter)
      self.env.plot_two_variable("actions","alpha","beta", self.alpha_history,self.beta_history, self.folder_path_actions, self.ep_counter)
      
      # if self.he_para_flag:
      #   self.env.plot_variable("diff_h", self.a_plot_history, self.folder_path_reward_analysis , self.ep_counter, flag_ep_nu_start=1)
      #   self.env.plot_two_variable("diff_hx_hy","diff_hx","diff_hy", self.e_plot_history,self.i_plot_history,  self.folder_path_reward_analysis, self.ep_counter, flag_ep_nu_start =1)
      #   self.env.plot_two_variable("diff_e", "diff_ex", "diff_ey",self.a_exp_plot_history,self.e_exp_plot_history, self.folder_path_reward_analysis , self.ep_counter, flag_ep_nu_start=1)     
      #   self.env.plot_variable("diff_h_w", self.i_exp_plot_history, self.folder_path_reward_analysis , self.ep_counter, flag_ep_nu_start=1)
      #   self.env.plot_two_variable("diff_hx_hy_W","diff_hx_w","diff_hy_w", self.a_total_plot_history,self.e_total_exp_plot_history,  self.folder_path_reward_analysis, self.ep_counter, flag_ep_nu_start =1)
      #   self.env.plot_two_variable("diff_e_w", "diff_ex_w", "diff_ey_w",self.i_total_exp_plot_history,self.reward_st_plot_history, self.folder_path_reward_analysis , self.ep_counter, flag_ep_nu_start=1)
      #   self.env.plot_variable("diff_h_w_up", self.reward_st1_exp_plot_history, self.folder_path_reward_analysis , self.ep_counter, flag_ep_nu_start=1)
      #   self.env.plot_two_variable("diff_hx_hy_W_UP","diff_hx_w_up","diff_hy_w_up", self.reward_st1_m_st_plot_history,self.reward_st1_m_st_m_tau_plot_history,  self.folder_path_reward_analysis, self.ep_counter, flag_ep_nu_start =1)
      #   self.env.plot_two_variable("diff_e_w_up", "diff_ex_w_up", "diff_ey_w_up",self.reward_st1_m_st_m_tau_m_100rf_plot_history,self.reward_st1_m_st_m_tau_m_100rf_p_ms_plot_history, self.folder_path_reward_analysis , self.ep_counter, flag_ep_nu_start=1)
      # else:
      #   self.env.plot_variable("r_a", self.a_plot_history, self.folder_path_reward_analysis , self.ep_counter, flag_ep_nu_start=1)
      #   self.env.plot_variable("r_e", self.e_plot_history, self.folder_path_reward_analysis , self.ep_counter, flag_ep_nu_start=1)
      #   self.env.plot_variable("r_i", self.i_plot_history, self.folder_path_reward_analysis , self.ep_counter, flag_ep_nu_start=1)
      #   self.env.plot_variable("r_a_exp", self.a_exp_plot_history, self.folder_path_reward_analysis , self.ep_counter, flag_ep_nu_start=1)
      #   self.env.plot_variable("r_e_exp", self.e_exp_plot_history, self.folder_path_reward_analysis , self.ep_counter, flag_ep_nu_start=1)
      #   self.env.plot_variable("r_i_exp", self.i_exp_plot_history, self.folder_path_reward_analysis , self.ep_counter, flag_ep_nu_start=1)
      #   self.env.plot_variable("r_a_total", self.a_total_plot_history, self.folder_path_reward_analysis , self.ep_counter, flag_ep_nu_start=1)
      #   self.env.plot_variable("r_e_total", self.e_total_exp_plot_history, self.folder_path_reward_analysis , self.ep_counter, flag_ep_nu_start=1)
      #   self.env.plot_variable("r_i_total", self.i_total_exp_plot_history, self.folder_path_reward_analysis , self.ep_counter, flag_ep_nu_start=1)
      #   self.env.plot_two_variable("r_st_st1", "r_st", "r_st1",self.reward_st_plot_history,self.reward_st1_exp_plot_history, self.folder_path_reward_analysis , self.ep_counter, flag_ep_nu_start=1)
      #   self.env.plot_variable("r_st1_m_st", self.reward_st1_m_st_plot_history, self.folder_path_reward_analysis , self.ep_counter, flag_ep_nu_start=1)
      #   self.env.plot_variable("r_st1_m_st_m_tau", self.reward_st1_m_st_m_tau_plot_history, self.folder_path_reward_analysis , self.ep_counter, flag_ep_nu_start=1)
      #   self.env.plot_variable("r_st1_m_st_m_tau_m_rf", self.reward_st1_m_st_m_tau_m_100rf_plot_history, self.folder_path_reward_analysis , self.ep_counter, flag_ep_nu_start=1)
      #   self.env.plot_variable("r_st1_m_st_m_tau_m_rf_p_ms", self.reward_st1_m_st_m_tau_m_100rf_p_ms_plot_history, self.folder_path_reward_analysis , self.ep_counter, flag_ep_nu_start=1)
  
  def step_writing_Successful_episodes(self, time_in_days, RAAN, argp, nurev):
      done_data = [(self.ep_counter-1),  self.ep_Cont_len_counter,  float(round(self.acc_reward,6)), float(round(time_in_days,6)), self.completeName_successful ]
      self.env.writing_Successful_episodes ( int(self.success_counter), int(done_data[0]), int(done_data[1]), 
                                            (float(done_data[2])),  (float(done_data[3])), 
                                            float(44364- (self.a_history[self.ep_counter-1][-1])), float(self.inclination_history[self.ep_counter-1][-1]), float(self.ecc_history[self.ep_counter-1][-1]),  float(self.mass_history[self.ep_counter-1][-1]), 
                                            float(self.next_state[0]),  float(self.next_state[1]), float(self.next_state[2]), float(self.next_state[3]), float(self.next_state[4]),
                                            float(RAAN), float(argp), float(nurev), self.completeName_successful  )
      
  def step_writing_final_states(self, RAAN, argp, nurev):  
      self.env.writing_final_states(  float(self.next_state[0]), float(self.next_state[1]), float(self.next_state[2]), 
                                            float(self.next_state[3]), float(self.next_state[4]), self.phi_0,
                                            float(self.time) ,  float(self.next_state[5]) , float(RAAN), float(argp), float(nurev), self.write_final_state )
  
  def succesful_folder_aei_plotvariables(self, target_state_parameters):
      self.env.plot_variable("ecc", self.ecc_history, self.figure_file_ecc, self.ep_counter, flag_ter_values=1, tsp=target_state_parameters, tsp_indexes=[3,4,15,16], flag_ms_1 = self.milestone)
      self.env.plot_variable("a", self.a_history, self.figure_file_a, self.ep_counter, flag_ter_values=1, tsp=target_state_parameters, tsp_indexes=[5,6,17,18], flag_ms_1 = self.milestone)
      self.env.plot_variable("inc", self.inclination_history, self.figure_file_inclination, self.ep_counter, flag_ter_values=1, tsp=target_state_parameters, tsp_indexes=[7,8,19,20], flag_ms_1 = self.milestone)
      self.env.plot_variable("Score_value", self.score_data, self.figure_file_reward, self.ep_counter, all_episode_plot_flag=1)
      self.env.plot_variable("Time_value", self.time_history, self.figure_file_time_in_days, self.ep_counter, plotting_time=1)

  def succesful_folder_plotvariables(self,target_state_parameters, scaled_ecc_param):
      if not os.path.exists(self.successful_episodes ):
          os.makedirs(self.successful_episodes)
      # self.Successful_Episode_num = os.path.join(self.successful_episodes +"/_ep_"+ str(self.ep_counter-1) )
      self.Successful_Episode_num = os.path.join(f"{self.successful_episodes}/{'_ep_'}{str(self.ep_counter-1)}")
      if not os.path.exists(self.Successful_Episode_num ):
            os.makedirs(self.Successful_Episode_num ) 
      self.env.plot_variable("H", self.h_history, self.Successful_Episode_num, self.ep_counter, flag_saving_with_no_ep_nu =1,  h_flag=1, tol_h=self.tol_h)
      self.env.plot_two_variable("hx_hy","hx","hy", self.hx_history,self.hy_history,  self.Successful_Episode_num, self.ep_counter, flag_saving_with_no_ep_nu =1, hx_hy_flag=1, tolhx= self.tol_hx, tolhy= self.tol_hy)
      self.env.plot_two_variable("ex_ey","ex","ey", self.ex_history,self.ey_history,  self.Successful_Episode_num, self.ep_counter, flag_saving_with_no_ep_nu =1, ex_ey_flag=1, tolex= self.tol_ex, toley= self.tol_ey)
      self.env.plot_two_variable("ex_ey_scaled","ex_sc","ey_sc", self.ex_scaled_history,self.ey_scaled_history,   self.Successful_Episode_num, self.ep_counter, flag_saving_with_no_ep_nu =1, ex_ey_flag=1, tolex= scaled_ecc_param[10], toley= scaled_ecc_param[10], scaled=1, target_ex=scaled_ecc_param[11], target_ey=scaled_ecc_param[12])
      self.env.plot_variable("ecc_scaled", self.ecc_scaled_history,  self.Successful_Episode_num, self.ep_counter, flag_saving_with_no_ep_nu =1, flag_ter_values=1, tsp=scaled_ecc_param, tsp_indexes=[3,4])
      self.env.plot_variable("ecc", self.ecc_history, self.Successful_Episode_num, self.ep_counter, flag_ter_values=1, tsp=target_state_parameters, tsp_indexes=[3,4,15,16], flag_saving_with_no_ep_nu =1, flag_ms_1 = self.milestone)
      self.env.plot_variable("a", self.a_history, self.Successful_Episode_num, self.ep_counter, flag_ter_values=1, tsp=target_state_parameters, tsp_indexes=[5,6,17,18], flag_saving_with_no_ep_nu =1, flag_ms_1 = self.milestone)
      self.env.plot_variable("inc", self.inclination_history, self.Successful_Episode_num, self.ep_counter, flag_ter_values=1, tsp=target_state_parameters, tsp_indexes=[7,8,19,20], flag_saving_with_no_ep_nu =1, flag_ms_1 = self.milestone)
      self.env.plot_variable("mass", self.mass_history, self.Successful_Episode_num, self.ep_counter, flag_saving_with_no_ep_nu =1)
      self.env.plot_variable("force", self.force_history, self.Successful_Episode_num, self.ep_counter, flag_saving_with_no_ep_nu =1)
      self.env.plot_variable("Reward", self.score_detailed_data, self.Successful_Episode_num, self.ep_counter, flag_saving_with_no_ep_nu =1)
      self.env.plot_variable("Phi", self.phi_history, self.Successful_Episode_num, self.ep_counter, flag_saving_with_no_ep_nu =1)
      self.env.plot_variable("RAAN", self.RAAN_history, self.Successful_Episode_num, self.ep_counter, flag_ter_values=1, tsp=target_state_parameters, tsp_indexes=[21,22,15,16],flag_saving_with_no_ep_nu =1)
      self.env.plot_variable("argp", self.argp_history, self.Successful_Episode_num, self.ep_counter, flag_ter_values=1, tsp=target_state_parameters, tsp_indexes=[23,24,17,18],flag_saving_with_no_ep_nu =1)
      self.env.plot_variable("nurev", self.nurev_history, self.Successful_Episode_num, self.ep_counter, flag_saving_with_no_ep_nu =1)
      self.env.plot_two_variable("actions","alpha","beta", self.alpha_history,self.beta_history,   self.Successful_Episode_num, self.ep_counter, flag_saving_with_no_ep_nu =1)

      if self.he_para_flag:
        self.env.plot_variable("diff_h", self.a_plot_history, self.Successful_Episode_num , self.ep_counter, flag_saving_with_no_ep_nu=1)
        self.env.plot_two_variable("diff_hx_hy","diff_hx","diff_hy", self.e_plot_history,self.i_plot_history,  self.Successful_Episode_num, self.ep_counter, flag_saving_with_no_ep_nu =1)
        self.env.plot_two_variable("diff_e", "diff_ex", "diff_ey",self.a_exp_plot_history,self.e_exp_plot_history, self.Successful_Episode_num , self.ep_counter, flag_saving_with_no_ep_nu=1)     
        self.env.plot_variable("diff_h_w", self.i_exp_plot_history, self.Successful_Episode_num , self.ep_counter, flag_saving_with_no_ep_nu=1)
        self.env.plot_two_variable("diff_hx_hy_W","diff_hx_w","diff_hy_w", self.a_total_plot_history,self.e_total_exp_plot_history,  self.Successful_Episode_num, self.ep_counter, flag_saving_with_no_ep_nu =1)
        self.env.plot_two_variable("diff_e_w", "diff_ex_w", "diff_ey_w",self.i_total_exp_plot_history,self.reward_st_plot_history, self.Successful_Episode_num , self.ep_counter, flag_saving_with_no_ep_nu=1)
        self.env.plot_variable("diff_h_w_up", self.reward_st1_exp_plot_history, self.Successful_Episode_num , self.ep_counter, flag_saving_with_no_ep_nu=1)
        self.env.plot_two_variable("diff_hx_hy_W_UP","diff_hx_w_up","diff_hy_w_up", self.reward_st1_m_st_plot_history,self.reward_st1_m_st_m_tau_plot_history,  self.Successful_Episode_num, self.ep_counter, flag_saving_with_no_ep_nu =1)
        self.env.plot_two_variable("diff_e_w_up", "diff_ex_w_up", "diff_ey_w_up",self.reward_st1_m_st_m_tau_m_100rf_plot_history,self.reward_st1_m_st_m_tau_m_100rf_p_ms_plot_history, self.Successful_Episode_num , self.ep_counter, flag_saving_with_no_ep_nu=1)
      else:
        self.env.plot_variable("r_a", self.a_plot_history, self.Successful_Episode_num , self.ep_counter, flag_saving_with_no_ep_nu=1)
        self.env.plot_variable("r_e", self.e_plot_history, self.Successful_Episode_num , self.ep_counter, flag_saving_with_no_ep_nu=1)
        self.env.plot_variable("r_i", self.i_plot_history, self.Successful_Episode_num , self.ep_counter, flag_saving_with_no_ep_nu=1)
        self.env.plot_variable("r_a_exp", self.a_exp_plot_history, self.Successful_Episode_num , self.ep_counter, flag_saving_with_no_ep_nu=1)
        self.env.plot_variable("r_e_exp", self.e_exp_plot_history, self.Successful_Episode_num , self.ep_counter, flag_saving_with_no_ep_nu=1)
        self.env.plot_variable("r_i_exp", self.i_exp_plot_history, self.Successful_Episode_num , self.ep_counter, flag_saving_with_no_ep_nu=1)
        self.env.plot_variable("r_a_total", self.a_total_plot_history, self.Successful_Episode_num , self.ep_counter, flag_saving_with_no_ep_nu=1)
        self.env.plot_variable("r_e_total", self.e_total_exp_plot_history, self.Successful_Episode_num , self.ep_counter, flag_saving_with_no_ep_nu=1)
        self.env.plot_variable("r_i_total", self.i_total_exp_plot_history, self.Successful_Episode_num , self.ep_counter, flag_saving_with_no_ep_nu=1)
        self.env.plot_two_variable("r_st_st1", "r_st", "r_st1",self.reward_st_plot_history,self.reward_st1_exp_plot_history, self.Successful_Episode_num , self.ep_counter, flag_saving_with_no_ep_nu=1)
        self.env.plot_variable("r_st1_m_st", self.reward_st1_m_st_plot_history, self.Successful_Episode_num , self.ep_counter, flag_saving_with_no_ep_nu=1)
        self.env.plot_variable("r_st1_m_st_m_tau", self.reward_st1_m_st_m_tau_plot_history, self.Successful_Episode_num , self.ep_counter, flag_saving_with_no_ep_nu=1)
        self.env.plot_variable("r_st1_m_st_m_tau_m_rf", self.reward_st1_m_st_m_tau_m_100rf_plot_history, self.Successful_Episode_num , self.ep_counter, flag_saving_with_no_ep_nu=1)
        self.env.plot_variable("r_st1_m_st_m_tau_m_rf_p_ms", self.reward_st1_m_st_m_tau_m_100rf_p_ms_plot_history, self.Successful_Episode_num , self.ep_counter, flag_saving_with_no_ep_nu=1)
        
  def step_print_values(self, time_in_days, target_state_parameters, raise_error_counter, reward_plot,current_state, target_state):  
      total_steps = args.max_nu_ep  * 2000
      self.target_inc = ((math.asin(math.sqrt((target_state[1]**2)+(target_state[2]**2))/target_state[0])) / np.pi)*180 
      self.target_ecc = math.sqrt((target_state[3]**2)+(target_state[4]**2))

      if self.he_para_flag:
        print(self.counter,"/",total_steps," Sh:",self.sh_command_nu,"  Ep:", self.ep_counter_SAC_step , "  Ep_len :", self.ep_fixed_len_counter , "  step_R:" , float(round(self.reward,2)),   "  Acc_R: ", float(round(self.acc_reward,2)), "  days: ", float(round(time_in_days,4)),
              "  ecc:" , float(round(target_state_parameters[0],6)), "  a_diff  :" , float(round((target_state_parameters[11]) - (self.a_history[self.ep_counter-1][-1]),2)),"  inc:" , float(round(target_state_parameters[2],4)),
              "  m:" , float(round(self.state1_ND[0][5],3)),
              "  df_h:" , float(round(np.abs((target_state[0])-current_state[0]),1)), "  df_hx,hy  :" , float(round(np.abs((target_state[1])-current_state[1]),1))," ",float(round(np.abs((target_state[2])-current_state[2]),1)),    
              "  df_ex,ey:" , float(round(np.abs((target_state[3])-current_state[3]),4))," ",float(round(np.abs((target_state[4])-current_state[4]),4)),
              "  df_ec:" , float(round(np.abs((self.target_ecc)-target_state_parameters[0]),4)), "  df_i:" , float(round(np.abs((self.target_inc)-target_state_parameters[2]),3)))
              # "  fl_eai:" , [target_state_parameters[12], target_state_parameters[13], target_state_parameters[14]] )
              
              # "  diff_h:" , float(round(np.abs((np.sqrt(reward_plot[0])*101055.56404)-101055.56404),4)), "  diff_hx,hy  :" , float(round(np.abs((np.sqrt(reward_plot[1])*101055.56404)-8993.12420),4))," ",float(round(np.abs((np.sqrt(reward_plot[2])*101055.56404)-44988.20968),4)),    
              # "  diff_ex_ey:" , float(round(np.sqrt(reward_plot[3]),4))," ",float(round(np.sqrt(reward_plot[4]),4)))
      else:
        print(self.counter,"/",total_steps," Sh_cmnd:",self.sh_command_nu,"  Ep:", self.ep_counter_SAC_step , "  Ep_len :", self.ep_fixed_len_counter , "  step_reward:" , float(round(self.reward,6)),   "  Acc_reward: ", float(round(self.acc_reward,6)), "  days: ", float(round(time_in_days,6)),
              "  ecc:" , float(round(target_state_parameters[0],6)), "  a_diff  :" , float(round((target_state_parameters[11]) - (self.a_history[self.ep_counter-1][-1]),4)),"  inc:" , float(round(target_state_parameters[2],4)),
              "  m:" , float(round(self.state1_ND[0][5],3)),
              "  df_h:" , float(round(np.abs((target_state[0])-current_state[0]),1)), "  df_hx,hy  :" , float(round(np.abs((target_state[1])-current_state[1]),1))," ",float(round(np.abs((target_state[2])-current_state[2]),1)),    
              "  df_ex_ey:" , float(round(np.abs((target_state[3])-current_state[3]),1))," ",float(round(np.abs((target_state[4])-current_state[4]),1)),
              "  flag_eai:" , [target_state_parameters[12], target_state_parameters[13], target_state_parameters[14]] )
        
  def step_all_ep_writing(self, time_in_days, time, target_state_parameters, action, reward_plot, RAAN, argp, nurev, segment, current_state, target_state):    
      self.env.writing_all_episodes_data ( int(self.ep_counter_SAC_step) , int(self.ep_fixed_len_counter), 
                                          float(round(self.orig_state[0][0],6)),float(round(self.orig_state[0][1],6)),float(round(self.orig_state[0][2],6)),float(round(self.orig_state[0][3],6)), float(round(self.orig_state[0][4],6)),float(round(self.orig_state[0][5],6)), 
                                          float(round(self.state1_ND[0][0],6)),float(round(self.state1_ND[0][1],6)),float(round(self.state1_ND[0][2],6)),float(round(self.state1_ND[0][3],6)), float(round(self.state1_ND[0][4],6)),float(round(self.state1_ND[0][5],6)), 
                                          float(round(self.state1[0][0],6)), float(round(self.state1[0][1],6)),float(round(self.state1[0][2],6)),float(round(self.state1[0][3],6)),float(round(self.state1[0][4],6)),float(round(self.state1[0][5],6)),                                     
                                          float(round(target_state_parameters[0],6)), float(round(target_state_parameters[1],6)), float(round(target_state_parameters[2],6)), 
                                          float(round(time_in_days,4)), float(round(time,6)), float(round(self.phi_0,6)), 
                                          float(round((target_state_parameters[11]) - (self.a_history[self.ep_counter-1][-1]),2)), float(round(np.abs((self.target_ecc)-target_state_parameters[0]),4)), float(round(np.abs((self.target_inc)-target_state_parameters[2]),3)),
                                          float(round(np.abs((target_state[0])-current_state[0]),1)),  float(round(np.abs((target_state[1])-current_state[1]),1)),  float(round(np.abs((target_state[2])-current_state[2]),1)),
                                          float(round(np.abs((target_state[3])-current_state[3]),4)),  float(round(np.abs((target_state[4])-current_state[4]),4)),
                                          float(round(action[0],4)), float(round(action[1],4)),float(round(self.reward,4)), float(round(self.acc_reward,4)),
                                          target_state_parameters[12] , target_state_parameters[13], target_state_parameters[14], segment*(180/3.14),
                                          float(round(reward_plot[0],6)), float(round(reward_plot[1],6)),float(round(reward_plot[2],6)), float(round(reward_plot[3],6)), float(round(reward_plot[4],6)),float(round(reward_plot[5],6)),
                                          float(round(reward_plot[6],6)), float(round(reward_plot[7],6)),float(round(reward_plot[8],6)), float(round(reward_plot[9],6)), float(round(reward_plot[10],6)),float(round(reward_plot[11],6)),
                                          float(round(reward_plot[12],6)), float(round(reward_plot[13],6)),float(round(reward_plot[14],6)), 
                                          float(round(self.force_history[self.ep_counter-1][-1],4)), 
                                          float(round(RAAN,4)),float(round(argp,4)), float(round(nurev,4)),
                                          self.completeName_all_data  )
      if self.done:
        self.env.writing_all_episodes_data ( int(self.ep_counter_SAC_step) , int(self.ep_fixed_len_counter), 
                                          float(round(self.orig_state[0][0],6)),float(round(self.orig_state[0][1],6)),float(round(self.orig_state[0][2],6)),float(round(self.orig_state[0][3],6)), float(round(self.orig_state[0][4],6)),float(round(self.orig_state[0][5],6)), 
                                          float(round(self.state1_ND[0][0],6)),float(round(self.state1_ND[0][1],6)),float(round(self.state1_ND[0][2],6)),float(round(self.state1_ND[0][3],6)), float(round(self.state1_ND[0][4],6)),float(round(self.state1_ND[0][5],6)), 
                                          float(round(self.state1[0][0],6)), float(round(self.state1[0][1],6)),float(round(self.state1[0][2],6)),float(round(self.state1[0][3],6)),float(round(self.state1[0][4],6)),float(round(self.state1[0][5],6)),                                     
                                          float(round(target_state_parameters[0],6)), float(round(target_state_parameters[1],6)), float(round(target_state_parameters[2],6)), 
                                          float(round(time_in_days,4)), float(round(time,6)), float(round(self.phi_0,6)), 
                                          float(round((target_state_parameters[11]) - (self.a_history[self.ep_counter-1][-1]),2)), float(round(np.abs((self.target_ecc)-target_state_parameters[0]),4)), float(round(np.abs((self.target_inc)-target_state_parameters[2]),3)),
                                          float(round(np.abs((target_state[0])-current_state[0]),1)),  float(round(np.abs((target_state[1])-current_state[1]),1)),  float(round(np.abs((target_state[2])-current_state[2]),1)),
                                          float(round(np.abs((target_state[3])-current_state[3]),4)),  float(round(np.abs((target_state[4])-current_state[4]),4)),
                                          float(round(action[0],4)), float(round(action[1],4)),float(round(self.reward,4)), float(round(self.acc_reward,4)),
                                          target_state_parameters[12] , target_state_parameters[13], target_state_parameters[14], segment*(180/3.14),
                                          float(round(reward_plot[0],6)), float(round(reward_plot[1],6)),float(round(reward_plot[2],6)), float(round(reward_plot[3],6)), float(round(reward_plot[4],6)),float(round(reward_plot[5],6)),
                                          float(round(reward_plot[6],6)), float(round(reward_plot[7],6)),float(round(reward_plot[8],6)), float(round(reward_plot[9],6)), float(round(reward_plot[10],6)),float(round(reward_plot[11],6)),
                                          float(round(reward_plot[12],6)), float(round(reward_plot[13],6)),float(round(reward_plot[14],6)), 
                                          float(round(self.force_history[self.ep_counter-1][-1],4)), 
                                          float(round(RAAN,4)),float(round(argp,4)), float(round(nurev,4)),
                                          self.completeName_all_data  )
  def compute_reward(self, achieved_goal, desired_goal, info):
        # Use the achieved_goal, desired_goal, and any additional information to calculate the reward
        
        w1_aei  = [[self.weights["w1"]["a"]] , [self.weights["w1"]["e"]], [self.weights["w1"]["i"]]]                            
        w1_aei_ = [[self.weights["w1_"]["a_"]] , [self.weights["w1_"]["e_"]], [self.weights["w1_"]["i_"]] ]       # test weighrts for normalized inputs
        c1_aei  = [self.weights["c1"]["a"] , self.weights["c1"]["e"], self.weights["c1"]["i"]]   
        tauu_aei= self.weights["tau"]
        
        
        ecc_prev = math.sqrt((self.state1_ND[3])**2 + (self.state1_ND[4])**2)
        ecc_new = math.sqrt((self.observation_1[3])**2 + (self.observation_1[4])**2)
        ecc_target = math.sqrt((self.ND_tar_state[3])**2 + (self.ND_tar_state[4])**2)
        
        a_prev = (((self.state1_ND[0])**2) /1) / ( 1- (ecc_prev **2))
        a_new = (((self.observation_1[0])**2) /1) / ( 1- (ecc_new **2))
        a_target = (((self.ND_tar_state[0])**2) /1) / ( 1- (ecc_target **2))
        
        i_prev_1 = ((math.asin (math.sqrt((self.state1_ND[1]**2)+(self.state1_ND[2]**2))/self.state1_ND[0])) / np.pi)*180
        i_new_1 = ((math.asin (math.sqrt((self.observation_1[1]**2)+(self.observation_1[2]**2))/self.observation_1[0])) / np.pi)*180
        i_target_1 = ((math.asin (math.sqrt((self.ND_tar_state[1]**2)+(self.ND_tar_state[2]**2))/self.ND_tar_state[0])) / np.pi)*180
    
        i_prev = i_prev_1 /10
        i_new = i_new_1 /10
        i_target = i_target_1 /10  
        
        # st_a_e_i_prev = [[a_prev], [ecc_prev], [i_prev]]
        # st_a_e_i_new = [[a_new], [ecc_new], [i_new]]
        # st_a_e_i_target = [[a_target], [ecc_target], [i_target]]
       
        # exp_value_t_aei      =  0 
        # exp_value_t_plus_1_aei =  0 
       
        # # r = -w abs(s_t-s_tar) + SUM_i [c_i e^{-w' abs(s_t - s_tar)} ]
        # for i in range(0,3):
        #   exp_value_t_aei        =  exp_value_t_aei        + ( c1_aei[i] * math.exp(-(w1_aei_[i] * abs(np.subtract(st_a_e_i_prev[i], st_a_e_i_target[i])) ) ) )
        #   exp_value_t_plus_1_aei =  exp_value_t_plus_1_aei + ( c1_aei[i] * math.exp(-(w1_aei_[i] * abs(np.subtract(st_a_e_i_new[i], st_a_e_i_target[i])) ) ) )
       
        # phi_st_aei        = - np.dot( np.transpose(np.array(w1_aei)) , abs(np.subtract(st_a_e_i_prev ,st_a_e_i_target))) + exp_value_t_aei      
        # phi_st_plus_1_aei = - np.dot( np.transpose(np.array(w1_aei)) , abs(np.subtract(st_a_e_i_new ,st_a_e_i_target))) + exp_value_t_plus_1_aei 
        
        # reward = phi_st_plus_1_aei  - phi_st_aei - tauu_aei + (self.done_ep_reward * self.done) - (self.rf *self.red_flag) 
        
       
        
        a_diff = np.abs(float(round(achieved_goal[0][0],6)) - float(round(desired_goal[0][0],6)))
        e_diff = np.abs(float(round(achieved_goal[1][0],6)) - float(round(desired_goal[1][0],6)))
        i_diff = np.abs(float(round(achieved_goal[2][0],6)) - float(round(desired_goal[2][0],6)))

        a_reward = 100 * a_diff * (100  * np.exp(-30 * a_diff))
        e_reward = 76 * e_diff  * (20   * np.exp(-40 * e_diff))
        i_reward = 50 * i_diff  * (50   * np.exp(-40 * i_diff))
        
        # a_reward = self.weights["w1"]["a"] * a_diff * (self.weights["w1_"]["a_"] * np.exp(-self.weights["c1"]["a"] * a_diff))
        # e_reward = self.weights["w1"]["e"] * e_diff * (self.weights["w1_"]["e_"] * np.exp(-self.weights["c1"]["e"] * e_diff))
        # i_reward = self.weights["w1"]["i"] * i_diff * (self.weights["w1_"]["i_"] * np.exp(-self.weights["c1"]["i"] * i_diff))

        reward = -(a_reward + e_reward + i_reward)  -2 + (self.done_ep_reward * self.done) - (self.rf * self.red_flag)
        
        
        # a_diff_pre = np.abs(a_prev   - float(round(desired_goal[0][0],6)))
        # e_diff_pre = np.abs(ecc_prev - float(round(desired_goal[0][0],6)))
        # i_diff_pre = np.abs(i_prev   - float(round(desired_goal[0][0],6)))
        
        # a_reward_pre = 800000 * a_diff_pre * (360000 * np.exp(-30 * a_diff_pre))
        # e_reward_pre = 560000 * e_diff_pre * (120000 * np.exp(-40 * e_diff_pre))
        # i_reward_pre = 150000 * i_diff_pre * (45000 * np.exp(-40 * i_diff_pre))
        
        # a_reward_pre = self.weights["w1"]["a"] * a_diff_pre * (self.weights["w1_"]["a_"] * np.exp(-self.weights["c1"]["a"] * a_diff_pre))
        # e_reward_pre = self.weights["w1"]["e"] * e_diff_pre * (self.weights["w1_"]["e_"] * np.exp(-self.weights["c1"]["e"] * e_diff_pre))
        # i_reward_pre = self.weights["w1"]["i"] * i_diff_pre * (self.weights["w1_"]["i_"] * np.exp(-self.weights["c1"]["i"] * i_diff_pre))

        # reward_pre = -(a_reward_pre + e_reward_pre + i_reward_pre)
        
        # reward_1 = reward - reward_pre  - tauu_aei + (self.done_ep_reward * self.done) - (self.rf * self.red_flag) 
        # reward = reward_1
    
        
        return reward
