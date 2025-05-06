import math
from operator import mod
from turtle import distance
import numpy as np
import random
# from IPython.display import clear_output
from collections import deque
#import progressbar          
import gym
# import gymnasium as gym
import random
import matplotlib.pyplot as plt
## enviornment start
import matlab.engine
import csv
#eng = matlab.engine.start_matlab();

from numpy.core.fromnumeric import shape         

# from tensorflow.keras import Model
# from tensorflow.keras.layers import Dense, Embedding, Reshape
# from tensorflow.keras.optimizers import Adam

import os.path
import re
# import tensorflow as tf 
# tf.debugging.set_log_device_placement(True)
from configs_3 import args
from collections import deque


class Normalization:
    def __init__(self,eng, args=args):
        self.GTO_to_Lunar =  args.GTO_to_Lunar
        self.eng = eng
        if self.GTO_to_Lunar:
            self.initial_state = np.array(args.GTO_state)   # [h;hx;hy;ex;ey;mass;time]  superGTO2 Adrean paper   Isp=1500 m0=1000kg F= 1N
            self.target_state  = np.array(args.Lunar_state)     # [h;hx;hy;ex;ey;mass;time]   NRHO lunar Adrean paper   Isp=1500 m0=1000kg F= 1N
        else:
            self.initial_state = np.array(args.Lunar_state)   # [h;hx;hy;ex;ey;mass;time]  NRHO lunar Adrean paper   Isp=1500 m0=1000kg F= 1N
            self.target_state  = np.array(args.GTO_state)     # [h;hx;hy;ex;ey;mass;time]  superGTO2 Adrean paper   Isp=1500 m0=1000kg F= 1N  
        self.new_scaling_flag = args.new_scaling_flag
        self.scaler_type_ST_MM = args.scaler_type_ST_MM
        self.initial_state_for_minmax_scaling = np.array([400000, 50000 , -150000, -0.020972952,-0.124993226, 1000, 59812 ]) 
        self.Px = args.Px[0:-1]
        self.qx = args.qx[0:-1]
        self.Pu = args.Pu
        self.qu = args.qu
        from sklearn.preprocessing import StandardScaler ,  MinMaxScaler
        Pxx = np.diag(self.Px)
        state_scaled = np.linalg.inv(Pxx).dot(self.initial_state[0:-1] - self.qx)
        state_scaled_minmax = np.linalg.inv(Pxx).dot(self.initial_state_for_minmax_scaling[0:-1] - self.qx)
        state_scaled_1 = np.array(state_scaled).reshape(-1, 1)
        state_scaled_minmax_1 = np.array(state_scaled_minmax).reshape(-1, 1)
        if self.scaler_type_ST_MM == 1:
            self.scaler = StandardScaler()
            state_scaled_11 = state_scaled_1
        elif self.scaler_type_ST_MM ==2:
            self.scaler = MinMaxScaler()
            state_scaled_11 = state_scaled_minmax_1
            
        self.scaler.fit(state_scaled_11 )
        # self.minmax_scaler.fit(state_scaled_minmax_1)
        
        
        
    def Orignal_to_Scaled_st (self,orignal_state): 
        Px = np.diag(self.Px)
        state_scaled = np.linalg.inv(Px).dot(orignal_state - self.qx)
        return state_scaled  
    
    def Scaled_to_Normalized_st (self,state_scaled): 
        state_scaled_1 = np.array(state_scaled).reshape(-1, 1) 
        normalized_state = self.scaler.transform(state_scaled_1)
        # normalized_state_minmaxsc= self.minmax_scaler.transform(state_scaled_1)
        return normalized_state.reshape(1,-1)  
    
    def Normalized_to_Scaled_st (self,normalized_state): 
        normalized_state_1 = np.array(normalized_state).reshape(-1, 1) 
        state_scaled = self.scaler.inverse_transform(normalized_state_1)
        # state_scaled = self.minmax_scaler.inverse_transform(normalized_state_minmaxsc)
        return state_scaled.reshape(1,-1) 
    
    def Scaled_to_Orignal_st (self,Scaled_state): 
        Px = np.diag(self.Px)
        state_scaled_1 = np.array(Scaled_state).reshape(-1, 1) 
        orignal_state = np.dot(Px,state_scaled_1 ) +  np.array(self.qx).reshape(-1, 1) 
        return orignal_state.reshape(1,-1)   
      
    
    def Single_value_Orignal_to_Scaled_st (self,single_orignal_state): 
        orignal_state = np.array([1,1,1,single_orignal_state,1,1])
        Px = np.diag(self.Px)
        state_scaled = np.linalg.inv(Px).dot(orignal_state - self.qx)
        single_state_scaled = state_scaled[3]
        return single_state_scaled 
    

class Enviornment:
    def __init__(self, eng, args):
        self.GTO_to_Lunar =  args.GTO_to_Lunar
        self.eng = eng
        if self.GTO_to_Lunar:
            self.initial_state = np.array(args.GTO_state)   # [h;hx;hy;ex;ey;mass;time]  superGTO2 Adrean paper   Isp=1500 m0=1000kg F= 1N
            self.target_state  = np.array(args.Lunar_state)     # [h;hx;hy;ex;ey;mass;time]   NRHO lunar Adrean paper   Isp=1500 m0=1000kg F= 1N
        else:
            self.initial_state = np.array(args.Lunar_state)   # [h;hx;hy;ex;ey;mass;time]  NRHO lunar Adrean paper   Isp=1500 m0=1000kg F= 1N
            self.target_state  = np.array(args.GTO_state)     # [h;hx;hy;ex;ey;mass;time]  superGTO2 Adrean paper   Isp=1500 m0=1000kg F= 1N 
            
        self.milestone     = args.milestone
        self.e_a_i_ms_1    = np.array(args.e_a_i_ms_1)
        self.ecc_ms_1, self.a_ms_1, self.inc_ms_1    = self.e_a_i_ms_1[0], self.e_a_i_ms_1[1], self.e_a_i_ms_1[2]
        self.ms_reward     = args.ms_reward
        
        current_folder = os.getcwd()
        parent_folder = os.path.dirname(current_folder)
        output_dir = os.path.join(parent_folder, 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        save_path_1 = os.path.join(output_dir, 'csv')
        os.makedirs(save_path_1, exist_ok=True)
        
        self.completeName_csvlist = os.path.join(save_path_1, "csvlist_"+args.csv_file_nu+".dat") 
        self.completeName_csvlistinitialize = os.path.join(save_path_1, "csvlistinitialize_"+args.csv_file_nu+".dat")
        
        if not os.path.exists(self.completeName_csvlist):
            with open(self.completeName_csvlist, 'w'):
                pass
        if not os.path.exists(self.completeName_csvlistinitialize):
            with open(self.completeName_csvlistinitialize, 'w'):
                pass  # Creates an empty file

        self.segment_flag = 0
        self.flag_raise_error = 0
        self.I_sp = 1500      #1800; # in sec
        self.mu = 398600.4418 # global mu matlab
        self.norm = Normalization(eng, args)
        
        self.F1          = args.force
        self.m0         = self.initial_state[-2]
        self.tol_inc    = args.tol_inc
        self.tol_ecc    = args.tol_ecc
        self.tol_a      = args.tol_a
        if self.GTO_to_Lunar:
            self.phi_0      = args.phi0
            self.segment    = (args.seg) *(3.14/180)
            self.phi_1      = self.phi_0 + self.segment 
            self.a_min      = 34000      #args.a_min 
            self.a_max      = 430000      #args.a_max 
            self.inc_min    = 10      #args.inc_min 
            self.inc_max    = 52      #args.inc_max
            self.ecc_min    = 0.0067      #args.ecc_min 
            self.ecc_max    = 0.94     #args.ecc_max
            self.RAAN_target = 21.1856
            self.argp_target = 238.2852

        else:
            self.phi_0      = args.phi0
            self.segment    = (args.seg) *(3.14/180) 
            self.phi_1      = self.phi_0 - self.segment
            self.a_min      = 40000      #args.a_min 
            self.a_max      = 420000      #args.a_max 
            self.inc_min    = 10      #args.inc_min 
            self.inc_max    = 52      #args.inc_max
            self.ecc_min    = 0.0067      #args.ecc_min 
            self.ecc_max    = 0.94     #args.ecc_max
            self.RAAN_target = 11.3
            self.argp_target = 0

        self.shadow_flag      = args.sh_flag
        self.flag_a_boudary   = args.a_boundary_flag
        self.flag_ecc_boudary = args.ecc_boundary_flag
        self.flag_inc_boudary = args.inc_boundary_flag
        
        
        self.rf         =  args.rf
        self.var_seg    =  args.var_seg
        self.normalize_case = args.normalization_case                   # 1: normalization of state vector(h,hx,hy) with general value,  2: normalization of state vector(h,hx,hy) with seprate values
        self.intermediate_state_use  = args.intermediate_state_use      # 0: Target is intermediate state,    1: Target is final position'
        self.manual_weights = args.manual_weights                       # 0: selct reward weights through args , 1: selec manualy reward weights in code
        self.RL_algo = args.algo
        self.done_ep_reward = args.done_ep_reward
        self.pre_trained_weights = args.pre_tr_weight
        self.reward_normalize = args.r_norm
        self.test= args.test
        self.sac_paper_cond = args.sac_paper_cond

        self.reward_normalize = args.reward_normalize
        self.running_mean = 0.0
        self.running_var = 0.0
        self.running_count = 0
        self.epsilon = 1e-5  
        self.r_norm_mult = args.r_norm_mult
        self.max_steps_one_ep = args.max_steps_one_ep
        
        self.only_a_conv = args.a_conv
        self.a_conv = args.a_conv
        self.ecc_conv = args.ecc_conv
        self.inc_conv = args.inc_conv
        self.h_conv = args.h_conv
        self.hx_conv = args.hx_conv
        self.hy_conv = args.hy_conv
        self.ex_conv = args.ex_conv
        self.ey_conv = args.ey_conv
        self.RAAN_conv = args.RAAN_conv
        self.argp_conv = args.argp_conv
        self.simpl_or_exp_rew = args.simpl_or_exp_rew
        self.SCst_NORMst_ORGNst_Reward = args.SCst_NORMst_ORGNst_Reward
        self.raise_error_counter = 0
        
        self.he_para_flag = args.he_ele_conv
        
        
        
        self.RAANtol = args.tol_RAAN
        self.argptol = args.tol_argp
        self.RAAN_tol_low = self.RAAN_target - args.tol_RAAN
        self.RAAN_tol_high = self.RAAN_target + args.tol_RAAN
        self.argp_tol_low =  self.argp_target -  args.tol_argp
        self.argp_tol_high =  self.argp_target +  args.tol_argp 
        
        self.nu_eps_hist = 1
        self.prev_eps_orignal_states = deque(maxlen=self.nu_eps_hist)
        self.prev_eps_norm_states = deque(maxlen=self.nu_eps_hist)
        self.prev_eps_scaled_states = deque(maxlen=self.nu_eps_hist)
        self.prev_eps_timeindays = deque(maxlen=self.nu_eps_hist)
        self.prev_eps_RAAN= deque(maxlen=self.nu_eps_hist)
        self.prev_eps_argp= deque(maxlen=self.nu_eps_hist) 
        self.prev_eps_nurev= deque(maxlen=self.nu_eps_hist)
        self.prev_eps_phi= deque(maxlen=self.nu_eps_hist)
            
        self.conv_list = [f"flag_{param}" for param, conv in {
            "a": self.a_conv,
            "ecc": self.ecc_conv,
            "inc": self.inc_conv,
            "h": self.h_conv,
            "hx": self.hx_conv,
            "hy": self.hy_conv,
            "ex": self.ex_conv,
            "ey": self.ey_conv,
            "RAAN": self.RAAN_conv,
            "argp": self.argp_conv
        }.items() if conv]
        
    
        self.tol_h = args.tol_h
        self.tol_hx = args.tol_hx
        self.tol_ex = args.tol_ex
        self.tol_hy = args.tol_hy
        self.tol_ey = args.tol_ey
        self.w_h_s = args.w_h_s
        self.w_hx_s = args.w_hx_s
        self.w_hy_s = args.w_hy_s
        self.w_ex_s = args.w_ex_s
        self.w_ey_s = args.w_ey_s
        self.w_a_s = args.w_a_s
        self.w_ecc_s = args.w_ecc_s
        self.w_inc_s = args.w_inc_s
        self.reward_weight_upscale = args.reward_weight_upscale
        self.bounday_term_or_stuck = args.bounday_term_or_stuck
        
        self.discrete_reward = args.discrete_reward
        self.w_a_dr  = args.w_a_dr
        self.w_e_dr  = args.w_e_dr
        self.w_i_dr  = args.w_i_dr
        self.w_h_dr  = args.w_h_dr
        self.w_hx_dr  = args.w_hx_dr
        self.w_hy_dr  = args.w_hy_dr
        self.w_ex_dr  = args.w_ex_dr
        self.w_ey_dr  = args.w_ey_dr
        self.w_argp_dr  = args.w_argp_dr
        self.w_raan_dr  = args.w_raan_dr
         
        if self.intermediate_state_use == 0:
            self.inter_state  = np.array([0,0,0,0,0,0,0]) # [h;hx;hy;ex;ey;mass;time]  No intermediate state   Isp=1500 m0=1000kg F= 1N 
            self.target_inc = ((math.asin(math.sqrt((self.target_state[1]**2)+(self.target_state[2]**2))/self.target_state[0])) / np.pi)*180 
            self.target_ecc = math.sqrt((self.target_state[3]**2)+(self.target_state[4]**2))
            self.target_a   = ((self.target_state[0]**2)/self.mu)  /(1-(self.target_ecc**2))
        
        elif  self.intermediate_state_use == 1:  
            self.inter_state  = np.array(args.intermediate_state) # [h;hx;hy;ex;ey;mass;time]  intermediate state considered as target state Adrean paper   Isp=1500 m0=1000kg F= 1N 
            # calculating orbital paramters throgh intermediate states
            self.target_inc = ((math.asin(math.sqrt((self.inter_state[1]**2)+(self.inter_state[2]**2))/self.inter_state[0])) / np.pi)*180 
            self.target_ecc = math.sqrt((self.inter_state[3]**2)+(self.inter_state[4]**2))
            self.target_a   = ((self.inter_state[0]**2)/self.mu)  /(1-(self.target_ecc**2))
            

        self.tol_inc_low, self.tol_inc_high = self.target_inc - self.tol_inc, self.target_inc + self.tol_inc
        self.tol_ecc_low, self.tol_ecc_high = self.target_ecc - self.tol_ecc, self.target_ecc + self.tol_ecc
        self.tol_a_low, self.tol_a_high = self.target_a - self.tol_a, self.target_a + self.tol_a 
        
        
        self.target_state_scaled  = self.norm.Orignal_to_Scaled_st(self.target_state[0:-1]) 
        self.tol_ex_scaled = self.norm.Single_value_Orignal_to_Scaled_st(self.tol_ex)
        self.tol_ey_scaled = self.norm.Single_value_Orignal_to_Scaled_st(self.tol_ey)
        
        self.tol_ex_low_scaled, self.tol_ex_high_scaled = self.target_state_scaled[3]  - self.tol_ex_scaled, self.target_state_scaled[3]+ self.tol_ex_scaled
        self.tol_ey_low_scaled, self.tol_ey_high_scaled = self.target_state_scaled[4] - self.tol_ey_scaled , self.target_state_scaled[4]+ self.tol_ey_scaled 
        
        self.tol_ecc_low_scaled  =  self.norm.Single_value_Orignal_to_Scaled_st(self.tol_ecc_low)
        self.tol_ecc_high_scaled =  self.norm.Single_value_Orignal_to_Scaled_st(self.tol_ecc_high)
        self.ecc_max_scaled  =  self.norm.Single_value_Orignal_to_Scaled_st(self.ecc_max)
        self.ecc_min_scaled  =  self.norm.Single_value_Orignal_to_Scaled_st(self.ecc_min)
        
    
        self.ecc_ms_1_tol_low,self.ecc_ms_1_tol_high,  self.a_ms_1_tol_low,self.a_ms_1_tol_high,  self.inc_ms_1_tol_low,self.inc_ms_1_tol_high = 0,0,0,0,0,0
        if self.milestone:
            #  self.ecc_ms_1_tol_low,self.ecc_ms_1_tol_high,  self.a_ms_1_tol_low,self.a_ms_1_tol_high,  self.inc_ms_1_tol_low,self.inc_ms_1_tol_high = self.tol_ecc_low-self.ecc_ms_1, self.tol_ecc_high+self.ecc_ms_1,      self.tol_a_low-self.a_ms_1, self.tol_a_high+self.a_ms_1 ,          self.tol_inc_low-self.inc_ms_1,self.tol_inc_high+self.inc_ms_1
             self.ecc_ms_1_tol_low,self.ecc_ms_1_tol_high,  self.a_ms_1_tol_low,self.a_ms_1_tol_high,  self.inc_ms_1_tol_low,self.inc_ms_1_tol_high = self.target_ecc-self.ecc_ms_1, self.target_ecc+self.ecc_ms_1,      self.target_a -self.a_ms_1, self.target_a+self.a_ms_1 ,          self.target_inc-self.inc_ms_1,self.target_inc+self.inc_ms_1
        
        
        
        if self.intermediate_state_use == 1:
            self.tol_inc_low = 27             # target inc
            self.tol_ecc_high = 0.65          # target ecc
            self.tol_a_low    = 44364         # target a
        
        
        if self.he_para_flag: 
            w1 =    {  "h": args.w1_h,   "hx": args.w1_hx,    "hy": args.w1_hy,  "ex": args.w1_ex,   "ey": args.w1_ey,         "a": args.w1_a,       "e": args.w1_e,      "i": args.w1_i,        "RAAN": args.w1_RAAN,      "argp": args.w1_argp   }
            w1_ =   {  "h_": args.w1_h_, "hx_": args.w1_hx_,  "hy_": args.w1_hy_, "ex_": args.w1_ex_, "ey_": args.w1_ey_,       "a_": args.w1_a_,     "e_": args.w1_e_,    "i_": args.w1_i_,    "RAAN_": args.w1_RAAN_,    "argp_": args.w1_argp_  }
            c1 =    {  "h": args.c1_h,   "hx": args.c1_hx,    "hy": args.c1_hy,  "ex": args.c1_ex,   "ey": args.c1_ey,         "a": args.c1_a,       "e": args.c1_e,     "i": args.c1_i,        "RAAN": args.c1_RAAN,     "argp": args.c1_argp}
            self.done_ep_reward = args.done_ep_reward
            
            if self.sac_paper_cond:
                w1 =    {  "h": args.w1_h_sac,   "hx": args.w1_hx_sac,    "hy": args.w1_hy_sac,  "ex": args.w1_ex_sac,   "ey": args.w1_ey_sac,         "a": args.w1_a_sac,       "e": args.w1_e_sac,      "i": args.w1_i_sac ,    "RAAN": args.w1_RAAN_sac,      "argp": args.w1_argp_sac   }
                w1_ =   {  "h_": args.w1_h_sac_, "hx_": args.w1_hx_sac_,  "hy_": args.w1_hy_sac_, "ex_": args.w1_ex_sac_, "ey_": args.w1_ey_sac_,       "a_": args.w1_a_sac_,     "e_": args.w1_e_sac_,    "i_": args.w1_i_sac_,  "RAAN_": args.w1_RAAN_sac_,    "argp_": args.w1_argp_sac_   }
                c1 =    {  "h": args.c1_h_sac,   "hx": args.c1_hx_sac,    "hy": args.c1_hy_sac,  "ex": args.c1_ex_sac,   "ey": args.c1_ey_sac,         "a": args.c1_a_sac,       "e": args.c1_e_sac,     "i": args.c1_i_sac,      "RAAN": args.c1_RAAN_sac,     "argp": args.c1_argp_sac   }
                self.done_ep_reward = args.done_ep_reward_sac
                self.rf = args.rf_sac
            self.weights = {
                "w1": w1,
                "w1_": w1_,
                "c1": c1,
                "tau": args.tau
            }
            
        else:
            w1 =    {   "a": args.w1_a,       "e": args.w1_e,      "i": args.w1_i   }
            w1_ =   {   "a_": args.w1_a_,   "e_": args.w1_e_,    "i_": args.w1_i_   }
            c1 =    {   "a": args.c1_a,       "e": args.c1_e,     "i": args.c1_i,   }
            
            self.weights = {
                "w1": w1,
                "w1_": w1_,
                "c1": c1,
                "tau": args.tau
            }
        
        
    def get_param_values (self):  
        return self.initial_state,self.target_state,self.inter_state, self.target_a,self.target_ecc,self.target_inc,self.tol_a_low,self.tol_a_high,self.tol_ecc_low,self.tol_ecc_high,self.tol_inc_low,self.tol_inc_high,self.manual_weights,self.intermediate_state_use,self.RL_algo , self.done_ep_reward , self.pre_trained_weights, self.milestone, self.ecc_ms_1_tol_low,self.ecc_ms_1_tol_high,  self.a_ms_1_tol_low,self.a_ms_1_tol_high,  self.inc_ms_1_tol_low,self.inc_ms_1_tol_high, self.ms_reward  ,        self.he_para_flag, self.a_conv,self.ecc_conv,self.inc_conv  ,  self.h_conv,self.hx_conv,self.hy_conv,self.ex_conv,self.ey_conv,    self.simpl_or_exp_rew,        self.tol_h,  self.tol_hx ,  self.tol_hy ,  self.tol_ex ,  self.tol_ey ,  self.RAANtol, self.argptol                               
     
    def is_terminal(self,new_state_orignal, new_state_scaled,tar_state_scaled, RAAN, argp):       
        #for moon to earth: inc should increase, ecc should increase , a should decrease    
        h_orig, hx_orig, hy_orig, ex_orig, ey_orig, _ = new_state_orignal[0]     
        h, hx, hy, ex, ey, _ = new_state_scaled[0]
        h_t, hx_t, hy_t, ex_t, ey_t, _= tar_state_scaled[0]
        mu = self.mu 
        tol_inc, tol_ecc, tol_a = self.tol_inc, self.tol_ecc, self.tol_a
        # for COnvergence Flag
        try:
            # for Inclination 
            i = ((math.asin(math.sqrt((hx**2)+(hy**2))/h)) / np.pi)*180
            flag_inc = 1 if (self.tol_inc_low) < i and i < (self.tol_inc_high) else 0        
            # for eccentricity  
            ecc = math.sqrt((ex_orig**2)+(ey_orig**2))
            ecc_scaled = self.norm.Single_value_Orignal_to_Scaled_st(ecc)
            flag_ecc = 1 if (self.tol_ecc_low_scaled) < ecc_scaled and ecc_scaled < (self.tol_ecc_high_scaled) else 0
            # for Semimajor axis a
            p = (h*h)/mu
            a = p/(1-ecc**2)
            flag_a = 1 if (self.tol_a_low) < a and a < (self.tol_a_high) else 0   
            # Check Convergence
            flag = 1 if flag_inc and flag_ecc and flag_a else 0
            # for Final Boundary Red flag
            flag_for_error_inc = 1 if (i > self.inc_max) or (i < self.inc_min) else 0
            flag_for_error_ecc = 1 if (ecc_scaled > self.ecc_max_scaled) or (ecc_scaled < self.ecc_min_scaled) else 0
            flag_for_error_a = 1 if (a > self.a_max) or (a < self.a_min) else 0
            red_flag = 1 if (flag_for_error_inc*self.flag_inc_boudary ) or (flag_for_error_ecc*self.flag_ecc_boudary) or (flag_for_error_a*self.flag_a_boudary) else 0
            # for Monitor Flag
            monitor_inc_flag = 1 if (i > self.target_inc) else 0
            monitor_ecc_flag = 1 if (ecc > self.target_ecc) else 0
            monitor_a_flag = 1 if (a < self.target_a) else 0
            monitor_flag = 1 if monitor_inc_flag or monitor_ecc_flag or monitor_a_flag else 0  
            # only semimajor axis Convergence
            if self.only_a_conv:
                flag = 1 if flag_a else 0
                monitor_flag = 1 if  monitor_a_flag else 0 
            # checking for milestone
            flag_ms_1 = 0
            if self.milestone :
                flag_a_ms_1 = 1 if (self.target_a - self.a_ms_1) < a and a < (self.target_a + self.a_ms_1) else 0   
                flag_ecc_ms_1 = 1 if (self.target_ecc - self.ecc_ms_1) < ecc and ecc < (self.target_ecc + self.ecc_ms_1) else 0 
                flag_inc_ms_1 = 1 if (self.target_inc - self.inc_ms_1) < i and i < (self.target_inc + self.inc_ms_1) else 0 
                flag_ms_1 = 1 if flag_inc_ms_1 and flag_ecc_ms_1 and flag_a_ms_1 else 0
                if self.only_a_conv:
                    flag_ms_1 = 1 if flag_a_ms_1 else 0
            if self.he_para_flag:
                flag_a = 1 if (self.tol_a_low) < a and a < (self.tol_a_high) else 0 
                flag_ecc = 1 if (self.tol_ecc_low_scaled) < ecc_scaled and ecc_scaled < (self.tol_ecc_high_scaled) else 0
                flag_inc = 1 if (self.tol_inc_low) < i and i < (self.tol_inc_high) else 0 
                # flag_a = 1 if (44314.98) < a and a < (44315) else 0 
                # flag_ecc = 1 if (64961) < ecc_scaled and ecc_scaled < (64963) else 0
                # flag_inc = 1 if 26.9 < i and i < (27.1) else 0 
                flag_h = 1 if h<(h_t+self.tol_h) and h>(h_t-self.tol_h) else 0
                flag_hx = 1 if hx<(hx_t+self.tol_hx) and hx>(hx_t-self.tol_hx) else 0
                flag_hy = 1 if hy<(hy_t+self.tol_hy) and hy>(hy_t-self.tol_hy) else 0
                flag_ex = 1 if ex<(self.tol_ex_high_scaled) and ex>(self.tol_ex_low_scaled) else 0
                flag_ey = 1 if ey<(self.tol_ey_high_scaled) and ey>(self.tol_ey_low_scaled) else 0
                flag_RAAN = 1 if RAAN<(self.RAAN_tol_high) and RAAN>(self.RAAN_tol_low) else 0
                flag_argp = 1 if argp<(self.argp_tol_high) and argp>(self.argp_tol_low) else 0
                # Check Convergence
                #flag = 1 if flag_h and flag_hx and flag_hy and flag_ex and flag_ey else 0
                # Check Convergence using self.conv_list
                convergence_flags = []           
                for flag_name in self.conv_list:
                    variable = locals().get(flag_name)   
                    #print(flag_name, variable)  
                    convergence_flags.append(variable)
                flag = 1 if all(cc for cc in convergence_flags) else 0
                
                
        # catch any exception that occurs
        except Exception as e:  
            print(f"Error raised in terminal func: {e}")
            self.flag_raise_error = 1  
            flag,red_flag, monitor_flag, flag_for_error_ecc, flag_for_error_a, flag_for_error_inc = 0,0,0,0,0,0
            ecc,a,i = self.ecc_min,self.a_min,self.inc_min
             
        return flag,red_flag,monitor_flag, ecc,ecc_scaled,a,i,  flag_for_error_ecc,flag_for_error_a,flag_for_error_inc, self.flag_raise_error , flag_ms_1  
    
    
    def reset_csv(self):
        state     = self.initial_state                                
        self.temp = state                                           # M(1:7)  [h,hx,hy,ex,ey,mass,time]
        self.temp = np.append(self.temp, 0.5)                # M(8)
        self.temp = np.append(self.temp, 0.5)                 # M(9)
        self.temp = np.append(self.temp, self.F1)                    # M(10)
        self.temp = np.append(self.temp, self.phi_0)                # M(11)
        self.temp = np.append(self.temp, self.phi_1)                # M(12)               
        self.temp = np.append(self.temp, self.tol_inc)              # M(13)
        self.temp = np.append(self.temp, self.tol_ecc)              # M(14)
        self.temp = np.append(self.temp, self.tol_a)                # M(15)
        self.temp = np.append(self.temp, self.shadow_flag)          # M(16)
        self.temp = np.append(self.temp, self.I_sp)                 # M(17)
        self.temp = np.append(self.temp, self.m0)                   # M(18)
        self.temp = np.append(self.temp, self.target_state )           # M(19:25)
        with open(self.completeName_csvlistinitialize , 'w') as csvfile: 
		       csvwriter = csv.writer(csvfile)
		       csvwriter.writerow(self.temp)
		       csvfile.close()
		       #print(a)    
        return state , self.phi_0
           
    
    def step( self, state11,state1_ND, step_nu,time,  alpha, beta ,phi_0, force, target_scaled, Px, qx, Pu, qu ,start_time ):
       
        # self.state1_ND111   = self.norm.Normalized_to_Scaled_st(state11)
        prev_orignal_state   = self.norm.Scaled_to_Orignal_st(state1_ND)
        
        self.prev_eps_orignal_states.append(prev_orignal_state)
        self.prev_eps_norm_states.append(state11)
        self.prev_eps_scaled_states.append(state1_ND)
        self.prev_eps_phi.append(phi_0) 
        
        if self.var_seg:
            if self.GTO_to_Lunar:
                h_var,hx_var,hy_var,ex_var,ey_var = prev_orignal_state[0][0], prev_orignal_state[0][1], prev_orignal_state[0][2], prev_orignal_state[0][3], prev_orignal_state[0][4]
                ecc_var = math.sqrt((ex_var**2)+(ey_var**2))
                p_var = (h_var*h_var)/self.mu
                a_var = p_var/(1-ecc_var**2)
                i_var = ((math.asin(math.sqrt((hx_var**2)+(hy_var**2))/h_var)) / np.pi)*180
                flaag_inc = 1 if (self.inc_ms_1_tol_low) < i_var and i_var < (self.inc_ms_1_tol_high) else 0  
                flaag_ecc = 1 if (self.ecc_ms_1_tol_low) < ecc_var and ecc_var < (self.ecc_ms_1_tol_high) else 0
                flaag_a = 1 if (self.a_ms_1_tol_low) < a_var and a_var < (self.a_ms_1_tol_high) else 0   
                flaaag = 1 if flaag_inc and flaag_ecc and flaag_a else 0
                if flaaag:
                   segg =  args.var_small_seg*(3.14/180)
                else: 
                   segg = self.segment
                phi_1 = phi_0  + segg
            else:
                if step_nu <= 2000:
                    starting_seg = args.var_small_seg *(3.14/180)
                    segg = starting_seg
                else:
                    segg = self.segment
                phi_1 = phi_0  - segg
            seggg = segg
        
        else:
            if self.GTO_to_Lunar:
                phi_1 = phi_0  + self.segment
            else:
                phi_1 = phi_0  - self.segment
            seggg = self.segment    
         
        
        alpha = alpha * np.pi
        beta  = beta  * np.pi/2   
        
        force = self.F1
        obs = state1_ND[0].tolist()
        
        force = 0 if args.mass_conv and self.GTO_to_Lunar==0 and obs[-1] >= 1000 else self.F1

        # h_var,hx_var,hy_var,ex_var,ey_var = prev_orignal_state[0][0], prev_orignal_state[0][1], prev_orignal_state[0][2], prev_orignal_state[0][3], prev_orignal_state[0][4]
        # ecc_var = math.sqrt((ex_var**2)+(ey_var**2))
        # p_var = (h_var*h_var)/self.mu
        # a_var = p_var/(1-ecc_var**2)
        # if  self.target_a - a_var < 1:
        #     force = 0

        state = np.array(obs + [time]).reshape((1, 7))              # adding time in state
        tar_state1 = target_scaled.tolist()
        tar_state1 = np.array(tar_state1 + [0.00]).reshape((1, 7))              # adding time in state
        self.temp= state                                            # M(1:7)
        self.temp = np.append(self.temp, alpha)                     # M(8)
        self.temp = np.append(self.temp, beta)                      # M(9)
        self.temp = np.append(self.temp, force)                    # M(10)
        self.temp = np.append(self.temp, phi_0)                     # M(11)
        self.temp = np.append(self.temp, phi_1)                     # M(12)
        self.temp = np.append(self.temp, self.tol_inc)              # M(13)
        self.temp = np.append(self.temp, self.tol_ecc)              # M(14)
        self.temp = np.append(self.temp, self.tol_a)                # M(15)
        self.temp = np.append(self.temp, self.shadow_flag)          # M(16)
        self.temp = np.append(self.temp, self.I_sp)                 # M(17)
        self.temp = np.append(self.temp, self.m0)                   # M(18)
        self.temp = np.append(self.temp, tar_state1 )            # M(19:25)
        self.temp = np.append(self.temp, Px)          # M(26:31)
        self.temp = np.append(self.temp, qx)                 # M(32:37)
        self.temp = np.append(self.temp, Pu)                   # M(38:39)
        self.temp = np.append(self.temp, qu )            # M(40:41)
        self.temp = np.append(self.temp, start_time )            # M(42)
        with open(self.completeName_csvlist, 'w') as csvfile: 
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(self.temp)
            csvfile.close()
            #print
        self.flag_raise_error = 0
        try:
            a1 =  self.eng.Mat_env_3.resulting(self.completeName_csvlist)
            #print("136env: state", a1)
            a3 = list(a1._data)
            state_1=np.array(a3[:6])    # getting next state
            a_prev_s,e_prev_s,i_prev_s,RAAN_prev_s,argp_prev_s,nurev_prev_s=a3[14:22]    # getting next state
            a,e,i,RAAN,argp,nurev=a3[8:14]    # getting kaplerian elemenrts
            phi_1 = phi_1 % (2 * math.pi)
            # print('phi_1  ',phi_1,'  RAAN  ', RAAN,'  argp  ', argp)
            time_1 = a3[6]
            Force = 0 if args.mass_conv and self.GTO_to_Lunar==0 and state_1[-1] >= 1000 else a3[7]
            # Force = a3[7]
            self.prev_eps_RAAN.append(RAAN) 
            self.prev_eps_argp.append(argp) 
            self.prev_eps_nurev.append(nurev) 
            
            ## Scaled States
            new_state_scaled = np.array([state_1])
            prev_state_scaled = state1_ND
            tar_state_scaled  = np.array([target_scaled])
            
            ## Normalized States
            new_state_normalized   = self.norm.Scaled_to_Normalized_st(new_state_scaled)
            prev_state_normalized  = state11
            tar_state_normalized   = self.norm.Scaled_to_Normalized_st(tar_state_scaled)
            
            ## Orignal States
            new_state_orignal   = self.norm.Scaled_to_Orignal_st(new_state_scaled)
            prev_state_orignal  = prev_orignal_state
            tar_state_orignal   = self.norm.Scaled_to_Orignal_st(tar_state_scaled)
            
            tar_state = self.target_state[:6]      #[h,hx,hy,ex,ey,mass]
            done,redflag,monitor_flag,  ecc,ecc_scaled,a,i,  flag_ecc,flag_a,flag_inc, self.flag_raise_error , flag_ms_1=  self.is_terminal(new_state_orignal, new_state_scaled,tar_state_scaled, RAAN,argp)
            if step_nu > (self.max_steps_one_ep-1) :
                redflag = 1
            
            
            if self.GTO_to_Lunar:
                time_in_days =  - (self.initial_state[-1] - time_1) 
            else:
                time_in_days =  (self.initial_state[-1] - time_1)  
                
            # done1 = done
            # if done1 == 1:
            #     self.F1 = 0
            # done = 1 if (time_in_days > 0.5699) and (done1 == 1) else 0
            # if done == 1:
            #     self.F1 = 1
            
            
            if self.he_para_flag:
                reward, reward_plot =  Enviornment.Reward_he_elements(self, new_state_orignal,prev_state_orignal,tar_state_orignal,  prev_state_normalized,new_state_normalized,tar_state_normalized,  new_state_scaled,prev_state_scaled,tar_state_scaled,    done,redflag, flag_ms_1, RAAN,argp, RAAN_prev_s,argp_prev_s, step_nu)
            else:
                reward, ecc_new,i_new,a_new,  ecc_target,i_target,a_target , reward_plot = Enviornment.Reward(self, prev_state_normalized,new_state_normalized,  prev_state_orignal,new_state_orignal, tar_state_normalized,tar_state_orignal,  done,redflag,monitor_flag, flag_ms_1)  
            
            target_state_parameters =[ecc,a,i,  self.tol_ecc_low,self.tol_ecc_high,  self.tol_a_low,self.tol_a_high,  self.tol_inc_low,self.tol_inc_high,   self.target_ecc,self.target_inc,self.target_a,  flag_ecc,flag_a,flag_inc,  self.ecc_ms_1_tol_low,self.ecc_ms_1_tol_high,  self.a_ms_1_tol_low,self.a_ms_1_tol_high,  self.inc_ms_1_tol_low,self.inc_ms_1_tol_high ,   self.RAAN_tol_low , self.RAAN_tol_high , self.argp_tol_low  ,self.argp_tol_high]                                                                                                                                  
            scaled_ecc_param   =  [ecc_scaled, self.tol_ecc_low_scaled,self.tol_ecc_high_scaled ,  self.ecc_min_scaled,self.ecc_max_scaled,  self.ecc_max_scaled,   self.tol_ex_low_scaled,self.tol_ex_high_scaled,  self.tol_ey_low_scaled,self.tol_ey_high_scaled, self.tol_ex_scaled, self.target_state_scaled[3], self.target_state_scaled[4]]
        

            self.prev_eps_timeindays.append(time_in_days) 
            

           
            
        except Exception as e:
            print(f"Error raised in step func: {e}")
            self.flag_raise_error = 1
        
        if (self.flag_raise_error==1) or ((redflag*self.bounday_term_or_stuck)==1):
            new_state_orignal,new_state_normalized,new_state_scaled =self.prev_eps_orignal_states[-self.nu_eps_hist], self.prev_eps_norm_states[-self.nu_eps_hist], self.prev_eps_scaled_states[-self.nu_eps_hist]   
            reward, done, redflag = np.array([-self.rf ]), 0, 1
            
            tar_state        =  self.target_state[0:-1]
            tar_state_scaled =  self.norm.Orignal_to_Scaled_st(self.target_state[0:-1])
            phi_1 = self.prev_eps_phi[-self.nu_eps_hist]
            
            h_orig, hx_orig, hy_orig, ex_orig, ey_orig, _ = new_state_orignal[0]     
            h, hx, hy, ex, ey, _ = new_state_scaled[0]
            h_t, hx_t, hy_t, ex_t, ey_t, _= tar_state_scaled
            # for inclination
            i = ((math.asin(math.sqrt((hx**2)+(hy**2))/h)) / np.pi)*180      
            # for eccentricity  
            ecc = math.sqrt((ex_orig**2)+(ey_orig**2))
            ecc_scaled = self.norm.Single_value_Orignal_to_Scaled_st(ecc)
            # for Semimajor axis a
            p = (h*h)/self.mu
            a = p/(1-ecc**2)
            
            flag_inc = 1 if (i > self.inc_max) or (i < self.inc_min) else 0
            flag_ecc = 1 if (ecc_scaled > self.ecc_max_scaled) or (ecc_scaled < self.ecc_min_scaled) else 0
            flag_a = 1 if (a > self.a_max) or (a < self.a_min) else 0
            
            target_state_parameters = [ecc,a,i,  self.tol_ecc_low,self.tol_ecc_high,  self.tol_a_low,self.tol_a_high,  self.tol_inc_low,self.tol_inc_high,   self.target_ecc,self.target_inc,self.target_a, flag_ecc,flag_a,flag_inc,   self.ecc_ms_1_tol_low,self.ecc_ms_1_tol_high,  self.a_ms_1_tol_low,self.a_ms_1_tol_high,  self.inc_ms_1_tol_low,self.inc_ms_1_tol_high,   self.RAAN_tol_low , self.RAAN_tol_high , self.argp_tol_low  ,self.argp_tol_high] 
            scaled_ecc_param   =  [ecc_scaled, self.tol_ecc_low_scaled,self.tol_ecc_high_scaled ,  self.ecc_min_scaled,self.ecc_max_scaled,  self.ecc_max_scaled,   self.tol_ex_low_scaled,self.tol_ex_high_scaled,  self.tol_ey_low_scaled,self.tol_ey_high_scaled, self.tol_ex_scaled, self.target_state_scaled[3], self.target_state_scaled[4]]
          
            # time_in_days =  self.prev_eps_timeindays[-self.nu_eps_hist]
            time_in_days = 0
            time_1 =  self.initial_state[-1]  - time_in_days  
            reward_plot = [0,0,0,  0,0,0,  0,0,0,  0,0,0,  0,0,0]
            Force = self.F1
            # RAAN,argp, nurev =self.prev_eps_RAAN[-self.nu_eps_hist], self.prev_eps_argp[-self.nu_eps_hist], self.prev_eps_nurev[-self.nu_eps_hist]
            RAAN,argp, nurev =0,0,0
            self.raise_error_counter = self.raise_error_counter + 1
            
            if ((redflag*self.bounday_term_or_stuck)==1):
                redflag=0
                
            
        return new_state_orignal[0],new_state_normalized[0], new_state_scaled ,   reward,1==done, redflag, target_state_parameters, scaled_ecc_param,   seggg, phi_1,time_1, time_in_days, reward_plot , Force, RAAN,argp, nurev, self.raise_error_counter
    
    
    
    def DimtoNonDim_states (self, state):
        h, hx, hy, ex, ey,mass= state     
        #normalizing
        h_1 = h  / (390000)                         #  384073.94322
        hx_1 = hx / (390000)                #  43240.35671
        hy_1 = hy / (390000)               #  -111563.29463
        mass_1 = mass /  1000  
        #normalizing for neural networks
        h_2 = (h - 6000) / (390000 - 6000)                         #  384073.94322
        hx_2 = (hx - (-45000)) / (45000 - (-45000))                #  43240.35671
        hy_2 = (hy - (-150000)) / (1000 - (-150000))               #  -111563.29463
        if self.normalize_case == 1 :                    
            return [h_1, hx_1, hy_1, ex, ey, mass_1] , [h_2, hx_2, hy_2, ex, ey, mass_1]
        if self.normalize_case == 2 : 
            return [h_2, hx_2, hy_2, ex, ey, mass_1] , [h_1, hx_1, hy_1, ex, ey, mass_1] 
        
    def NonDimtoDim_states (self, state ):
        h, hx, hy, ex, ey,mass= state     
        #normalizing
        h_1 = h  * (390000)                #  384073.94322
        hx_1 = hx * (390000)               #  43240.35671
        hy_1 = hy * (390000)               #  -111563.29463
        #normalizing for neural networks
        h_2 = h * (390000 - 6000) + 6000 
        hx_2 = hx * (45000 - (-45000)) - 45000 
        hy_2 = hy * (1000 - (-150000)) - 150000
        mass_1 = mass *  1000 
        if self.normalize_case == 1 :                    
            return [h_1, hx_1, hy_1, ex, ey, mass_1] , [h_2, hx_2, hy_2, ex, ey, mass_1]
        if self.normalize_case == 2 : 
            return [h_2, hx_2, hy_2, ex, ey, mass_1] , [h_1, hx_1, hy_1, ex, ey, mass_1] 
    
    def normalize_value(self, value, minimum, maximum):
        if maximum == minimum:
            return 0.5  # Return 0.5 if max and min are the same, to avoid division by 0 error
        else:
            normalized_value = (value - minimum) / (maximum - minimum)
            return normalized_value
    
    
    def Reward_he_elements (self, newstate,prevstate,tarstate,    prev_state_ND,new_state_ND,tar_state_ND,  newstateSC,prevstateSC,tarstateSC,  done,redflag,flag_ms_1, RAAN,argp, RAAN_prev_s,argp_prev_s, step_nu):
        h,hx,hy,ex,ey = newstate[0][0],newstate[0][1],newstate[0][2],newstate[0][3],newstate[0][4]
        h_t,hx_t,hy_t,ex_t,ey_t = tarstate[0][0],tarstate[0][1],tarstate[0][2],tarstate[0][3],tarstate[0][4]
        w_h, w_hx, w_hy, w_ex, w_ey ,w_a, w_ecc, w_inc = self.w_h_s, self.w_hx_s, self.w_hy_s, self.w_ex_s, self.w_ey_s, self.w_a_s, self.w_ecc_s, self.w_inc_s
        if (self.simpl_or_exp_rew == 0) and (self.discrete_reward==0):    
            ecc = math.sqrt(ex**2 + ey**2)
            ecc_t = math.sqrt(ex_t **2 + ey_t**2)
            a = (h**2 /398600.44) / ( 1- (ecc**2))
            a_t = (h_t**2 /398600.44) / ( 1- (ecc_t**2))
            i = ((math.asin (math.sqrt((hx**2)+(hy**2))/h)) / np.pi)*180
            i_t = ((math.asin (math.sqrt((hx_t**2)+(hy_t**2))/h_t)) / np.pi)*180
            
            i, i_t = i/10, i_t/10
            
            h,hx,hy,ex,ey           = new_state_ND[0][0],new_state_ND[0][1],new_state_ND[0][2],new_state_ND[0][3],new_state_ND[0][4]
            h_t,hx_t,hy_t,ex_t,ey_t = tar_state_ND[0][0],tar_state_ND[0][1],tar_state_ND[0][2],tar_state_ND[0][3],tar_state_ND[0][4]
            diff_h, diff_hx ,diff_hy, diff_ex, diff_ey, diff_a, diff_ecc,diff_i = (np.abs(h-h_t))**2,  (np.abs(hx-hx_t))**2, (np.abs(hy-hy_t))**2, np.abs(ex-ex_t)**2, np.abs(ey-ey_t)**2,  (np.abs(a-a_t)/a)**2, np.abs(ecc-ecc_t)**2, np.abs(i-i_t)**2
            # diff_h, diff_hx ,diff_hy, diff_ex, diff_ey, diff_a, diff_ecc,diff_i = (np.abs(h-h_t)/h_t)**2,  (np.abs(hx-hx_t)/h_t)**2, (np.abs(hy-hy_t)/h_t)**2, np.abs(ex-ex_t)**2, np.abs(ey-ey_t)**2,  (np.abs(a-a_t)/a)**2, np.abs(ecc-ecc_t)**2, np.abs(i-i_t)**2
            up = self.reward_weight_upscale
            diff_h_w, diff_hx_w ,diff_hy_w, diff_ex_w, diff_ey_w , diff_a_w, diff_ecc_w,diff_i_w = w_h*diff_h, w_hx*diff_hx, w_hy*diff_hy, w_ex*diff_ex, w_ey*diff_ey,  w_a*diff_a, w_ecc*diff_ecc, w_inc*diff_i  
            diff_h_w_up, diff_hx_w_up ,diff_hy_w_up, diff_ex_w_up, diff_ey_w_up , diff_a_w_up, diff_ecc_w_up ,diff_i_w_up = (up)*diff_h_w, up*diff_hx_w, up*diff_hy_w, up*diff_ex_w,  up*diff_ey_w , up*diff_a_w, up*diff_ecc_w,  up*diff_i_w
            error = (diff_h_w_up*self.h_conv) + (diff_hx_w_up*self.hx_conv) + (diff_hy_w_up*self.hy_conv) + (diff_ex_w_up*self.ex_conv) + (diff_ey_w_up*self.ey_conv)  +   (diff_a_w_up*self.a_conv) + (diff_ecc_w_up*self.ecc_conv) + (diff_i_w_up*self.inc_conv)
            
            tauu_aei= self.weights["tau"]
            reward = -error - tauu_aei - (self.rf*redflag)  + (self.ms_reward*flag_ms_1) + (self.done_ep_reward * done) 
            reward_plot = [diff_h, diff_hx ,diff_hy, diff_ex, diff_ey,    diff_h_w, diff_hx_w ,diff_hy_w, diff_ex_w, diff_ey_w ,     diff_h_w_up, diff_hx_w_up ,diff_hy_w_up, diff_ex_w_up, diff_ey_w_up  ]
        
        
        if (self.simpl_or_exp_rew == 1):
            if  self.manual_weights == 0 :
                w1_aei  = [[self.weights["w1"]["a"]] , [self.weights["w1"]["e"]], [self.weights["w1"]["i"]], [self.weights["w1"]["h"]],[self.weights["w1"]["hx"]],[self.weights["w1"]["hy"]],[self.weights["w1"]["ex"]],[self.weights["w1"]["ey"]],[self.weights["w1"]["RAAN"]],[self.weights["w1"]["argp"]]  ]                          
                w1_aei_ = [[self.weights["w1_"]["a_"]] , [self.weights["w1_"]["e_"]], [self.weights["w1_"]["i_"]], [self.weights["w1_"]["h_"]],[self.weights["w1_"]["hx_"]],[self.weights["w1_"]["hy_"]],[self.weights["w1_"]["ex_"]],[self.weights["w1_"]["ey_"]],[self.weights["w1_"]["RAAN_"]],[self.weights["w1_"]["argp_"]]  ]       # test weighrts for normalized inputs
                c1_aei  = [self.weights["c1"]["a"] , self.weights["c1"]["e"], self.weights["c1"]["i"], self.weights["c1"]["h"],self.weights["c1"]["hx"],self.weights["c1"]["hy"],self.weights["c1"]["ex"],self.weights["c1"]["ey"],self.weights["c1"]["RAAN"],self.weights["c1"]["argp"]]   
                tauu_aei= self.weights["tau"]
            
            if self.SCst_NORMst_ORGNst_Reward == 1:
                #SCALED
                prev_states = prevstateSC[0]                 
                new_state   = newstateSC[0]
                tar_state   = tarstateSC[0]
            elif self.SCst_NORMst_ORGNst_Reward == 2:
                #NORMLAIZED
                prev_states = prev_state_ND[0]
                new_state   = new_state_ND[0]
                tar_state   = tar_state_ND[0]
    
            elif self.SCst_NORMst_ORGNst_Reward == 3:
                #Orignal Values
                prev_states = prevstate[0]
                new_state   = newstate[0]
                tar_state   = tarstate[0]
            
            ecc_prev = math.sqrt((prevstate[0][3])**2 + (prevstate[0][4])**2)
            ecc_new = math.sqrt((newstate[0][3])**2 + (newstate[0][4])**2)
            ecc_target = math.sqrt((tarstate[0][3])**2 + (tarstate[0][4])**2)
            
            a_prev = (((prevstate[0][0])**2) /self.mu ) / ( 1- (ecc_prev **2))  
            a_new = (((newstate[0][0])**2) /self.mu ) / ( 1- (ecc_new **2))
            a_target = (((tarstate[0][0])**2) /self.mu ) / ( 1- (ecc_target **2))
            
            i_prev = ((math.asin (math.sqrt((prevstate[0][1]**2)+(prevstate[0][2]**2))/prevstate[0][0])) / np.pi)*180
            i_new = ((math.asin (math.sqrt((newstate[0][1]**2)+(newstate[0][2]**2))/newstate[0][0])) / np.pi)*180
            i_target = ((math.asin (math.sqrt((tarstate[0][1]**2)+(tarstate[0][2]**2))/tarstate[0][0])) / np.pi)*180
        
            i_prev = i_prev /10
            i_new = i_new /10
            i_target = i_target /10  
            
            if self.SCst_NORMst_ORGNst_Reward == 2:
                a_prev = a_prev/ self.a_max
                a_new = a_new/ self.a_max
                a_target = a_target/ self.a_max
            
            st_a_e_i_prev = [[a_prev], [ecc_prev], [i_prev], [prev_states[0]], [prev_states[1]], [prev_states[2]], [prev_states[3]], [prev_states[4]], [RAAN_prev_s/360], [argp_prev_s/360]   ] 
            st_a_e_i_new = [[a_new], [ecc_new], [i_new],[new_state[0]], [new_state[1]], [new_state[2]], [new_state[3]], [new_state[4]], [RAAN/360], [argp/360]]
            st_a_e_i_target = [[a_target], [ecc_target], [i_target], [tar_state[0]], [tar_state[1]], [tar_state[2]], [tar_state[3]], [tar_state[4]], [self.RAAN_target/360], [self.argp_target/360]]
            
            if self.discrete_reward == 0:   
                exp_value_t_aei      =  0 
                exp_value_t_plus_1_aei =  0 

                phi_st_aei = (-w1_aei[0][0]*abs(st_a_e_i_prev[0][0] -st_a_e_i_target[0][0]) + (c1_aei[0]*math.exp(-(w1_aei_[0][0]*abs(st_a_e_i_prev[0][0] - st_a_e_i_target[0][0]))))) * self.a_conv   + \
                                (-w1_aei[1][0]*abs(st_a_e_i_prev[1][0] -st_a_e_i_target[1][0]) + (c1_aei[1]*math.exp(-(w1_aei_[1][0]*abs(st_a_e_i_prev[1][0] - st_a_e_i_target[1][0]))))) * self.ecc_conv + \
                                (-w1_aei[2][0]*abs(st_a_e_i_prev[2][0] -st_a_e_i_target[2][0]) + (c1_aei[2]*math.exp(-(w1_aei_[2][0]*abs(st_a_e_i_prev[2][0] - st_a_e_i_target[2][0]))))) * self.inc_conv + \
                                (-w1_aei[3][0]*abs(st_a_e_i_prev[3][0] -st_a_e_i_target[3][0]) + (c1_aei[3]*math.exp(-(w1_aei_[3][0]*abs(st_a_e_i_prev[3][0] - st_a_e_i_target[3][0]))))) * self.h_conv   + \
                                (-w1_aei[4][0]*abs(st_a_e_i_prev[4][0] -st_a_e_i_target[4][0]) + (c1_aei[4]*math.exp(-(w1_aei_[4][0]*abs(st_a_e_i_prev[4][0] - st_a_e_i_target[4][0]))))) * self.hx_conv  + \
                                (-w1_aei[5][0]*abs(st_a_e_i_prev[5][0] -st_a_e_i_target[5][0]) + (c1_aei[5]*math.exp(-(w1_aei_[5][0]*abs(st_a_e_i_prev[5][0] - st_a_e_i_target[5][0]))))) * self.hy_conv  + \
                                (-w1_aei[6][0]*abs(st_a_e_i_prev[6][0] -st_a_e_i_target[6][0]) + (c1_aei[6]*math.exp(-(w1_aei_[6][0]*abs(st_a_e_i_prev[6][0] - st_a_e_i_target[6][0]))))) * self.ex_conv  + \
                                (-w1_aei[7][0]*abs(st_a_e_i_prev[7][0] -st_a_e_i_target[7][0]) + (c1_aei[7]*math.exp(-(w1_aei_[7][0]*abs(st_a_e_i_prev[7][0] - st_a_e_i_target[7][0]))))) * self.ey_conv  + \
                                (-w1_aei[8][0]*abs(st_a_e_i_prev[8][0] -st_a_e_i_target[8][0]) + (c1_aei[8]*math.exp(-(w1_aei_[8][0]*abs(st_a_e_i_prev[8][0] - st_a_e_i_target[8][0]))))) * self.RAAN_conv  + \
                                (-w1_aei[9][0]*abs(st_a_e_i_prev[9][0] -st_a_e_i_target[9][0]) + (c1_aei[9]*math.exp(-(w1_aei_[9][0]*abs(st_a_e_i_prev[9][0] - st_a_e_i_target[9][0]))))) * self.argp_conv  
                                             
                phi_st_plus_1_aei = (-w1_aei[0][0]*abs(st_a_e_i_new[0][0] -st_a_e_i_target[0][0]) + (c1_aei[0]*math.exp(-(w1_aei_[0][0]*abs(st_a_e_i_new[0][0] - st_a_e_i_target[0][0]))))) * self.a_conv   + \
                                (-w1_aei[1][0]*abs(st_a_e_i_new[1][0] -st_a_e_i_target[1][0]) + (c1_aei[1]*math.exp(-(w1_aei_[1][0]*abs(st_a_e_i_new[1][0] - st_a_e_i_target[1][0]))))) * self.ecc_conv + \
                                (-w1_aei[2][0]*abs(st_a_e_i_new[2][0] -st_a_e_i_target[2][0]) + (c1_aei[2]*math.exp(-(w1_aei_[2][0]*abs(st_a_e_i_new[2][0] - st_a_e_i_target[2][0]))))) * self.inc_conv + \
                                (-w1_aei[3][0]*abs(st_a_e_i_new[3][0] -st_a_e_i_target[3][0]) + (c1_aei[3]*math.exp(-(w1_aei_[3][0]*abs(st_a_e_i_new[3][0] - st_a_e_i_target[3][0]))))) * self.h_conv   + \
                                (-w1_aei[4][0]*abs(st_a_e_i_new[4][0] -st_a_e_i_target[4][0]) + (c1_aei[4]*math.exp(-(w1_aei_[4][0]*abs(st_a_e_i_new[4][0] - st_a_e_i_target[4][0]))))) * self.hx_conv  + \
                                (-w1_aei[5][0]*abs(st_a_e_i_new[5][0] -st_a_e_i_target[5][0]) + (c1_aei[5]*math.exp(-(w1_aei_[5][0]*abs(st_a_e_i_new[5][0] - st_a_e_i_target[5][0]))))) * self.hy_conv  + \
                                (-w1_aei[6][0]*abs(st_a_e_i_new[6][0] -st_a_e_i_target[6][0]) + (c1_aei[6]*math.exp(-(w1_aei_[6][0]*abs(st_a_e_i_new[6][0] - st_a_e_i_target[6][0]))))) * self.ex_conv  + \
                                (-w1_aei[7][0]*abs(st_a_e_i_new[7][0] -st_a_e_i_target[7][0]) + (c1_aei[7]*math.exp(-(w1_aei_[7][0]*abs(st_a_e_i_new[7][0] - st_a_e_i_target[7][0]))))) * self.ey_conv  + \
                                (-w1_aei[8][0]*abs(st_a_e_i_new[8][0] -st_a_e_i_target[8][0]) + (c1_aei[6]*math.exp(-(w1_aei_[8][0]*abs(st_a_e_i_new[8][0] - st_a_e_i_target[8][0]))))) * self.RAAN_conv  + \
                                (-w1_aei[9][0]*abs(st_a_e_i_new[9][0] -st_a_e_i_target[9][0]) + (c1_aei[7]*math.exp(-(w1_aei_[9][0]*abs(st_a_e_i_new[9][0] - st_a_e_i_target[9][0]))))) * self.argp_conv 
                                
                # phi_st_aei = (-w1_aei[0][0]*abs(st_a_e_i_prev[0][0] -st_a_e_i_target[0][0]) + (c1_aei[0]*math.exp(-(w1_aei_[0][0]*abs(st_a_e_i_prev[0][0] - st_a_e_i_target[0][0]))))) * self.a_conv   + \
                #                 (-w1_aei[1][0]*abs(st_a_e_i_prev[1][0] -st_a_e_i_target[1][0]) + (c1_aei[1]*math.exp(-(w1_aei_[1][0]*abs(st_a_e_i_prev[1][0] - st_a_e_i_target[1][0]))))) * self.ecc_conv + \
                #                 (-w1_aei[2][0]*abs(st_a_e_i_prev[2][0] -st_a_e_i_target[2][0]) + (c1_aei[2]*math.exp(-(w1_aei_[2][0]*abs(st_a_e_i_prev[2][0] - st_a_e_i_target[2][0]))))) * self.inc_conv + \
                #                 (-700*abs(st_a_e_i_prev[3][0] -st_a_e_i_target[3][0]) + (0*        math.exp(-(0.5*abs(st_a_e_i_prev[3][0] - st_a_e_i_target[3][0]))) ) ) * 1   + \
                #                 (-700*abs(st_a_e_i_prev[4][0] -st_a_e_i_target[4][0]) + (0*        math.exp(-(0.5*abs(st_a_e_i_prev[4][0] - st_a_e_i_target[4][0]))) ) ) * 1  + \
                #                 (-700*abs(st_a_e_i_prev[5][0] -st_a_e_i_target[5][0]) + (0*        math.exp(-(0.5*abs(st_a_e_i_prev[5][0] - st_a_e_i_target[5][0]))) ) ) * 1  + \
                #                 (-300*abs(st_a_e_i_prev[6][0] -st_a_e_i_target[6][0]) + (0*        math.exp(-(0.00005*abs(st_a_e_i_prev[6][0] - st_a_e_i_target[6][0])))) ) * 1 + \
                #                 (-300*abs(st_a_e_i_prev[7][0] -st_a_e_i_target[7][0]) + (0*        math.exp(-(0.00005*abs(st_a_e_i_prev[7][0] - st_a_e_i_target[7][0])))) ) * 1  + \
                #                 (-700*abs(st_a_e_i_new[8][0] -st_a_e_i_target[8][0]) +  (0*        math.exp(-(w1_aei_[8][0]*abs(st_a_e_i_new[8][0] - st_a_e_i_target[8][0]))))) * self.RAAN_conv  + \
                #                 (-700*abs(st_a_e_i_new[9][0] -st_a_e_i_target[9][0]) +  (0*        math.exp(-(w1_aei_[9][0]*abs(st_a_e_i_new[9][0] - st_a_e_i_target[9][0]))))) * self.argp_conv  
                                
                                
                # phi_st_plus_1_aei = (-w1_aei[0][0]*abs(st_a_e_i_new[0][0] -st_a_e_i_target[0][0]) + (c1_aei[0]*math.exp(-(w1_aei_[0][0]*abs(st_a_e_i_new[0][0] - st_a_e_i_target[0][0]))))) * self.a_conv   + \
                #                 (-w1_aei[1][0]*abs(st_a_e_i_new[1][0] -st_a_e_i_target[1][0]) + (c1_aei[1]*math.exp(-(w1_aei_[1][0]*abs(st_a_e_i_new[1][0] - st_a_e_i_target[1][0]))))) * self.ecc_conv + \
                #                 (-w1_aei[2][0]*abs(st_a_e_i_new[2][0] -st_a_e_i_target[2][0]) + (c1_aei[2]*math.exp(-(w1_aei_[2][0]*abs(st_a_e_i_new[2][0] - st_a_e_i_target[2][0]))))) * self.inc_conv + \
                #                 (-700*abs(st_a_e_i_new[3][0] -st_a_e_i_target[3][0]) + (0*       math.exp(-(0.00005*abs(st_a_e_i_new[3][0] - st_a_e_i_target[3][0]))))) * 1   + \
                #                 (-700*abs(st_a_e_i_new[4][0] -st_a_e_i_target[4][0]) + (0*       math.exp(-(0.00005*abs(st_a_e_i_new[4][0] - st_a_e_i_target[4][0]))))) * 1  + \
                #                 (-700*abs(st_a_e_i_new[5][0] -st_a_e_i_target[5][0]) + (0*       math.exp(-(0.00005*abs(st_a_e_i_new[5][0] - st_a_e_i_target[5][0]))))) * 1 + \
                #                 (-300*abs(st_a_e_i_new[6][0] -st_a_e_i_target[6][0]) + (0*       math.exp(-(0.00005*abs(st_a_e_i_new[6][0] - st_a_e_i_target[6][0]))))) * 1 + \
                #                 (-300*abs(st_a_e_i_new[7][0] -st_a_e_i_target[7][0]) + (0*       math.exp(-(0.00005*abs(st_a_e_i_new[7][0] - st_a_e_i_target[7][0]))))) * 1   + \
                #                 (-700*abs(st_a_e_i_new[8][0] -st_a_e_i_target[8][0]) + (0*       math.exp(-(w1_aei_[8][0]*abs(st_a_e_i_new[8][0] - st_a_e_i_target[8][0]))))) * self.RAAN_conv  + \
                #                 (-700*abs(st_a_e_i_new[9][0] -st_a_e_i_target[9][0]) + (0*       math.exp(-(w1_aei_[9][0]*abs(st_a_e_i_new[9][0] - st_a_e_i_target[9][0]))))) * self.argp_conv    
                            
                reward_t_aei_1 = phi_st_plus_1_aei  - phi_st_aei - tauu_aei + (self.done_ep_reward * done) - (self.rf *redflag) 
                reward = reward_t_aei_1 + (flag_ms_1*self.ms_reward)                 
                                
            diff_h, diff_hx ,diff_hy, diff_ex, diff_ey, diff_a, diff_ecc,diff_i = (np.abs(h-h_t)/h_t)**2,  (np.abs(hx-hx_t)/h_t)**2, (np.abs(hy-hy_t)/h_t)**2, np.abs(ex-ex_t)**2, np.abs(ey-ey_t)**2,  (np.abs(a_new-a_target)/a_target)**2, np.abs(ecc_new-ecc_target)**2, np.abs(i_new-i_target)**2
            reward_plot = [diff_h, diff_hx ,diff_hy, diff_ex, diff_ey,   diff_h, diff_hx ,diff_hy, diff_ex, diff_ey,    diff_h, diff_hx ,diff_hy, diff_ex, diff_ey  ]
            
            
            if self.discrete_reward:
                            a_reward    = self.w_a_dr if ( abs(st_a_e_i_prev[0][0] -st_a_e_i_target[0][0])  - abs(st_a_e_i_new[0][0] -st_a_e_i_target[0][0])  )  >= 0  else - self.w_a_dr
                            e_reward    = self.w_e_dr  if ( abs(st_a_e_i_prev[1][0] -st_a_e_i_target[1][0])  - abs(st_a_e_i_new[1][0] -st_a_e_i_target[1][0])  )  >= 0  else - self.w_e_dr           
                            i_reward    = self.w_i_dr  if ( abs(st_a_e_i_prev[2][0] -st_a_e_i_target[2][0])  - abs(st_a_e_i_new[2][0] -st_a_e_i_target[2][0])  )  >= 0  else - self.w_i_dr
                            h_reward    = self.w_h_dr  if ( abs(st_a_e_i_prev[3][0] -st_a_e_i_target[3][0])  - abs(st_a_e_i_new[3][0] -st_a_e_i_target[3][0])  )  >= 0  else - self.w_h_dr
                            hx_reward   = self.w_hx_dr  if ( abs(st_a_e_i_prev[4][0] -st_a_e_i_target[4][0])  - abs(st_a_e_i_new[4][0] -st_a_e_i_target[4][0])  )  >= 0  else - self.w_hx_dr
                            hy_reward   = self.w_hy_dr  if ( abs(st_a_e_i_prev[5][0] -st_a_e_i_target[5][0])  - abs(st_a_e_i_new[5][0] -st_a_e_i_target[5][0])  )  >= 0  else - self.w_hy_dr
                            ex_reward   = self.w_ex_dr  if ( abs(st_a_e_i_prev[6][0] -st_a_e_i_target[6][0])  - abs(st_a_e_i_new[6][0] -st_a_e_i_target[6][0])  )  >= 0  else - self.w_ex_dr
                            ey_reward   = self.w_ey_dr  if ( abs(st_a_e_i_prev[7][0] -st_a_e_i_target[7][0])  - abs(st_a_e_i_new[7][0] -st_a_e_i_target[7][0])  )  >= 0  else - self.w_ey_dr
                            raan_reward = self.w_raan_dr  if ( abs(st_a_e_i_prev[8][0] -st_a_e_i_target[8][0])  - abs(st_a_e_i_new[8][0] -st_a_e_i_target[8][0])  )  >= 0  else - self.w_raan_dr   
                            argp_reward = self.w_argp_dr  if ( abs(st_a_e_i_prev[9][0] -st_a_e_i_target[9][0])  - abs(st_a_e_i_new[9][0] -st_a_e_i_target[9][0])  )  >= 0  else - self.w_argp_dr  
                            
                            reward = (a_reward * self.a_conv )    + (e_reward * self.ecc_conv )   + (i_reward   * self.inc_conv )  + \
                                     (h_reward * self.h_conv )    + (hx_reward * self.hx_conv )  + (hy_reward  * self.hy_conv )  + (ex_reward  * self.ex_conv ) + (ey_reward  * self.ey_conv ) + \
                                     (raan_reward * self.RAAN_conv ) + (argp_reward * self.argp_conv ) + \
                                     (self.done_ep_reward * done) - (self.rf *redflag)   + (flag_ms_1*self.ms_reward)   - tauu_aei
                            
            if self.reward_normalize:
                if done:
                    reward = reward - self.done_ep_reward
                if redflag:
                    reward = reward + self.rf
                ####### CAlcultaing Running mean and std #############    
                self.running_count += 1
                delta = reward - self.running_mean
                self.running_mean += delta / self.running_count     #  Running Mean (μ) at step N = Running Mean at step N-1 + (Reward(N) - Running Mean at step N-1) / N
                delta2 = reward - self.running_mean
                self.running_var += delta * delta2                  # Running Variance (σ²) at step N = Running Variance at step N-1 + (Reward(N) - Running Mean at step N-1) * (Reward(N) - Running Mean at step N)
                #################################################### 
                if self.running_count > 1:
                    std = (self.running_var / (self.running_count - 1)) ** 0.5       
                    reward = (reward - self.running_mean) / (std + self.epsilon)
                    reward = reward * self.r_norm_mult
                else:
                    reward
                
                if (step_nu > (self.max_steps_one_ep)) or (redflag==1)  or (done==1):
                    self.running_count = 0
                    self.running_mean  = 0
                    self.running_var   = 0

                if done:
                    reward = reward + self.done_ep_reward

                if redflag:
                    reward = reward - self.rf


        
        return np.array([reward]), reward_plot
    
    
    def Reward (self, prev_state_ND,new_state_ND,  prev_state,new_state,   tar_state_ND,tar_state,  done,redflag,monitor_flag, flag_ms_1):
       
       if  self.manual_weights == 1 :
            w1_aei  = [[1000] , [2010], [300] ]                            # GTO    
            w1_aei_ = [[0.010] , [0.000000198], [0.0003] ]      
            c1_aei  = [500 , 700, 300]
            tauu_aei= 0.003

            w1_aei  = [[1000] , [4010], [300] ]                            # superGTO    best
            w1_aei_ = [[0.010] , [0.000000000198], [0.0003] ]      
            c1_aei  = [500 , 2000, 300]
            tauu_aei= 0.003
            
            w1_aei  = [[1000*2] , [2010], [300*5] ]                            # GTO    
            w1_aei_ = [[0.010] , [0.000000198], [0.0003] ]      
            c1_aei  = [500*2 , 700, 300*5]
            tauu_aei= 0.003
       
       if  self.manual_weights == 0 :
            w1_aei  = [[self.weights["w1"]["a"]] , [self.weights["w1"]["e"]], [self.weights["w1"]["i"]]]                            
            w1_aei_ = [[self.weights["w1_"]["a_"]] , [self.weights["w1_"]["e_"]], [self.weights["w1_"]["i_"]] ]       # test weighrts for normalized inputs
            c1_aei  = [self.weights["c1"]["a"] , self.weights["c1"]["e"], self.weights["c1"]["i"]]   
            tauu_aei= self.weights["tau"]
   
    #################################################################################################           
       if self.reward_normalize:       
            ecc_prev = math.sqrt((prev_state[3])**2 + (prev_state[4])**2)
            ecc_new = math.sqrt((new_state[3])**2 + (new_state[4])**2)
            ecc_target = math.sqrt((tar_state[3])**2 + (tar_state[4])**2)
            
            a_prev = (((prev_state[0])**2) /self.mu) / ( 1- (ecc_prev**2))
            a_new = (((new_state[0])**2) /self.mu) / ( 1- (ecc_new**2))
            a_target = (((tar_state[0])**2) /self.mu) / ( 1- (ecc_target**2))

            i_prev = ((math.asin (math.sqrt((prev_state[1]**2)+(prev_state[2]**2))/prev_state[0])) / np.pi)*180      
            i_new = ((math.asin (math.sqrt((new_state[1]**2)+(new_state[2]**2))/new_state[0])) / np.pi)*180          
            i_target = ((math.asin (math.sqrt((tar_state[1]**2)+(tar_state[2]**2))/tar_state[0])) / np.pi)*180 
            
            # For normalizing between max and min range values
            ecc_prev = self.normalize_value(ecc_prev, self.ecc_min, self.ecc_max)
            ecc_new = self.normalize_value(ecc_new, self.ecc_min, self.ecc_max)
            ecc_target = self.normalize_value(ecc_target, self.ecc_min, self.ecc_max)
            a_prev = self.normalize_value(a_prev, self.a_min, self.a_max)
            a_new = self.normalize_value(a_new, self.a_min, self.a_max)
            a_target = self.normalize_value(a_target, self.a_min, self.a_max)
            i_prev = self.normalize_value(i_prev, self.inc_min, self.inc_max) 
            i_new = self.normalize_value(i_new, self.inc_min, self.inc_max)
            i_target = self.normalize_value(i_target, self.inc_min, self.inc_max)      
            #################################################################################################
       else:
            ecc_prev = math.sqrt((prev_state_ND[3])**2 + (prev_state_ND[4])**2)
            ecc_new = math.sqrt((new_state_ND[3])**2 + (new_state_ND[4])**2)
            ecc_target = math.sqrt((tar_state_ND[3])**2 + (tar_state_ND[4])**2)
            
            a_prev = (((prev_state_ND[0])**2) /1) / ( 1- (ecc_prev **2))
            a_new = (((new_state_ND[0])**2) /1) / ( 1- (ecc_new **2))
            a_target = (((tar_state_ND[0])**2) /1) / ( 1- (ecc_target **2))
            
            i_prev = ((math.asin (math.sqrt((prev_state_ND[1]**2)+(prev_state_ND[2]**2))/prev_state_ND[0])) / np.pi)*180
            i_new = ((math.asin (math.sqrt((new_state_ND[1]**2)+(new_state_ND[2]**2))/new_state_ND[0])) / np.pi)*180
            i_target = ((math.asin (math.sqrt((tar_state_ND[1]**2)+(tar_state_ND[2]**2))/tar_state_ND[0])) / np.pi)*180
       
       
            i_prev = i_prev /10
            i_new = i_new /10
            i_target = i_target /10    
       
       ##########################################################################################################
       
       st_a_e_i_prev = [[a_prev], [ecc_prev], [i_prev]]
       st_a_e_i_new = [[a_new], [ecc_new], [i_new]]
       st_a_e_i_target = [[a_target], [ecc_target], [i_target]]
       
       exp_value_t_aei      =  0 
       exp_value_t_plus_1_aei =  0 
       
       # r = -w abs(s_t-s_tar) + SUM_i [c_i e^{-w' abs(s_t - s_tar)} ]
       for i in range(0,3):
          exp_value_t_aei        =  exp_value_t_aei        + ( c1_aei[i] * math.exp(-(w1_aei_[i] * abs(np.subtract(st_a_e_i_prev[i], st_a_e_i_target[i])) ) ) )
          exp_value_t_plus_1_aei =  exp_value_t_plus_1_aei + ( c1_aei[i] * math.exp(-(w1_aei_[i] * abs(np.subtract(st_a_e_i_new[i], st_a_e_i_target[i])) ) ) )
       
       phi_st_aei        = - np.dot( np.transpose(np.array(w1_aei)) , abs(np.subtract(st_a_e_i_prev ,st_a_e_i_target))) + exp_value_t_aei       -(0.03*monitor_flag)
       phi_st_plus_1_aei = - np.dot( np.transpose(np.array(w1_aei)) , abs(np.subtract(st_a_e_i_new ,st_a_e_i_target))) + exp_value_t_plus_1_aei -(0.03*monitor_flag)
       
       if self.only_a_conv:
           exp_value_t_aei        =  0 
           exp_value_t_plus_1_aei =  0 
           exp_value_t_aei        =  exp_value_t_aei        + ( c1_aei[0] * math.exp(-(w1_aei_[0] * abs(np.subtract(st_a_e_i_prev[0], st_a_e_i_target[0])) ) ) )
           exp_value_t_plus_1_aei =  exp_value_t_plus_1_aei + ( c1_aei[0] * math.exp(-(w1_aei_[0] * abs(np.subtract(st_a_e_i_new[0], st_a_e_i_target[0])) ) ) )
           phi_st_aei        = - np.dot(w1_aei[0] , abs(np.subtract(st_a_e_i_prev[0] ,st_a_e_i_target[0]))) + exp_value_t_aei       -(0.03*monitor_flag)
           phi_st_plus_1_aei = - np.dot(w1_aei[0] , abs(np.subtract(st_a_e_i_new[0] ,st_a_e_i_target[0]))) + exp_value_t_plus_1_aei -(0.03*monitor_flag)
           phi_st_aei = np.array([[phi_st_aei]])
           phi_st_plus_1_aei = np.array([[phi_st_plus_1_aei]])
       
       if self.test==1: 
            phi_st_aei = phi_st_aei / 100000
            phi_st_plus_1_aei = phi_st_plus_1_aei / 100000
            
       reward_t_aei_1 = phi_st_plus_1_aei  - phi_st_aei - tauu_aei + (self.done_ep_reward * done) - (self.rf *redflag) 
       reward_t_aei = reward_t_aei_1[0] + (flag_ms_1*self.ms_reward)
            
       if self.test==3: 
            # Calculate differences between achieved and desired goals
            a_diff = np.abs(float(round(st_a_e_i_prev[0][0], 6)) - float(round(st_a_e_i_target[0][0], 6)))
            e_diff = np.abs(float(round(st_a_e_i_prev[1][0], 6)) - float(round(st_a_e_i_target[1][0], 6)))
            i_diff = np.abs(float(round(st_a_e_i_prev[2][0], 6)) - float(round(st_a_e_i_target[2][0], 6)))

            # Calculate rewards for each dimension
            a_reward = 100 * a_diff * (100 * np.exp(-30 * a_diff))
            e_reward = 76 * e_diff * (20 * np.exp(-40 * e_diff))
            i_reward = 50 * i_diff * (50 * np.exp(-40 * i_diff))
            
            reward_t_aei  = -(self.weights["w1"]["a"] *a_diff  + self.weights["w1"]["e"] *e_diff   + self.weights["w1"]["i"] *i_diff) \
                           - tauu_aei + (self.done_ep_reward * done) - (self.rf*redflag) + (flag_ms_1*self.ms_reward)
            
            reward_t_aei  =  np.array([reward_t_aei])
            # a_reward = self.weights["w1"]["a"] * a_diff + (self.weights["c1"]["a"] * np.exp(-self.weights["w1_"]["a_"] * a_diff))
            # e_reward = self.weights["w1"]["e"] * e_diff + (self.weights["c1"]["e"] * np.exp(-self.weights["w1_"]["e_"] * e_diff))
            # i_reward = self.weights["w1"]["i"] * i_diff + (self.weights["c1"]["i"] * np.exp(-self.weights["w1_"]["i_"] * i_diff))
            
            # reward0 = -(a_reward + e_reward + i_reward) 
          
            
            # diff     =   self.weights["w1"]["a"] * a_diff  + self.weights["w1"]["e"] * e_diff +  self.weights["w1"]["i"] * i_diff
            # diff_exp =   self.weights["c1"]["a"] * np.exp(-self.weights["w1_"]["a_"] * a_diff) + \
            #              self.weights["c1"]["e"] * np.exp(-self.weights["w1_"]["e_"] * e_diff) + \
            #              self.weights["c1"]["i"] * np.exp(-self.weights["w1_"]["i_"] * i_diff)
            # reward1   =  -(diff + diff_exp ) 
      
      ########################################################################################################## 
    #    exp_value_t_aei = 0
    #    c1_aei  = [1 , 0.2, 0.8]
    #    w1_aei  = [10,10,10]
    #    tauu_aei = 1.5
       
    #    for i in range(0,3):
    #       reward_1              = c1_aei[i] *  (1-abs(np.subtract(st_a_e_i_new[i], st_a_e_i_target[i])) ** 2) 
    #       reward_2              = math.exp(-(w1_aei[i] * abs(np.subtract(st_a_e_i_new[i], st_a_e_i_target[i]))))
    #       exp_value_t_aei       =  exp_value_t_aei + reward_1 + reward_2 - (0.03*monitor_flag)
       

    #    reward_t_aei = exp_value_t_aei - tauu_aei + (100 * done) - (30*redflag) 
    
     ###########  Visulizating Weights ############################################################################# 
    
       a_plot= -self.weights["w1"]["a"] * abs(np.subtract(a_prev ,a_target))
       e_plot= -self.weights["w1"]["e"] * abs(np.subtract(ecc_prev ,ecc_target))
       i_plot= -self.weights["w1"]["i"] * abs(np.subtract(i_prev ,i_target))
    
       a_exp_plot=  self.weights["c1"]["a"] * math.exp(-(self.weights["w1_"]["a_"] * abs(np.subtract(a_prev ,a_target)) ))
       e_exp_plot=  self.weights["c1"]["e"] * math.exp(-(self.weights["w1_"]["e_"] * abs(np.subtract(ecc_prev ,ecc_target)) ))
       i_exp_plot=  self.weights["c1"]["i"] * math.exp(-(self.weights["w1_"]["i_"] * abs(np.subtract(i_prev ,i_target)) ))
       
       a_total_plot =  a_plot + a_exp_plot
       e_total_plot =  e_plot + e_exp_plot
       i_total_plot =  i_plot + i_exp_plot
       
       sum_aei_plot = a_total_plot + e_total_plot + i_total_plot
       reward_st1  = phi_st_plus_1_aei[0][0]
       
       if self.test==2 or self.test==1: 
            sum_aei_plot = sum_aei_plot / 100000
                
       reward_st1_minus_st = reward_st1 - sum_aei_plot
       reward_st1_minus_st_minus_tau  = reward_st1_minus_st  - tauu_aei
       reward_st1_minus_st_minus_tau_minus_100redflag = reward_st1_minus_st_minus_tau - (self.rf*redflag) 
       reward_ms = reward_st1_minus_st_minus_tau_minus_100redflag + (flag_ms_1*self.ms_reward)
       
       if self.test==3: 
            a_plot,e_plot,i_plot, a_exp_plot,e_exp_plot,i_exp_plot = 0,0,0,0,0,0  
            reward_st1,reward_st1_minus_st = 0,0
            a_total_plot  = -(self.weights["w1"]["a"] *a_diff)  
            e_total_plot  = -(self.weights["w1"]["e"] *e_diff)  
            i_total_plot  = -(self.weights["w1"]["i"] *i_diff) 
            sum_aei_plot  = a_total_plot + e_total_plot + i_total_plot 
            reward_st1_minus_st_minus_tau = sum_aei_plot - tauu_aei 
            reward_st1_minus_st_minus_tau_minus_100redflag =   reward_st1_minus_st_minus_tau  - (self.rf*redflag) + (flag_ms_1*self.ms_reward)
            reward_ms     =  reward_st1_minus_st_minus_tau_minus_100redflag + (self.done_ep_reward * done) 
       
       if self.test==2: 
            reward_ms = reward_st1 - tauu_aei - (self.rf*redflag)  + (flag_ms_1*self.ms_reward)
            
       if self.only_a_conv:
            reward_ms = reward_t_aei[0]
       
       reward_plot = [a_plot,e_plot,i_plot, a_exp_plot,e_exp_plot,i_exp_plot, a_total_plot,e_total_plot,i_total_plot, sum_aei_plot,reward_st1,reward_st1_minus_st,reward_st1_minus_st_minus_tau,reward_st1_minus_st_minus_tau_minus_100redflag, reward_ms]
    
     ########################################################################################################## 
     
       return reward_t_aei, ecc_new,i_new,a_new,  ecc_target,i_target,a_target, reward_plot 
   
 
    
    def writing_Successful_episodes( self,success_ep_counter, episode, len_episode, score ,time_in_days, a_last, Inc_last, ecc_last, mass_last, h_last, hx_last,hy_last,ex_last,ey_last, RAAN, argp, nurev,   completeName_successful  ):
        self.temp = ['Succ_ep_counter : ', success_ep_counter, '    ', '    ',
             'ep : ', episode, '    ', '    ', 'ep_length : ', len_episode, '    ', '    ',
             'score: ', score, '    ', '    ', 'time(days): ', time_in_days,'    ', '    ', 
             'targ-a[-1]: ', a_last, '    ', '    ',
             'inc[-1]: ', Inc_last, '    ', '    ', 'ecc[-1]: ', ecc_last, '    ', '    ', 'mass[-1]: ', mass_last, '    ', '    ', 
             'h[-1]: ', h_last, '    ', '    ', 'hx[-1]: ', hx_last, '    ', '    ',
             'hy[-1]: ', hy_last, '    ', '    ', 'ex[-1]: ', ex_last, '    ', '    ',
             'ey[-1]: ', ey_last, '    ', '    ', 'RAAN: ', RAAN, '    ', '    ', 'argp: ', argp, '    ', '    ', 'nurev: ', nurev]
        with open(completeName_successful, 'a') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(self.temp)
            csvfile.close()
            
    def writing_info_file( self,initial_state,target_state , scaled_state, xtargetscaled, w1_a,w1_e,w1_i,w1_a_,w1_e_,w1_i_,c_a,c_e,c_i, tau,normalization_case ,sh_command_nu, mass, seg, force, tol_inc, tol_ecc, tol_a, a_boundary_flag, inc_boundary_flag, ecc_boundary_flag, a_min,a_max,inc_min,inc_max,ecc_min,ecc_max,sh_flag, target_a,target_ecc,target_inc,   tol_a_low,tol_a_high,    tol_ecc_low,tol_ecc_high,      tol_inc_low,tol_inc_high, manual_weight, intermediate_state_use, RL_algo,done_ep_reward,pre_trained_weights_flag,pre_trained_weights_path, milestone, ecc_ms_1_tol_low,ecc_ms_1_tol_high,  a_ms_1_tol_low,a_ms_1_tol_high,  inc_ms_1_tol_low,inc_ms_1_tol_high,ms_reward,  he_para_flag,   a_conv,ecc_conv,inc_conv,    h_conv,hx_conv,hy_conv,ex_conv,ey_conv,     RAAN_conv,argp_conv,   simpl_or_exp_rew, tol_h,tol_hx , tol_hy , tol_ex ,tol_ey,tol_RAAN, tol_argp, w1_h, w1_hx, w1_hy, w1_ex,w1_ey, w1_h_, w1_hx_, w1_hy_, w1_ex_, w1_ey_, c1_h, c1_hx, c1_hy, c1_ex, c1_ey,   w1_RAAN, w1_argp,w1_RAAN_, w1_argp_,c1_RAAN, c1_argp,   w1_h_s, w1_hx_s, w1_hy_s, w1_ex_s, w1_ey_s, w1_a_s, w1_ecc_s, w1_inc_s, up_sample, max_nu_ep,  max_steps_one_ep , phi0, sac_paper_cond, new_scaling_flag, discrete_reward, w_a_dr, w_e_dr, w_i_dr, w_h_dr, w_hx_dr, w_hy_dr, w_ex_dr, w_ey_dr, w_argp_dr, w_raan_dr, completeName_successful  ):
        self.temp = [['initial_state(Lunar) {h, hx, hy, ex, ey, mass, time}:                 ', initial_state[0], '    ',initial_state[1],'    ', initial_state[2], '    ',initial_state[3],'    ', initial_state[4], '    ',initial_state[5], '    ',initial_state[6]],
                    ['target_state(S-GTO) {h, hx, hy, ex, ey, mass, time}:                   ', target_state[0], '    ',target_state[1],'    ', target_state[2], '    ',target_state[3], '    ',target_state[4], '    ',target_state[5], '    ',target_state[6]],
                    ['Scaled_state {h, hx, hy, ex, ey, mass, time}:                            ', scaled_state[0], '    ',scaled_state[1], '    ',scaled_state[2], '    ',scaled_state[3], '    ',scaled_state[4], '    ',scaled_state[5], '    ',initial_state[6]],
                    ['Scaled_target_state {h, hx, hy, ex, ey, mass, time}:                            ', xtargetscaled[0], '    ',xtargetscaled[1], '    ',xtargetscaled[2], '    ',xtargetscaled[3], '    ',xtargetscaled[4], '    ',xtargetscaled[5], '    ',target_state[6]],
                    ['weights {w1_a, w1_e, w1_i}:                                           ',   w1_a,'    ',w1_e,'    ',w1_i],
                    ['weights {w1_a_, w1_e_, w1_i_}:                                        ',   w1_a_,'    ',w1_e_,'    ',w1_i_],
                    ['weights {c_a, c_e, c_i}:                                              ',   c_a,'    ',c_e,'    ',c_i],
                    ['weights (tau):      ',   tau],
                    ['normalization_case   (1:general_norm, 0:seperate norm ):              ',normalization_case],
                    ['sh_command_nu:                                        ',sh_command_nu], 
                    ['mass(kg):                                         ',mass], 
                    ['seg(deg):                                         ',seg], 
                    ['force:                                                ',force],
                    ['tol_a, tol_ecc, tol_inc :                                      ',    tol_a,'    ',tol_ecc,'    ',tol_inc],
                    ['tol_h,tol_hx , tol_hy , tol_ex ,tol_ey,tol_RAAN, tol_argp:                                      ',    tol_h,'    ',tol_hx , '    ',tol_hy , '    ',tol_ex ,'    ',tol_ey,'    ',tol_RAAN, '    ',tol_argp],
                    ['a_boundary_flag, ecc_boundary_flag, inc_boundary_flag:         ',   a_boundary_flag,'    ', ecc_boundary_flag,'    ', inc_boundary_flag ],
                    ['a_min,a_max               ',  a_min,'    ',a_max],
                    ['ecc_min,ecc_max         ', ecc_min,'    ',ecc_max],  
                    ['inc_min,inc_max            ',  inc_min,'    ',inc_max], 
                    ['target_a, target_ecc, target_i :                                                                        ',    target_a,'    ',target_ecc,'    ',target_inc],
                    ['tol_a_low,tol_a_high  (tar_a-+tol_a) or manualy for intermediate                              ',  tol_a_low,'    ',tol_a_high ],
                    ['tol_ecc_low,tol_ecc_high  (tar_ecc-+tol_ecc) or manualy for intermediate                      ',  tol_ecc_low,'    ',tol_ecc_high ],
                    ['tol_inc_low,tol_inc_high  (tar_inc-+tol_inc) or manualy for intermediate                      ',  tol_inc_low,'    ',tol_inc_high ],
                    ['sh_flag 0:no shadow 1:shadow        ', sh_flag],
                    ['manual_weights 0:weight through args, 1:manualy in file         ', manual_weight],
                    ['intermediate_state_use :  (0:False, 1:True)                     ', intermediate_state_use],
                    ['RL_algo :                      ', RL_algo],
                    ['done_ep_reard : (positive reward value when episode is completed)             ',    done_ep_reward],
                    ['pre_trained_weights_flag : (0: no pretraining, 1 pretrained weights)             ',    pre_trained_weights_flag],
                    ['pre_trained_weights_path :            ',    pre_trained_weights_path],
                    ['MileStone_check_flag :            ',    milestone],
                    ['MS_tol_a_low,MS_tol_a_high,    MS_tol_ecc_low,MS_tol_ecc_high,    MS_tol_inc_low,MS_tol_inc_hight, :         ',     a_ms_1_tol_low,a_ms_1_tol_high,'       ',ecc_ms_1_tol_low,ecc_ms_1_tol_high,'      ',inc_ms_1_tol_low,inc_ms_1_tol_high ],
                    ['MileStone_reward_when_inside_milestone :            ',    ms_reward],
                    ['Hx parameters program, new reward (1) or old reward(0) :            ',    he_para_flag],
                    ['para included in reward/terminal    [a, e, i,   h, hx, hy, ex, ey, RAAN, argp ]:            [  ', a_conv,' ',ecc_conv,' ',inc_conv,'        ',h_conv,' ',hx_conv,' ',hy_conv,' ',ex_conv,' ',ey_conv,' ',RAAN_conv,' ',argp_conv,'  ]'  ],
                    ['Simple reward(0) / complex reward (1) :            ',    simpl_or_exp_rew],
                    ['weights {w1_h, w1_hx, w1_hy, w1_ex, w1_ey,w1_RAAN,w1_argp }:                                           ',   w1_h,'    ',w1_hx,'    ',w1_hy, '    ',w1_ex,'   ', w1_ey, '    ',w1_RAAN,'   ', w1_argp  ],
                    ['weights {w1_h_, w1_hx_, w1_hy_, w1_ex_, w1_ey_,w1_RAAN_,w1_argp_ }:                                      ',   w1_h_,'    ',w1_hx_,'    ',w1_hy_, '    ',w1_ex_,'   ', w1_ey_, '    ',w1_RAAN_,'   ', w1_argp_],
                    ['weights {c1_h, c1_hx, c1_hy, c1_ex, c1_ey,c1_RAAN,c1_argp }:                                           ',   c1_h,'    ',c1_hx,'    ',c1_hy, '    ',c1_ex,'   ', c1_ey, '    ',c1_RAAN,'   ', c1_argp],
                    ['weights for simple reward {w1_h_s, w1_hx_s, w1_hy_s, w1_ex_s, w1_ey_s, w1_a_s, w1_ecc_s, w1_inc_s, up_sampl }:       ', w1_h_s,'  ', w1_hx_s,'  ', w1_hy_s,'  ', w1_ex_s,'  ', w1_ey_s,'  ', w1_a_s,'  ', w1_ecc_s,'  ', w1_inc_s,'  ',up_sample],
                    ['max_nu_ep,  max_steps_one_ep            :',  max_nu_ep,'    ',max_steps_one_ep], 
                    ['phi0                                    :',  phi0],
                    ['sac_paper_cond                                    :',  sac_paper_cond],
                    ['new_scaling_flag   0 no scaling, 1 new scaling       :',  new_scaling_flag],
                    ['discrete_reward    0 other reward, 1 discrete reward       :',  discrete_reward],
                    ['weights {w_a_dr, w_e_dr, w_i_dr, w_h_dr, w_hx_dr, w_hy_dr, w_ex_dr, w_ey_dr, w_argp_dr, w_raan_dr  }:             ', w_a_dr, ' ',w_e_dr,' ', w_i_dr,' ', w_h_dr,' ', w_hx_dr,' ', w_hy_dr,' ', w_ex_dr,' ', w_ey_dr,' ', w_argp_dr,' ', w_raan_dr  ],
                    ['                                                                                                                              '],
                    ['                                                                                                                              '],
                    ['                                                                                                                              '],
                    ['                                                                                                                              ']
                    ]

        with open(completeName_successful, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            for row in self.temp:
                csvwriter.writerow(row)
            
            csvfile.close()

        
        
    def writing_all_episodes_data( self, episode, ep_fixed_len_counter,  ho,hxo,hyo,exo,eyo,masso,   hsc,hxsc,hysc,exsc,eysc,masssc,   ND_h,ND_hx,ND_hy,ND_ex,ND_ey,ND_mass ,  ecc,a,inc, time_days,time, phi, diff_a,diff_ecc,diff_inc, diff_h,diff_hx,diff_hy,diff_ex,diff_ey,    alpha,beta, reward_step, score,  flag_ecc,flag_a,flag_inc,segment,     r_a,r_e,r_i,r_a_exp,r_e_exp,r_i_exp, r_a_sum,r_e_sum,  r_i_sum, r_st, r_st1, r_st1_m_st,r_st1_m_st_m_tau,r_st1_m_st_m_tau_100rf, r_st1_m_st_m_tau_100rf_p_ms, force,  RAAN, argp, nurev,completeName_successful  ):
        self.temp = ['ep : ', episode, '    ', 'ep_step : ', ep_fixed_len_counter, '    ', 'orignalstate : ', ho, hxo, hyo, exo, eyo, masso, '    ', 'ecc,a,inc:', ecc, a, inc, '    ', 'Reward,score : ', reward_step, score, '    ', '    ',
                     'Normalized_State: ', ND_h, ND_hx, ND_hy, ND_ex, ND_ey, ND_mass, '    ',  'Scaled_State: ', hsc,hxsc,hysc,exsc,eysc,masssc,  '    ',  '    ','time(days) : ',time_days , ' ',time,'    ', 'Phi: ',phi ,
                     '    ', 'action_values : ', alpha, beta, '    ', '    ', 'Mass : ', masso, '    ', '    ', 'flag_ecc : ', flag_ecc, '    ',
                     'flag_a : ', flag_a, '    ', 'flag_inc : ', flag_inc, '    ', 'Segment : ', segment, 
                     'diff_a : ', diff_a, '    ', 'diff_inc : ', diff_inc, '    ', 'diff_ecc : ', diff_ecc, 
                     'diff_h : ', diff_h, '    ', 'diff_hx_hy : ', diff_hx,' ',diff_hy, '    ', 'diff_ex_ey : ', diff_ex,' ',diff_ey,   
                     '    ', 'force: ', force  , 'RAAN: ', RAAN, '    ',  'argp: ', argp, '    ',  'nurev: ', nurev]      
                    #  'r_a : ', r_a, '    ', 'r_e : ', r_e,  '    ', 'r_i : ', r_i,   '    ',     'r_a_exp : ', r_a_exp, '    ', 'r_e_exp : ', r_e_exp,  '    ', 'r_i_exp : ', r_i_exp, '    ',
                    #  'r_a_sum : ', r_a_sum, '    ', 'r_e_sum : ', r_e_sum,  '    ', 'r_i_sum : ', r_i_sum,   '    ',     'r_st : ', r_st, '    ', 'r_st1 : ', r_st1,  '    ', 'r_st1-r_st : ', r_st1_m_st, '    ',
                    #  'r_st1-r_st-tau: ', r_st1_m_st_m_tau, '    ', 'r_st1-r_st-tau-100redflag : ', r_st1_m_st_m_tau_100rf,  '    ', 'r_st1-r_st-tau-100redflag+ms : ', r_st1_m_st_m_tau_100rf_p_ms,
                    
        with open(completeName_successful , 'a') as csvfile: 
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(self.temp)      
            csvfile.close()     

                
    def writing_final_states ( self, h,hx,hy,ex,ey,phi,time,mass,  RAAN, argp, nurev, completeName_1 ):
        self.temp = []
        self.temp = np.append(self.temp, h)                                                           
        self.temp = np.append(self.temp, hx )                      
        self.temp = np.append(self.temp, hy)        
        self.temp = np.append(self.temp, ex)   
        self.temp = np.append(self.temp, ey)           #   [h;hx;hy;ex;ey;phi;time;fuel_burnt]  
        self.temp = np.append(self.temp, phi)   
        self.temp = np.append(self.temp, time)  
        self.temp = np.append(self.temp, mass) 
        self.temp = np.append(self.temp, RAAN)   
        self.temp = np.append(self.temp, argp)  
        self.temp = np.append(self.temp, nurev) 
        with open(completeName_1 , 'a') as csvfile: 
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(self.temp)
            csvfile.close()
    
    def plot_variable(self, name, hist, folder_path, ep_counter, flag_ter_values= None,  tsp=None, tsp_indexes=None, all_episode_plot_flag=None, flag_saving_with_no_ep_nu =None, plotting_time=0, flag_ms_1=0, flag_ep_nu_start=None , h_flag=None, tol_h=None):
        plt.figure(figsize=(12, 8))
        if all_episode_plot_flag == 1:
            if hist[0] == []:
                plt.plot(hist[1:-2])   
            else:
                plt.plot(hist[0:-2]) 
        elif plotting_time == 1:
             plt.plot(hist) 
        else:
             plt.plot(hist[ep_counter-1])     
        plt.title(f'state parameters values {name}')
        plt.ylabel(f'{name.lower()} values')
        if flag_ter_values == 1:
            plt.axhline(tsp[tsp_indexes[0]], color='r', linestyle='-')
            plt.axhline(tsp[tsp_indexes[1]], color='r', linestyle='-')
        
        if h_flag == 1:
            h_1_t, h_2_t = self.target_state[0] ,  self.target_state[0] + tol_h   #h = 101055.56404
            plt.axhline(h_1_t, color='b', linestyle='--')
            plt.axhline(h_2_t, color='b', linestyle='--')
               
        if flag_ms_1 == 1:    
            plt.axhline(tsp[tsp_indexes[2]], color='g', linestyle='--')
            plt.axhline(tsp[tsp_indexes[3]], color='g', linestyle='--')
            
        plt.grid(True)
        
        if flag_saving_with_no_ep_nu  == 1:
            plt.savefig(folder_path+"/_"+name + ".png")
        elif flag_ep_nu_start == 1:
            plt.savefig(folder_path+"/ep_"+str(ep_counter-1)+ name + ".png")
        else:
            plt.savefig(folder_path+"_ep_"+str(ep_counter-1) + ".png")
            
        plt.close()
    
    def plot_two_variable(self, name, a,b, hist_1, hist_2, folder_path, ep_counter, flag_ter_values= None,  tsp=None, tsp_indexes=None , flag_saving_with_no_ep_nu =None, flag_ep_nu_start=None, hx_hy_flag=None , ex_ey_flag=None, tolhx=None,tolhy=None, tolex=None,toley=None, scaled=None, target_ex=None, target_ey=None):
        plt.figure(figsize=(12, 8))
        plt.plot(hist_1[ep_counter-1], c='b', label=a, linewidth=1.5)     
        plt.plot(hist_2[ep_counter-1], c='r', label=b, linewidth=1.5)
        plt.legend()
        plt.grid(True)
        plt.title(f'state parameters values {name}')
        plt.ylabel(f'{name.lower()} values')
        if flag_ter_values == 1:
            plt.axhline(tsp[tsp_indexes[0]], color='r', linestyle='-')
            plt.axhline(tsp[tsp_indexes[1]], color='r', linestyle='-')
            
        if hx_hy_flag == 1:
            hx_1_t, hx_2_t = self.target_state[1] - tolhx, self.target_state[1] + tolhx    # hx = 8993.12420 
            hy_1_t, hy_2_t = self.target_state[2] - tolhy , self.target_state[2] + tolhy  # hy = -44988.20968 
            plt.axhline(hx_1_t, color='b', linestyle='--')
            plt.axhline(hx_2_t, color='b', linestyle='--')
            plt.axhline(hy_1_t, color='r', linestyle='--')
            plt.axhline(hy_2_t, color='r', linestyle='--')
            
        if ex_ey_flag == 1:
            if scaled:
                ex_1_t, ex_2_t =  target_ex- tolex ,  target_ex+ tolex
                ey_1_t, ey_2_t = target_ey - toley  , target_ey+ toley
            else:
                ex_1_t, ex_2_t = self.target_state[3] - tolex , self.target_state[3] + tolex #  ex=0.634234170
                ey_1_t, ey_2_t = self.target_state[4] - toley  , self.target_state[4] + toley  #ey = 0.142292050
            plt.axhline(ex_1_t, color='b', linestyle='--')
            plt.axhline(ex_2_t, color='b', linestyle='--')
            plt.axhline(ey_1_t, color='r', linestyle='--')
            plt.axhline(ey_2_t, color='r', linestyle='--')
           
            
        if flag_saving_with_no_ep_nu  == 1:
            plt.savefig(folder_path+"/_"+name + ".png")
        elif flag_ep_nu_start == 1:
            plt.savefig(folder_path+"/ep_"+str(ep_counter-1)+ name + ".png")
        else:
            plt.savefig(folder_path+"_ep_"+str(ep_counter-1) + ".png")
            
        plt.close()
