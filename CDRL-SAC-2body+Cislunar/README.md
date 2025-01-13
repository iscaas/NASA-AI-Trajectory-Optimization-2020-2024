# Automated Trajectory Planning: A Cascaded Deep Reinforcement Learning Approach for Low-Thrust Spacecraft Orbit-Raising
We provide the code repository for our paper This repository includes the necessary code to replicate our experiments and utilize our DRL model for spacecraft trajectory planning. By accessing the repository, researchers and practitioners can benefit from our approach to efficiently transfer spacecraft to GEO using low-thrust propulsion systems.

CDRL based GTO to GEO transfer  
<![CDRL based GTO to GEO transfer](https://github.com/talhazaidi13/Cascaded-Deep-Reinforcement-Learning-Based-Multi-Revolution-Low-Thrust-Spacecraft-Orbit-Transfer/blob/main/paper-outputs/GTO-GEO.gif) 

## Appendix
For comprehensive implementation details including all modeling assumptions and algorithm parameter settings, please check the attached Appendix.pdf in files. 

## Files Description

- `config.py`: Contains the configurations or initial parameters to run the code.

- `Scenarios.py`: Contains the parameters for six transfer cases, which are as follows:
    - GTO-1 to GEO_1st network
    - GTO-1 to GEO_2nd network
    - GTO-2 to GEO_1st network
    - GTO-2 to GEO_2nd network
    - Super-GTO to GEO_1st network
    - Super-GTO to GEO_2nd network

- `Spacecraft_env.py`: Contains the gym structured environment, which includes `env.reset()`, `env.step()`, and `env.render()` functions.

- `environment.py`: Contains custom environment functions. These functions in `environment.py` are called by the gym environment in `Spacecraft_env.py`.

- `spacecraftEnivironment.m`: A MATLAB code used to calculate `env.step()` values.

- `environment.yml`: Contains all the required commands to recreate the Conda environment in any local system. This will recreate the exact same environment in which we trained our algorithms.

- `environment_info.txt`: Contains the versions of all installed packages present in our Conda environment.

- `test.py`: Python file which can be used to run the scenerios from pre trained weights.
- `test.sh`: Shell file that contains the code to run test.py file. You can select the case number and max number of episode and all other parameters which are defined in config file in here. <br>
             e.g If you want to run Case 1 i.e 'GTO-1 to GEO_1st network' then in test.sh file you will write as follow:
  ```python
  python test.py  --case 1  --max_nu_ep 100
  ```

- `train.py`: Python file which can be used to train the scenarios from scratch.
- `train.sh`: Shell file that contains the code to run train.py file. You can select the case number and max number of episode and all other parameters which are defined in config file in here. <br>
             e.g If you want to run Case 1 i.e 'GTO-1 to GEO_1st network' then in train.sh file you will write as follow:
  ```python
  python train.py  --case 1 --sh_flag 0
  ```
   Note:  Make sure that while training --sh_flag 0 
- `Final weights` folder: Contains the final trained weights for all six scenarios.
- `CSV Files` folder:  Contains Csv files which is used to communicate between Matlab and python programs data.
- `Plots` : The resulting plots from training or testing the DRL agents will be saved in plots folder.
- `Model_training_weights`: The resulting weights from training the DRL agent will be saved in Model_training_weights folder.
- `Model_training_logs`:    The resulting logs from training the DRL agent will be saved in Model_training_logs folder.

## Setting up Enviornment:


1. Install Conda:        If conda is not installed in your system then install conda. (I used 4.10.1) <br>
2. Install CUDA & CUDNN:  We installed Cuda 11.7.  Follow the instructions in  https://medium.com/geekculture/install-cuda-and-cudnn-on-windows-linux-52d1501a8805#3e72
3. create conda environment named as mat_py3.7 and install python 3.7 in it. 
   ```shell
       conda create --name mat_py3.7 python=3.7
   ```
4. Activate the environment: Use the following command to activate the environment: 
   ```shell                                        
       conda activate mat_py3.7  
   ```
   
5. Install pytorch with gpu: 
    we installed torch 2.0.1+cu117. <br>
    Please follow the instructions as follows to install torch. <br>
        Install PyTorch 2.0.1 with CUDA 11.7:<br>
   ```shell   
   pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117 
   ```        
   Verify if it installed correctly as follows:<br>
   ```shell   
   python -c "import torch; print(f'Torch Version: {torch.__version__}\nGPU Available: {torch.cuda.is_available()}')"
   ```
   It should show the version as 2.0.1+cu117 and GPU Available: True  (if there is any GPU) <br>

6. Install MATLAB: Install MATLAB on your system. (I am using MATLAB 2021a). If you dont have matlab, you can use the following link to  install MATLAB <br> https://www.mathworks.com/products/new_products/previous_release_overview.html <br>
   Navigate to the MATLAB folder: In the activated Conda environment, go to the MATLAB folder by running the following command:
   ```shell   
       cd "<MATLAB_installation_folder>"  
   ```
   Replace <MATLAB_installation_folder> with the path to your MATLAB installation folder. By default, the MATLAB folder is located at "C:/Program Files/MATLAB". Make sure to include the double quotes if the path contains spaces.
   Go to the MATLAB Engine Python folder: Change the directory to the MATLAB Engine Python folder by running the following command:
   ```shell
       cd "R2021a\extern\engines\python"  
   ```
   This will navigate you to the relevant folder containing the MATLAB Engine Python setup file.
   Install the MATLAB Engine: To install the MATLAB Engine in your Conda environment, execute the setup.py file by running the following command:
   ```shell
       python setup.py install  
   ```
   if this doesnt install directly then open setup.py file and in __main__ replace the version as version="0.1.0" and then save it. 
   after that just run the command in command line "pip install ."
   
   This command will install the MATLAB Engine package in your Conda environment.
    Verify the installation: To check if the MATLAB Engine is installed correctly, run the following command:
   ```shell
       python -c "import matlab.engine" 
   ```
   
8. Install git:   - If git is not installed in system then install git ( I used 2.40.0.windows.1)<br>
                    - Download git from  https://git-scm.com/downloads and install it. <br>
                    - While installing: On the "Adjusting your PATH environment" page, select the option "Git from the command line and also from 3rd-party software." This ensures that Git is added to your system's PATH 
                          Environment variable, allowing you to use Git from the command prompt or terminal.<br>
                        - Verify the installation:  After the installation is complete, open a new command prompt or terminal window and run the following command to verify the Git version:
   ```shell
       git --version
   ```

## Running the code:

- Finally, you can run the test.sh or train.sh file  for testing with trained weights and training from the scratch, using the bash command:
```shell
bash test.sh
bash train.sh
```

## Results
CDRL based GTO to GEO transfer  | CDRL based Super-GTO to GEO transfer
:-: | :-:
<image src='/paper-outputs/fig7.PNG' width=500/> | <image src='/paper-outputs/fig13.PNG' width=500/>
<image src='/paper-outputs/tab3.PNG' width=500/> | <image src='/paper-outputs/tab6.PNG' width=500/>
<image src='/paper-outputs/fig5.PNG' width=500/> | <image src='/paper-outputs/fig10.PNG' width=500/>
<image src='/paper-outputs/fig8.PNG' width=500/> | <image src='/paper-outputs/fig11.PNG' width=500/>
<image src='/paper-outputs/fig6.PNG' width=500/> | <image src='/paper-outputs/fig12.PNG' width=500/>

## Citation
If you find this work beneficial to your research or project, I kindly request that you cite it:

