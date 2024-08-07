# Cascaded-Deep-Reinforcement-Learning (CDRL) Based Multi-Revolution Low-Thrust-Spacecraft Orbit Transfer
We provide the code repository for our paper This repository includes the necessary code to replicate our experiments and utilize our DRL model for spacecraft trajectory planning. By accessing the repository, researchers and practitioners can benefit from our approach to efficiently transfer spacecraft to GEO using low-thrust propulsion systems.
https://ieeexplore.ieee.org/abstract/document/10207710

CDRL based GTO to GEO transfer  | CDRL based Super-GTO to GEO transfer
:-: | :-:
<![CDRL based GTO to GEO transfer](/CDRL-SAC-2Body/paper-outputs/GTO-GEO.gif) > | <![CDRL based GTO to GEO transfer](/CDRL-SAC-2Body/paper-outputs/SuperGTO-GEO.gif) >


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
5. Install tensorflow-gpu 2.7.0 as follows:
   ```shell
    pip uninstall tensorflow-intel
    pip install tensorflow-gpu==2.7.0
   ```
   if there are any errors appeared while installing then try to remove those errors first e.g as follows:<br>
   ```shell
   pip install array-record dm-tree etils[enp,epath]>=0.9.0 promise tensorflow-metadata toml
   ```
   Verify its installation by checking its version with GPU as follows:<br>
   ```shell
   python -c "import tensorflow as tf; print(tf.config.experimental.list_physical_devices('GPU'))"
   ```
   it should return the GPU details like [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')] etc<br>
   
6. Install pytorch with gpu: 
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

5. Install MATLAB: Install MATLAB on your system. (I am using MATLAB 2021a). If you dont have matlab, you can use the following link to  install MATLAB <br> https://www.mathworks.com/products/new_products/previous_release_overview.html <br>
6. Navigate to the MATLAB folder: In the activated Conda environment, go to the MATLAB folder by running the following command:
   ```shell   
       cd "<MATLAB_installation_folder>"  
   ```
   Replace <MATLAB_installation_folder> with the path to your MATLAB installation folder. By default, the MATLAB folder is located at "C:/Program Files/MATLAB". Make sure to include the double quotes if the path contains spaces.
7. Go to the MATLAB Engine Python folder: Change the directory to the MATLAB Engine Python folder by running the following command:
   ```shell
       cd "R2021a\extern\engines\python"  
   ```
   This will navigate you to the relevant folder containing the MATLAB Engine Python setup file.
8. Install the MATLAB Engine: To install the MATLAB Engine in your Conda environment, execute the setup.py file by running the following command:
   ```shell
       python setup.py install  
   ```
   if this doesnt install directly then open setup.py file and in __main__ replace the version as version="0.1.0" and then save it. 
   after that just run the command in command line "pip install ."
   
   This command will install the MATLAB Engine package in your Conda environment.
10. Verify the installation: To check if the MATLAB Engine is installed correctly, run the following command:
   ```shell
       python -c "import matlab.engine" 
   ```
   
11. Install git:   - If git is not installed in system then install git ( I used 2.40.0.windows.1)<br>
                    - Download git from  https://git-scm.com/downloads and install it. <br>
                    - While installing: On the "Adjusting your PATH environment" page, select the option "Git from the command line and also from 3rd-party software." This ensures that Git is added to your system's PATH 
                          Environment variable, allowing you to use Git from the command prompt or terminal.<br>
                        - Verify the installation:  After the installation is complete, open a new command prompt or terminal window and run the following command to verify the Git version:
   ```shell
       git --version
   ```
11. Clone the repository: Run the following command in your command prompt or terminal to clone the GitHub repository to your local system:

  ```shell
       git clone https://github.com/talhazaidi13/Cascaded-Deep-Reinforcement-Learning-Based-Multi-Revolution-Low-Thrust-Spacecraft-Orbit-Transfer.git
  ```
   Alternatively, you can just download the code files from the above link. 
            
12. Navigate to the project directory:  Navigate to the project directory on your local system, which contains the cloned repository. In that folder you will find the environment.yml file. you can use the cd command 
                                       to navigate to the folder. e.g if environment.yml is at the location of D:\project then
   ```shell   
       cd "D:\project"
   ```
13. Update conda environment: Update the Conda environment using the environment.yml file. use the following code: <br>
   ```shell
       conda env update -f environment.yml  
   ```
   This command will update the Conda environment based on the specifications in the environment.yml file. <br>
14. Activate the environment: Use the following command to activate the environment: 
   ```shell                                        
       conda activate mat_py3.7  
   ```
   Please note that the name mat_py3.7 is the name of environment specified in the enviornment.yml file. You can rename it according to you.  <br>

## Running the code:

Before running the code, please ensure that you have set the paths for the CSV files. The CSV files serve as the communication link between the MATLAB environment's step function and the Python code. Without correctly setting up the paths, the state values will not be updated.

- To set the paths for the CSV files, follow these steps:

1. Open the Mat_env.m file.
2. Locate lines #126 and #184 in the Mat_env.m file.
3. In those lines, modify the path for the csvlist.dat file to match the location of the file on your system. <br>
   For example, if the location of the csvlist.dat file on your system is D:/Cascaded-DRL/csv_files/csvlist.dat, update the lines as follows:
   ```python
   M = csvread('D:/Cascaded-DRL/csv_files/csvlist.dat')
   ```
  Replace D:/Cascaded-DRL/csv_files/csvlist.dat with the actual path to the csvlist.dat file on your system.

-  Open Git Bash from the Start menu and Activate your Conda environment by running the appropriate command. For example, if your Conda environment is named "mat_py3.7," you can use the following command:
 ```shell
conda activate mat_py3.7
```

- Change the current directory to the folder containing the test.sh or train.sh file using the cd command. For example, if the test.sh file is located D:/Cascaded-DRL, you can use the following command:
```shell
cd "D:/Cascaded-DRL"
```
- Finally, you can run the test.sh or train.sh file  for testing with trained weights and training from the scratch, using the bash command:
```shell
bash test.sh
bash train.sh
```

## Results
CDRL based GTO to GEO transfer  | CDRL based Super-GTO to GEO transfer
:-: | :-:
<image src='/CDRL-SAC-2Body/paper-outputs/fig7.PNG' width=500/> | <image src='/CDRL-SAC-2Body/paper-outputs/fig13.PNG' width=500/>
<image src='/CDRL-SAC-2Body/paper-outputs/tab3.PNG' width=500/> | <image src='/CDRL-SAC-2Body/paper-outputs/tab6.PNG' width=500/>
<image src='/CDRL-SAC-2Body/paper-outputs/fig5.PNG' width=500/> | <image src='/CDRL-SAC-2Body/paper-outputs/fig10.PNG' width=500/>
<image src='/CDRL-SAC-2Body/paper-outputs/fig8.PNG' width=500/> | <image src='/CDRL-SAC-2Body/paper-outputs/fig11.PNG' width=500/>
<image src='/CDRL-SAC-2Body/paper-outputs/fig6.PNG' width=500/> | <image src='/CDRL-SAC-2Body/paper-outputs/fig12.PNG' width=500/>

## Citation
If you find this work beneficial to your research or project, I kindly request that you cite it:
```
@article{zaidi2023cascaded,
  title={Cascaded Deep Reinforcement Learning-Based Multi-Revolution Low-Thrust Spacecraft Orbit-Transfer},
  author={Zaidi, Syed Muhammad Talha and Chadalavada, Pardhasai and Ullah, Hayat and Munir, Arslan and Dutta, Atri},
  journal={IEEE Access},
  year={2023},
  publisher={IEEE}
}
```
