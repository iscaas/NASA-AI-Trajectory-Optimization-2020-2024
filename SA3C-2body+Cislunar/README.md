# Single-Agent Attention Actor-Critic (SA3C): A Novel Solution for Low-Thrust Spacecraft Trajectory Optimization
We provide the code repository for our paper. This repository includes the necessary code to replicate our experiments and utilize our SA3C-DRL model for spacecraft trajectory planning. By accessing the repository, researchers and practitioners can benefit from our approach to efficiently transfer spacecraft to GEO and CisLunar Transfer Points (NRHO and Patch-Point).


SA3C based Super-GTO to Cislunar Patchpoint transfer  | SA3C based Super-GTO to NRHO-L2 transfer
:-: | :-:
![SA3C based S-GTO to PatchPoint transfer](/readmeplots/PP_to_SuperGEO.gif)  | ![SA3C based S-GTO to NRHO transfer](/readmeplots/NRHO3dvid.gif) 




SA3C based GTO to GEO transfer  | SA3C based Super-GTO to GEO transfer
:-: | :-:
![SA3C based GTO to GEO transfer](/readmeplots/GTO-GEO.gif)  | ![SA3C based S-GTO to GEO transfer](/readmeplots/SuperGTO-GEO.gif) 

## Appendix
For comprehensive implementation details including all modeling assumptions and algorithm parameter settings, please check the attached Appendix.pdf in files. 

## Files Description
#### Circular2body Folder
Contains code, plots, and final optimized weights for GEO-targeted scenarios ie from GTO-1, GTO-2 abnd from Super-GTO-1 to GEO orbit
#### CisLunar3body Folder
Contains code, plots, and final optimized weights for Super-GTO-2 to NRHO and Super-GTO-2 to Patch-point transfer scenarios.
#### Common Files
- ****config.py****: General configurations and initial parameters.
- ****scenerios.py****: Scenario-specific configurations.
- ****Spacecraft_Env.py**** & ****enviornment.py****: General RL and model-specific environment functions.
- ****SA3C_test.py****: Main file to reproduce results using trained weights.
- ****SA3C_train.py****: Train the model from scratch.
   

## Setting up Enviornment:
#### Prerequisites:
- Python >=3.7.1,<3.11
- Matlab 2021 or later
  
To run experiments locally:
```sh
git clone https://github.com/iscaas/NASA-AI-TrajectoryOpt_2020-2024.git
pip install -r requirements.txt
```

Next We need to install the MATLAB extension in above enviornment. 
For that first install MATLAB on your system (e.g., MATLAB 2021a)  and then Activate your above Conda environment, navigate to the following MATLAB installation folder and install the python setup file:
```sh
conda activate <env_name>
cd "<MATLAB_installation_folder>/R2021a/extern/engines/python"
e.g cd "C:/Program Files/MATLAB/R2021a/extern/engines/python"
python setup.py install
```
     
## Running the code:
- To train the model from scratch:
```sh
      python SA3C_train.py  --case 1 --testing_weights 0  --single_weight_test 0
```
- To reproduce the results using trained weights:
```sh
      python SA3C_test.py  --case 1  --testing_weights 1  --single_weight_test 1 
```
      
Where ****--case**** is defined to select the scenerios, as follows:
   
In Circular2body scenerios:
- ****case 1**** : GTO-1 to GEO DRL-1,                 ****case 2**** : GTO-1 to GEO DRL-2
- ****case 3**** : GTO-2 to GEO DRL-1,                 ****case 4**** : GTO-2 to GEO DRL-2
- ****case 7**** : Super-GTO-1 to GEO DRL-1,           ****case 8**** : Super-GTO-1 to GEO DRL-2
    
In Circular3body/PatchPoint scenerios:  
 - ****case 1**** : Super-GTO-2 to PatchPoint DRL-1,    ****case 2**** : Super-GTO-2 to PatchPoint DRL-2
    
In Circular3body/NRHO scenerios:  
 - ****case 1**** : Super-GTO-2 to NRHO DRL-1,          ****case 2**** : Super-GTO-2 to NRHO DRL-2

## Methodology
<image src='/readmeplots/methodology.PNG' width=1000/>

## Results
<image src='/readmeplots/results.PNG' width=1000/>
  <div align="center">
<image src='/readmeplots/ablation studies.PNG' width=500/>
  </div>
  
SA3C based 2 body transfer  | SA3C based 3 body  transfer
:-: | :-:
<image src='/readmeplots/fig7.PNG' width=500/> | <image src='/readmeplots/NRHO3d.PNG' width=500/>
<image src='/readmeplots/2bodyscores.PNG' width=500/> | <image src='/readmeplots/3bodyscores.PNG' width=500/>
<image src='/readmeplots/2bodyresultsplots.PNG' width=500/> | <image src='/readmeplots/NRHOresultsplots.PNG' width=500/>

## Citation
If you find this work beneficial to your research or project, I kindly request that you cite it:
```

```
